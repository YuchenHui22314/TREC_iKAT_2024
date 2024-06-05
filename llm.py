import re
import torch
import copy
import time
import random
import numpy as np
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    logging
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Mapping, Tuple, List, Optional, Union
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.get_logger(__name__)

# Specify the custom cache directory
cache_dir = "/data/rech/huiyuche/huggingface"

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",cache_dir = cache_dir)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir = cache_dir)


# tokenizer = AutoTokenizer.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2", cache_dir=cache_dir
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.2", cache_dir=cache_dir
# )


class LM(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        padding_side="left",
        dtype="bf16",
        device_map=None,
        use_flash_attention_2=False,
        access_token=None,
        cache_dir=cache_dir,
        accelerator: Accelerator = None,
    ) -> None:
        super().__init__()

        logger.info(f"loading tokenizer from {tokenizer_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=cache_dir,
            padding_side=padding_side,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                pad_token = "<|endoftext|>"
            else:
                pad_token = tokenizer.eos_token
            tokenizer.pad_token = pad_token

        if dtype == "bf16":
            dtype = torch.bfloat16
        elif dtype == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        if device_map is None:
            if accelerator is not None:
                device_map = {"": accelerator.device}
            else:
                device_map = {"": "cpu"}

        logger.info(f"loading model from {model_name_or_path}...")
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if config.is_encoder_decoder:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
                token=access_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=device_map,
                use_flash_attention_2=use_flash_attention_2,
                token=access_token,
            )

        self.config = model.config
        self.tokenizer = tokenizer

        if accelerator is not None:
            self.model = accelerator.prepare_model(
                model, device_placement=True, evaluation_mode=True
            )
        else:
            self.model = model

        self.rng = np.random.default_rng(42)
        self.eval()

 
    @torch.no_grad()
    def generate(self, return_new_tokens_only=True, decode=True, accelerator:Optional[Accelerator]=None, **inputs):
        outputs = self.model.generate(**inputs)

        if return_new_tokens_only:
            if self.model.config.is_encoder_decoder:
                if "decoder_input_ids" in inputs:
                    start_idx = inputs["decoder_input_ids"].shape[1] + 1
                else:
                    start_idx = 1
            else:
                start_idx = inputs["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        if accelerator is not None:
            # must be contiguous
            outputs = outputs.contiguous()
            outputs = accelerator.pad_across_processes(outputs, pad_index=self.tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs)
        
        outputs = outputs.tolist()
        if decode:
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output


    @torch.no_grad()
    def hf_llm_generate(
        self,
        context : List[dict] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        ) -> str:

        '''
        hf llm inference for single prompt. Yield single response in form of a string. 

        example context:
        [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
                ]
        '''

        tokenizer = self.tokenizer
        model = self.model

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        input_ids = tokenizer.apply_chat_template(
            context,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)

    @torch.no_grad()
    def hf_llm_generate_via_pipline(
        self,
        context : List[List[dict]] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        ) -> List[str]:

        '''
        hf llm inference for a batch (list) of single prompts. Yield multiple responses in form of list of strings. 

        example context:
        [
            [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": "Who are you?"},
            ],
            [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": "Who are you?"},
            ]
        ]
        '''


        tokenizer = self.tokenizer
        model = self.model

        generate_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        terminators = [
            generate_pipeline.tokenizer.eos_token_id,
            generate_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = generate_pipeline(
            context,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p
        )

        return [output[0]["generated_text"][-1]["content"] for output in outputs]

