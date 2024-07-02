import os
import re
import torch
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from tqdm import tqdm

from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    logging,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Mapping, Tuple, List, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.get_logger(__name__)

# Specify the custom cache directory
cache_dir = "/data/rech/huiyuche/huggingface"

class OpenAILM():
    def __init__(self, 
                 api_key, 
                 model_name="gpt-3.5-turbo", 
                 n=1, 
                 max_tokens=512, 
                 temperature=0, 
                 top_p=1, 
                 frequency_penalty=0.0, 
                 presence_penalty=0.0, 
                 stop=['\n\n\n'], 
                 wait_till_success=False,
                 logprobs = False
                 ):
        super().__init__(model_name, api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.wait_till_success = wait_till_success
        self.logprobs = logprobs
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI"),
            )
    
    @staticmethod
    def parse_response(response):
        reformulated_query = response.choices[0].message.content
        return reformulated_query

    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                result = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prompt,
                            temperature=self.temperature,
                            logprobs=self.logprobs,
                            top_p=self.top_p,
                            n=self.n,
                            max_tokens=self.max_tokens,
                            stop=["\n\n\n"]
                        )
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return self.parse_response(result)


class LM(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        padding_side="left",
        dtype="bf16",
        device_map=None,
        attn_implementation="flash_attention_2",
        #use_flash_attention_2=False, (deprecated)
        access_token=None,
        cache_dir=cache_dir,
        accelerator: Accelerator = None,
        load_in_8bit = False,
        load_in_4bit = False
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
                attn_implementation="flash_attention_2",
                #use_flash_attention_2=False, (deprecated)
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
        num_beams: int = 1,
        num_return_sequences: int = 1
        ) -> List[str]:

        '''
        hf llm inference for single prompt. Yield single response in form of a list of responses (len(list)>1 while num_return_sequences > 1). 

        example context with num_return_sequences = 2:
        [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
                ]

        response:[
            " I am a large language model trained by Mistral AI....",
            " I am a large language model trained by Mistral AI...."
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
            return_tensors="pt",
            padding = True,
            ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )

        responses = outputs[...,input_ids.shape[-1]:] 
        return tokenizer.batch_decode(responses, skip_special_tokens=True)

    @torch.no_grad()
    def hf_llm_generate_via_pipline(
        self,
        context : List[List[dict]] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        num_beams: int = 1,
        num_return_sequences: int = 1
        ) -> List[List[str]]:

        '''
        hf llm inference for a batch (list) of single prompts. Yield multiple responses in form of list of strings. 

        example context with num_return_sequences = 2:
        messages = [
            [{"role": "user", "content": "Who are you?"}],
            [{"role": "user", "content": "what is the capital of Germany?"}],
        ]

        this will yield
        [
            [
                " I am a large language model trained by Mistral AI....",
                " I am a large language model trained by Mistral AI...."
                ],
            [
                " The capital of Germany is Berlin...",
                " The capital of Germany is Berlin...",
                ],
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
            top_p=top_p,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )

        return [[alternative["generated_text"][-1]["content"] for alternative in output] for output in outputs ]
    


class monoT5(T5ForConditionalGeneration):
    targeted_tokens = ['true', 'false']
    # tokenizer_name = 'google/t5-base'

    def data_parallel(self, parallel = True) -> None:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if parallel:
            self.model = DataParallel(self).to(device)
        else:
            self.model = self.to(device)

    def set_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif 'large' in self.name_or_path:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        elif 'base' in self.name_or_path:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def set_targets(self, tokens=None):
        """
        Parameters:
            tokens: list of string
        """
        if tokens is None:
            tokens = self.targeted_tokens

        tokenized_tokens = self.tokenizer(tokens, add_special_tokens=False)
        self.targeted_ids = [x for xs in tokenized_tokens.input_ids for x in xs]
        # print(f"{len(tokens)} targeted tokens set")
        # print(list(zip(tokens, self.targeted_ids)))
        print("Ready for predict()")

    def predict(self, batch):
        """
        Parameters:
            batch: batch inputs of tokenized query-passage pair.
        """
        softmax = nn.Softmax(dim=1)

        for k in batch:
            batch[k] = batch[k].to(self.device)

        dummy_labels = torch.full(
                batch.input_ids.size(), 
                self.config.decoder_start_token_id
        ).to(self.device)
        
        batch_logits = self.model(**batch, labels=dummy_labels).logits

        return softmax(batch_logits[:, 0, self.targeted_ids]).detach().cpu().numpy() # B 2