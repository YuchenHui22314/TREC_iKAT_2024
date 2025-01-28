import torch
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize

from pyserini.encode import DocumentEncoder, QueryEncoder

from openai import OpenAI
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    logging,
    T5Tokenizer,
    T5ForConditionalGeneration
)


from accelerate import Accelerator
from typing import Tuple, List, Optional, Any
from peft import PeftModel, PeftConfig
logger = logging.get_logger(__name__)


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
                 logprobs = False,
                 top_logprobs = 1,
                 ):
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
            api_key=self.api_key,
            )
    
    @staticmethod
    def get_text(response):
        responses = []
        for choice in response.choices:
            responses.append(choice.message.content)

        return responses
    
    def get_probabilities(self, response):
        pass


    def generate(self, prompt):
        message = [
                ## TODO: add system prompt. For example,
                # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": prompt}
            ]
        get_results = False
        while not get_results:
            try:
                result = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=message,
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
        return result

    def generate_text(self,prompt):
        result = self.generate(prompt)
        return self.get_text(result)

    def generate_probabilities(self,prompt):
        result = self.generate(prompt)
        return self.get_probabilities(result) 

class LM(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        cache_dir,
        padding_side="left",
        dtype="bf16",
        device_map=None,
        attn_implementation="flash_attention_2",
        #use_flash_attention_2=False, (deprecated)
        access_token=None,
        accelerator: Accelerator = None,
        load_in_8bit = False,
        load_in_4bit = False,
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
                attn_implementation=attn_implementation,
                #use_flash_attention_2=False, (deprecated)
                token=access_token,
                load_in_8bit = load_in_8bit,
                load_in_4bit = load_in_4bit
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
        ) -> Tuple[List[str], Tuple[torch.Tensor]]:

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
            num_return_sequences=num_return_sequences,
            output_scores = True,
            output_logits = True,
            return_dict_in_generate = True
        )

        responses = outputs.sequences[...,input_ids.shape[-1]:] 
        return tokenizer.batch_decode(responses, skip_special_tokens=True),outputs.logits

    def yield_expansion_terms(self, logits, num_expansion_terms, indicator_token):
        '''
        logits: tuple(tensor(1,vocab_size), tensor(1,vocab_size), ...)
        '''
        found_indicator = False
        for i in range(len(logits)):
            if found_indicator:
                # This is position for the expansion term
                top_n_id = torch.tensor(list(logits[i].flatten().cpu().numpy().argsort())[-num_expansion_terms:][::-1], dtype=torch.long)
                top_tokens = self.tokenizer.batch_decode(top_n_id, skip_special_tokens=True)
                top_tokens = [token.strip() for token in top_tokens]

                '''
                should be like:
                {token: logit, token: logit, ...}
                '''
                token_logits_dict = dict(zip(top_tokens, sorted(list(logits[i].flatten().cpu().numpy()),reverse=True)[:num_expansion_terms]))
                return token_logits_dict

            token = self.tokenizer.decode(np.argmax(logits[i].cpu().numpy())).strip()

            if token == indicator_token:
                found_indicator = True
        
        if not found_indicator:
            return None


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
        
        batch_logits = self(**batch, labels=dummy_labels).logits

        return softmax(batch_logits[:, 0, self.targeted_ids]).detach().cpu().numpy() # B 2


def get_model_repllama(
    peft_model_name, 
    cache_dir,
    device_map,
    quant_8bit = True,
    quant_4bit = False,
    ):
    config = PeftConfig.from_pretrained(peft_model_name, cache_dir=cache_dir)
    base_model = AutoModel.from_pretrained(
        config.base_model_name_or_path, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        load_in_8bit = quant_8bit,
        load_in_4bit = quant_4bit,
        )

    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()

    return model

def load_repllama(
    cache_dir: str,
    device_map,
    quant_8bit: bool = True,
    quant_4bit: bool = False
    ) -> Tuple[Any,Any]:

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model_repllama(
        'castorini/repllama-v1-7b-lora-passage',
        cache_dir,
        device_map,
        quant_8bit,
        quant_4bit
        )
    
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.config.pad_token_id = 0
    
    return tokenizer, model

class RepllamaDocumentEncoder(DocumentEncoder):

    def __init__(
        self, 
        cache_dir: str,
        device_map, 
        quant_4bit: bool = False, 
        quant_8bit: bool = False
        ):
        self.tokenizer, self.model = load_repllama(
            cache_dir=cache_dir,
            device_map = device_map,
            quant_4bit=quant_4bit,
            quant_8bit=quant_8bit
        ) 
        self.device = device_map

    def encode(self, texts):

        shared_tokenizer_kwargs = dict(
            max_length=2048,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        input_kwargs = {}
        input_kwargs["text"] = [f'passage: {text}</s>' for text in texts]  


        inputs = self.tokenizer(**input_kwargs, **shared_tokenizer_kwargs)

        if "cuda" in self.device:
            inputs.to(self.device)

        outputs = self.model(**inputs)
        # last place of the sequence
        passage_embeddings = outputs.last_hidden_state[:,-1,:].detach().cpu()
        passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)
        passage_embeddings = torch.tensor(passage_embeddings, dtype=torch.float32).numpy()

        return passage_embeddings




class AutoQueryEncoder(QueryEncoder):
    def __init__(self, encoder_dir: str, tokenizer_name: str = None, device: str = 'cpu',
                 pooling: str = 'cls', l2_norm: bool = False, prefix=None):
        self.device = device
        self.model = AutoModel.from_pretrained(encoder_dir)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or encoder_dir)
        self.pooling = pooling
        self.l2_norm = l2_norm
        self.prefix = prefix

    def encode(self, query: str, **kwargs):
        if self.prefix:
            query = f'{self.prefix} {query}'
        inputs = self.tokenizer(
            query,
            add_special_tokens=True,
            return_tensors='pt',
            truncation='only_first',
            padding='longest',
            return_token_type_ids=False,
        )
        inputs.to(self.device)
        outputs = self.model(**inputs)[0].detach().cpu().numpy()
        if self.pooling == "mean":
            embeddings = np.average(outputs, axis=-2)
        else:
            embeddings = outputs[:, 0, :]
        if self.l2_norm:
            embeddings = normalize(embeddings, norm='l2')
        return embeddings.flatten()