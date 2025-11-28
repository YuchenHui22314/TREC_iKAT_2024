import os
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from typing import Tuple, List, Optional, Any

import torch
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn

from tqdm import tqdm

from sklearn.preprocessing import normalize

from vllm import LLM, SamplingParams

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
        return outputs


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

        
def _launch_llm_on_gpu(
    gpu_id: int,
    prompts_subset: List[str],
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    temperature: float,
    max_tokens: int,
    return_queue: multiprocessing.Queue
):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )

    outputs = llm.generate(prompts_subset, sampling_params)
    texts = [output.outputs[0].text for output in outputs]

    # print the prompt and text pair
    for prompt, text in zip(prompts_subset, texts):
        print("==========================")
        print(f"Prompt: {prompt}\nText: {text}\n")

    return_queue.put((gpu_id, texts))

def generate_with_multi_gpu_vllm(
    prompts: List[str],
    num_gpus: int,
    model_path: str = "/data/rech/huiyuche/huggingface/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
    temperature: float = 0.7,
    max_tokens: int = 2500,
    max_model_len: int = 20000,
    tensor_parallel_size: int = 1
) -> List[str]:
    """
    Generates responses for a list of prompts using multiple GPUs in parallel via vLLM,
    updating a progress bar for each completed prompt (fine-grained).
    """

    chunk_size = (len(prompts) + num_gpus - 1) // num_gpus
    prompt_chunks = [prompts[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]

    procs = []
    manager = multiprocessing.Manager()
    return_queue = manager.Queue()

    for i in range(num_gpus):
        chunk = prompt_chunks[i]
        if not chunk:
            continue

        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
        print(os.environ["CUDA_VISIBLE_DEVICES"])

        p = multiprocessing.Process(
            target=_launch_llm_on_gpu,
            args=(i, chunk, model_path, tensor_parallel_size, max_model_len, temperature, max_tokens, return_queue)
        )
        p.start()
        procs.append(p)


    for p in procs:
        p.join()

     # Collect and sort results by GPU ID to maintain order
    outputs_by_gpu = {}
    while not return_queue.empty():
        gpu_id, texts = return_queue.get()
        outputs_by_gpu[gpu_id] = texts

    # Merge outputs in original prompt order
    outputs = []
    for i in range(num_gpus):
        if i in outputs_by_gpu:
            outputs.extend(outputs_by_gpu[i])

    return outputs


###### BEIR encoders #######


import numpy as np
import torch
from typing import List, Dict
from tqdm import tqdm

###### BEIR encoders #######
class BeirConvdrEncoder:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        max_length_query=512,
        max_length_doc=512
    ):
        """
        
        Args:
            model: convdr BiEncoder instance
            tokenizer: corresponding tokenizer
            device: torch.device
            max_length_query: maximum length for queries
            max_length_doc: maximum length for documents
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc
        
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                q_embs = self.model.query_emb(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                embeddings.append(q_embs.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                d_embs = self.model.body_emb(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                embeddings.append(d_embs.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)


class BeirCLSEncoder:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Encoder using Bert-based encoder with [CLS] token embedding (not mean pooling).

        Args:
            model_path (str): Hugging Face model repo name or local path (e.g., "roberta-base")
            device (torch.device, optional): Device to run the model on. Defaults to CPU if not provided.
            max_length_query (int): Max length for query encoding.
            max_length_doc (int): Max length for document encoding.
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)

        # Move to device and set eval mode
        self.model.eval()
        self.model.to(self.device)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                #  Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
                # L2 normalize
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Preprocess: combine title and text like BEIR expects
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                #  Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                # L2 normalize
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

        
# class BeirAsymmetricEncoder:
#     def __init__(
#         self,
#         qeury_encoder_path= None, 
#         document_encoder_path = None):

#         self.model = None # ---> HERE Load your custom model
    
#     # Write your own encoding query function (Returns: Query embeddings as numpy array)
#     def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
#         pass
    
#     # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
#     def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
#         pass
