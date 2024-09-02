import torch
from torch.nn import DataParallel
import torch.nn as nn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Manager

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    AutoModel
    )

from llm import monoT5

from peft import PeftModel, PeftConfig
from typing import List, Tuple, Any, Dict
import numpy as np
import json

cache_dir = "/data/rech/huiyuche/huggingface"

def get_model(
    peft_model_name, 
    cache_dir,
    quant_8bit = True,
    quant_4bit = False,
    ):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        load_in_8bit = quant_8bit,
        load_in_4bit = quant_4bit,
        num_labels=1)

    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()

    return model




def load_rankllama(
    cache_dir: str,
    quant_8bit: bool = True,
    quant_4bit: bool = False
    ) -> Tuple[Any,Any]:

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    model = get_model(
        'castorini/rankllama-v1-7b-lora-passage', 
        cache_dir,
        quant_8bit,
        quant_4bit
        )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.config.pad_token_id = 0

    return tokenizer, model



def load_t5_DP(
    cache_dir: str,
    model_name: str = 'castorini/monot5-base-msmarco',
    ) -> Tuple[PreTrainedTokenizer,T5ForConditionalGeneration,Any,Any]:
    

    # load model
    model = monoT5.from_pretrained(model_name, cache_dir=cache_dir)
    model.set_tokenizer()
    model.set_targets(['true', 'false'])
    tokenizer = model.tokenizer
    decoder_stard_id = model.config.decoder_start_token_id
    targeted_ids = model.targeted_ids


    parallel = True
    # data parallel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if parallel:
        model = DataParallel(model).to(device)
    else:
        model = model.to(device)

    model.eval()

    return tokenizer, model, decoder_stard_id, targeted_ids


def load_t5_DDP(
    cache_dir: str,
    model_name: str = 'castorini/monot5-base-msmarco',
    ) -> Tuple[PreTrainedTokenizer,T5ForConditionalGeneration,Any,Any]:
    

    # load model
    model = monoT5.from_pretrained(model_name, cache_dir=cache_dir)
    model.set_tokenizer()
    model.set_targets(['true', 'false'])
    tokenizer = model.tokenizer
    decoder_stard_id = model.config.decoder_start_token_id
    targeted_ids = model.targeted_ids

    model.eval()

    return tokenizer, model, decoder_stard_id, targeted_ids

def rerank_rankllama(
    query: str,
    passages: List[str],
    tokenizer: Any,
    model: Any
) -> List[float]: 

    # Split passages into groups of 10 passages
    # due to GPU resources limitation.
    passages_parts = np.array_split(passages, 5)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f'query: {query}'] * len(passages_part), [f'document: {passage}' for passage in passages_part], 
            return_tensors='pt',
            padding = True,
            max_length = 2048,
            truncation = True
            )

        # Run the model forward
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            part_scores = logits[:,0]
            scores.extend(list(part_scores))
        
        # transform to torch float32
        scores = [float(score) for score in scores]
        

    return scores

def rerank_t5_DP(
    query: str,
    passages: List[str],
    tokenizer: Any,
    model: Any,
    decoder_input_ids: Any,
    targeted_ids: Any
    ) -> List:

    # Split passages into groups of 67 passages
    # due to GPU resources limitation.
    # 15 on octal31, 6 on octal40 when reranking top 1000
    passages_parts = np.array_split(passages, 1)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f"Query: {query} Document: {passage} Relevant:" for passage in passages_part],
            max_length = 512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # predict
        with torch.no_grad():

            softmax = nn.Softmax(dim=1)
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")

            dummy_labels = torch.full(
                inputs.input_ids.size(), 
                decoder_input_ids
            ).to("cuda")
            
            batch_logits = model(**inputs, labels=dummy_labels).logits
            true_false = softmax(batch_logits[:, 0, targeted_ids]).detach().cpu().numpy() # B 2
            true_prob = true_false[:,0]
            scores.extend(true_prob.tolist())
        

    return scores


def rerank_t5_DDP(
    rank: int,
    world_size: int,
    query: str,
    passages: List[str],
    tokenizer: Any,
    decoder_input_ids: Any,
    targeted_ids: Any,
    outputs: Manager().list(),
    ) -> None:

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    # construct DDP model
    tokenizer, model, decoder_stard_id, targeted_ids= load_t5_DDP(
        cache_dir = cache_dir,
        model_name = 'castorini/monot5-base-msmarco'
        )
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Split passages into groups of 67 passages
    # due to GPU resources limitation.
    passages_parts = np.array_split(passages, 15)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f"Query: {query} Document: {passage} Relevant:" for passage in passages_part],
            max_length = 512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # predict
        with torch.no_grad():

            softmax = nn.Softmax(dim=1)
            for k in inputs:
                inputs[k] = inputs[k].to(rank)

            dummy_labels = torch.full(
                inputs.input_ids.size(), 
                decoder_input_ids
            ).to(rank)
            
            batch_logits = ddp_model(**inputs, labels=dummy_labels).logits
            true_false = softmax(batch_logits[:, 0, targeted_ids]).detach().cpu().numpy() # B 2
            true_prob = true_false[:,0]
            scores.extend(true_prob.tolist())

    if rank == 0:
        outputs.append(scores)

    # Clean up
    dist.destroy_process_group()


def hits_2_rankgpt_list(
    searcher: Any,
    query_dict: Dict[str, str],
    hits_dict: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:


    rankgpt_list = []
    new_hits_dict = {}

    for qid, query in query_dict.items():
        hits = hits_dict[qid]
        new_hits_dict[qid] = []
        rankgpt_list.append({'query': query, 'hits': []})

        # assuming that hits are sorted by rank
        for rank, hit in enumerate(hits):
            # get passage text
            content = json.loads(searcher.doc(hit.docid).raw())
            content = content['contents']
            content = ' '.join(content.split())

            document_hit_dict = {
                'content': content,
                'qid': qid, 
                'docid': hit.docid, 
                'rank': rank, 
                'score': hit.score}

            rankgpt_list[-1]['hits'].append(document_hit_dict)
            
            new_hits_dict[qid].append(document_hit_dict)

    return rankgpt_list, new_hits_dict