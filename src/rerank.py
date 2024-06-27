import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from typing import List, Tuple, Any
import numpy as np

cache_dir = "/data/rech/huiyuche/huggingface"

def get_model(peft_model_name, cache_dir):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        load_in_8bit = True,
        num_labels=1)

    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()

    return model


def load_rankllama(
    cache_dir: str,
) -> Tuple[Any,Any]:

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    model = get_model('castorini/rankllama-v1-7b-lora-passage', cache_dir)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.config.pad_token_id = 0

    return tokenizer, model

# def rerank_rankllama(
#     query: str,
#     passages: List[str],
#     tokenizer: Any,
#     model: Any,
# ) -> List[float]: 

#     inputs = tokenizer(
#         [f'query: {query}'] * len(passages), [f'document: {passage}' for passage in passages], 
#         return_tensors='pt',
#         padding = True,
#         max_length = 2048,
#         truncation = True
#         )

#     # Run the model forward
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         scores = logits[:,0]

#     return list(scores)

def rerank_rankllama(
    query: str,
    passages: List[str],
    tokenizer: Any,
    model: Any,
) -> List[float]: 

    # Split passages into 5 parts
    passages_parts = np.array_split(passages, 100)
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

    return scores