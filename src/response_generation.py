
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer
    )

from llm import monoT5
from peft import PeftModel, PeftConfig
from typing import List, Tuple, Any, Dict
import numpy as np
import json
from tqdm import tqdm

def generate_responses(
    hits: Dict[str, List[Any]],
    args: Any,
) -> Dict[str, List[str]]:
    response_dict = {}
    # TODO: 3 should be specified by the user of evaluation.py
    top_k = 3
    for qid, hit_list in tqdm(hits.items(), desc="Generating responses", total = len(hits)):

        # TODO: add response generation code here
        response_dict[qid] = ["pseudo response 1"]
    
    return response_dict