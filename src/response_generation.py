
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
from pyserini.search.lucene import LuceneSearcher

sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')

from topics import (
    Turn,
    load_document_by_id,
    get_trun_by_qid,
    get_context_by_qid,
)

from promptor import (
    PersonalizedResponseGenPromptor
)
from llm import (
    LM,
    OpenAILM
)

def generate_responses(
    turn_list: List[Turn],
    hits: Dict[str, List[Any]],
    generation_query_list: List[str],
    qid_list_string: List[str],
    args: Any,
) -> Dict[str, List[str]]:

    generation_query_dic = {qid: generation_query for qid, generation_query in zip(qid_list_string, generation_query_list)}

    ####################################
    # initialize the response generator
    ####################################
    if args.generation_prompt == "raw":
        generate_promptor = PersonalizedResponseGenPromptor()

    ####################################
    # initialize LLM
    ####################################

    if "gpt" in args.generation_model:
        generator = OpenAILM(
        api_key = os.environ['openai_key'],
        model_name = rewrite_model,
        n = 1,
        max_tokens=2048,
        wait_till_success=True 
        )
    

    response_dict = {}
    top_k = args.generation_top_k
    searcher = LuceneSearcher(args.index_dir_path)

    for qid, hit_list in tqdm(hits.items(), desc="Generating responses", total = len(hits)):

        # arguments needed for prompting
        current_turn = get_trun_by_qid(qid, turn_list)

        ptkb_dict = current_turn.ptkb
        context = get_context_by_qid(qid, turn_list)
        candidate_doc_list = [load_document_by_id(doc_object.docid,searcher)["contents"] for doc_object in hit_list[:top_k]]
        

        # build prompt
        prompt = generate_promptor.build_turn_prompt(
            context=context,
            ptkb_dict=ptkb_dict,
            passages_list=candidate_doc_list,
            last_question=generation_query_dic[qid],
            )
        
        # generate response
        response = generator.generate_text(prompt)
        response_finale = generate_promptor.parse_returned_text(response[0])
        if response_finale == None:
            print("error with qid: ", qid)
        
        # TODO: add response generation code here
        response_dict[qid] = [response_finale]
        print("############################################")
        print("qid: ", qid)
        print("query: ", generation_query_dic[qid])
        print("candidate_doc_list: \n")
        for doc in candidate_doc_list:
            print("###")
            print(doc)
        print("###")
        print("response: ", response_finale)
        
    
    return response_dict