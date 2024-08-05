'''
Rewrite all queries in input_query_path 
'''
import pickle
import sys
import os
import numpy as np
import json
import argparse
import re
from tqdm import tqdm



sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
#sys.path.append('../')
from llm import OpenAILM
from promptor import (
    RewriteAndResponsePromptor
)

from topics import (
    Turn, 
    load_turns_from_ikat_topic_files, 
    save_turns_to_json, 
    load_turns_from_json,
    Result,
    Reformulation,
    get_context_by_qid,
    get_turn_by_qid
)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/ikat_2023_test.json")

    parser.add_argument("--demo_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json")

    parser.add_argument("--rewrite_model", type=str, default="gpt-3.5-turbo", choices=[
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-16k", 
        "gpt-4o",
        "mistral-7b",
        "llama3-8b"])

    parser.add_argument("--reformulation_name", type = str, default="RAR", choices=[
        "RAR",
        "RAR_COT",
        "RAR_personalized_CoT",
        "expansion"
        ]
    ) 
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    input_query_path = args.input_query_path
    rewrite_model = args.rewrite_model
    demo_file = args.demo_file
    reformulation_name = args.reformulation_name

    # ask user for openai api key
    if 'openai_key' not in os.environ:
        os.environ['openai_key'] = input("Please Enter your OpenAI API key: ")

    ###########################
    ## load prompter
    ###########################

    if reformulation_name == "RAR":
        prompter = RewriteAndResponsePromptor(
            demo_file = demo_file, 
            enable_cot = False
        )

    ###########################
    ## load langauge model
    ###########################
    if "gpt" in rewrite_model:
        rewriter = OpenAILM(
        api_key = os.environ['openai_key'],
        model_name = rewrite_model,
        n = 1 
        )

    #################################
    ## load topic file and rewrite
    #################################

    turn_list = load_turns_from_json(input_query_path)
    for turn in tqdm(turn_list, total=len(turn_list), desc="Rewriting"):
        context = get_context_by_qid(turn.turn_id,turn_list)

        # generate prompt for the current turn
        prompt = prompter.build_turn_prompt(context,turn)

        # rewrite the prompt
        #responses = rewriter.generate_text(prompt)

        # we generated just 1 response
        if reformulation_name == "RAR":

            responses = ['Rewrite: What about the DASH diet? I heard it is a healthy diet.\nResponse: The DASH (Dietary Approaches to Stop Hypertension) diet is indeed a healthy eating plan that is designed to help lower blood pressure and improve overall health.']

            rewrite_resposne_cot = prompter.parse_returned_text(responses[0]) 
            rewrite = rewrite_resposne_cot[0]
            response = rewrite_resposne_cot[1]

            turn.add_reformulation(
                reformulation_name = "RAR:rewrite",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = "RAR:response",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
    
    #################################
    ## save turn list
    #################################

    save_turns_to_json(turn_list, "/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2023_test.json")
    


         