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

import nltk
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def is_english_word(word):
    base_form = lemmatizer.lemmatize(word.lower())
    return base_form in words.words()


sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
#sys.path.append('../')
from llm import OpenAILM
from promptor import (
    RewriteAndResponsePromptor,
    PersonalizedCIRQueryExpansionPromptor,
    SummarizePTKBPromptor,
    PersonalizeViaPTKBSummaryPrompter
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

from llm import (
    LM
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

    parser.add_argument("--reformulation_name", type = str, default="rar", choices=[
        "rar",
        "rar_cot",
        "rar_ptkb_sum_cot0",
        "rar_ptkb_sum",
        "ptkb_summarize",
        "raw_llm_rm_PDCReORf",
        # P -> personalize, D -> demo, C -> cot, Re -> rel explain
        # O -> oracle, Rf -> rel feedback
        "raw_llm_rm_P__Re___",
        "raw_llm_rm____Re___",
        ]
    ) 
    args = parser.parse_args()

    return args


def get_llm_rm_expansion_terms(
    prompter_choices: str,
    turn,
    turn_list,
    llm_model, # A hugging face LLM model
    num_expansion_terms,
):

    if llm_model == None:
        cache_dir = "/data/rech/huiyuche/huggingface"
        llm_model = LM(
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            padding_side="left",
            dtype="bf16",
            device_map= "auto",
            attn_implementation="flash_attention_2",
            access_token=None,
            cache_dir=cache_dir,
            accelerator = None,
            load_in_8bit = False,
            load_in_4bit = False,
        )


    enable_personalization=False
    enable_demo=False
    enable_cot=False
    enable_relevance_explanation=False
    enable_oracle=False
    enable_relevance_feedback=False

    # pasre the prompter choices pattern "PDCReORf"
    if prompter_choices[0] == "P":
        enable_personalization = True
    if prompter_choices[1] == "D":
        enable_demo = True
    if prompter_choices[2] == "C":
        enable_cot = True
    if prompter_choices[3:5] == "Re":
        enable_relevance_explanation = True
    if prompter_choices[5] == "O":
        enable_oracle = True
    if prompter_choices[6:7] == "Rf":
        enable_relevance_feedback = True


    prompter = PersonalizedCIRQueryExpansionPromptor(
    demo_file = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json",
    enable_personalization=enable_personalization,
    enable_demo=enable_demo,
    enable_cot=enable_cot,
    enable_relevance_explanation=enable_relevance_explanation,
    enable_oracle=enable_oracle,
    enable_relevance_feedback=enable_relevance_feedback
    )

    context = get_context_by_qid(turn.turn_id,turn_list)
    current_turn_ptkb_dict = turn.ptkb
    prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)

    messages = [
        {
            "role": "user",
            "content": prompt 
        }
    ]

    sequences,logits = llm_model.hf_llm_generate(
        messages,
        temperature = 1,
        top_p = 0.9,
        max_new_tokens = 256,
        do_sample = True,
        num_beams = 1,
        num_return_sequences = 1
    )

    expension_terms_weights_dict = llm_model.yield_expansion_terms(logits,num_expansion_terms,":")

    return expension_terms_weights_dict, prompt

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

    if reformulation_name == "rar":
        prompter = RewriteAndResponsePromptor(
            demo_file = demo_file, 
            enable_cot = False
        )

    if "summarize" in reformulation_name:
        prompter= SummarizePTKBPromptor()
    
    if reformulation_name == "rar_ptkb_sum_cot0":
        prompter = PersonalizeViaPTKBSummaryPrompter(
            enable_cot = True
        )
    
    if reformulation_name == "rar_ptkb_sum":
        prompter = PersonalizeViaPTKBSummaryPrompter(
            enable_cot = False
        )
        


    ###########################
    ## load langauge model
    ###########################
    if "gpt" in rewrite_model:
        rewriter = OpenAILM(
        api_key = os.environ['openai_key'],
        model_name = rewrite_model,
        n = 1,
        wait_till_success=True 
        )

    if "llm_rm" in  reformulation_name:
        llm_model = LM(
            model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
            padding_side="left",
            dtype="bf16",
            device_map= "auto",
            attn_implementation="flash_attention_2",
            access_token=None,
            cache_dir="/data/rech/huiyuche/huggingface",
            accelerator = None,
            load_in_8bit = False,
            load_in_4bit = False,
        )

    #################################
    ## load topic file and rewrite
    #################################

    turn_list = load_turns_from_json(input_query_path)
    for turn in tqdm(turn_list, total=len(turn_list), desc="Rewriting"):
        context = get_context_by_qid(turn.turn_id,turn_list)

        if "_ptkb_sum" in reformulation_name:

            ptkb_summary =\
                 turn.find_reformulation("ptkb_summarize").reformulated_query
            decontextualized_query =\
                 turn.find_reformulation("rar_rw").reformulated_query

            prompt = prompter.build_turn_prompt(
                summary = ptkb_summary,
                user_query = decontextualized_query,
            )

            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])
        
            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(turn)
                continue

            rewrite = liste[0]
            response = liste[1]
            cot = liste[2]

            if "cot" in reformulation_name:
                turn.add_reformulation(
                    reformulation_name = reformulation_name + "_cot",
                    reformulated_query = cot,
                    ptkb_provenance = []
                )
            
            turn.add_reformulation(
                reformulation_name = reformulation_name + "_rw",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )

            turn.add_reformulation(
                reformulation_name = reformulation_name + "_rs",
                reformulated_query = response,
                ptkb_provenance = []
            )



        if "llm_rm" in reformulation_name:
            pattern = reformulation_name[-8:]
            expansion_terms_weights_dict, prompt = get_llm_rm_expansion_terms(
                pattern,
                turn,
                turn_list,
                llm_model,
                num_expansion_terms = 100
            )

            if expansion_terms_weights_dict == None:
                print(f"error with turn id {turn.turn_id}")
                print(turn)
                continue

            for key in expansion_terms_weights_dict.copy():
                if not is_english_word(key):
                    del expansion_terms_weights_dict[key]

            reformulated_query = " ".join(list(expansion_terms_weights_dict.keys()))
            turn.add_reformulation(
                reformulation_name = reformulation_name,
                reformulated_query = reformulated_query,
                ptkb_provenance = []
            )


        elif "summarize" in reformulation_name:
            prompt = prompter.build_turn_prompt(turn.ptkb)
            response = rewriter.generate_text(prompt)[0]
            summary = prompter.parse_returned_text(response)
            if summary == None:
                print(f"error with turn id {turn.turn_id}")
                print(turn)
                continue

            turn.add_reformulation(
                reformulation_name = reformulation_name,
                reformulated_query = summary,
                ptkb_provenance = []
            )


        # we generated just 1 response
        elif reformulation_name == "rar":
            # generate prompt for the current turn
            prompt = prompter.build_turn_prompt(context,turn)

            # rewrite the prompt
            responses = rewriter.generate_text(prompt)

            rewrite_resposne_cot = prompter.parse_returned_text(responses[0]) 
            if rewrite_resposne_cot == None:
                print(f"error with turn id {turn.turn_id}")
                print(turn)
                continue
            rewrite = rewrite_resposne_cot[0]
            response = rewrite_resposne_cot[1]

            turn.add_reformulation(
                reformulation_name = "rar_rw",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = "rar_rs",
                reformulated_query = response,
                ptkb_provenance = []
            )

    
    #################################
    ## save turn list
    #################################

    save_turns_to_json(turn_list, "/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2023_test.json")
    


         