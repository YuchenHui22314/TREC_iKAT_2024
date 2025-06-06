'''
Rewrite all queries in input_query_path 
'''
import os
import argparse
import tkinter as tk
from tkinter import simpledialog
from tqdm import tqdm
import json
import math

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

import nltk
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

from vllm import LLM, SamplingParams
from .personalized_weight_topic_entropy import calculate_topic_entrop
from .personalized_weight_DEPS import calculate_std_top_k_list
from apcir.search.models import ANCE

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def is_english_word(word):
    base_form = lemmatizer.lemmatize(word.lower())
    return base_form in words.words()


from apcir.functional.promptor import (
    RewriteAndResponsePromptor,
    PersonalizedCIRQueryExpansionPromptor,
    SummarizePTKBPromptor,
    PersonalizeViaPTKBSummaryPrompter,
    RARPersonalizedCoTPromptor,
    RARNonPersonalizedCoTPromptor,
    JudgePersonalizeLevelPromptor, 
    JudgeThenRewritePromptor,
    MQ4CSPrompter,
    MQ4CSRWPrompter,
    GtR_RS,
    GtR_RW,
    Fengran_10QR_Prompter,
    Fengran_10GRF_Prompter
)

from apcir.functional.topics import (
    save_turns_to_json, 
    save_turns_to_topiocqa, 
    load_turns_from_json,
    load_turns_from_topiocqa,
    get_context_by_qid,
)

from apcir.functional.llm import (
    LM,
    OpenAILM
)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/ikat_2023_test.json")

    parser.add_argument("--output_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2024_test.json")

    parser.add_argument("--demo_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json")

    parser.add_argument("--result_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/results/ClueWeb_ikat/ikat_23_test/ranking/S1[gpt-4o_rar_rw]-S2[none]-g[none]-[splade_v3]-[none_4_1_none]-[s2_top50].txt")

    parser.add_argument("--deps_entropy_top_k", type=int, default=100)

    parser.add_argument("--cache_dir", type=str, default="/data/rech/huiyuche/huggingface")

    parser.add_argument("--rewrite_model", type=str, default="gpt-3.5-turbo", choices=[
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-16k", 
        "gpt-4-0613",
        "gpt-4o-2024-08-06",
        "mistral-8b",
        "llama3-8b",
        'none'])

    parser.add_argument("--reformulation_name", type = str, default="rar", choices=[
        "rar",
        "rar_cot",
        "gpt-4o_rar",
        "gpt-4o_rar_cot",
        "rar_ptkb_sum_cot0",
        "rar_ptkb_sum",
        "ptkb_summarize",
        "raw_llm_rm_PDCReORf",
        # P -> personalize, D -> demo, C -> cot, Re -> rel explain
        # O -> oracle, Rf -> rel feedback
        "raw_llm_rm_P__Re___",
        "raw_llm_rm____Re___",
        "rar_personalized_cot1",
        "rar_personalized_cot0",
        "rar_personalized_cotN",
        "gpt-4o_rar_personalized_cot1",
        "gpt-4o_rar_non_personalized_cot1",              # modify the few shot example to remove personalization
        "gpt-4o_rar_manual_depersonalized_cot1",         # directly modify the resulting rewrite to remove personalizaiton.
        "personalization_level",
        "gpt-4o_judge_and_rewrite",
        "gpt-3.5_judge_and_rewrite",
        "gpt-4_judge_and_rewrite",
        "llama3.1_rar",
        "mistral_rar",
        "mistral_judge_and_rewrite",
        "llama3.1_judge_and_rewrite",
        "gpt-4o_MQ4CS_mq",
        "gpt-4o_MQ4CS_persq",
        "gpt-4o_jtr_wo_cot",
        "gpt-4o_jtr_wo_in_context",
        "gpt-4o_MQ4CS_mq_3",
        "gpt-4o_GtR_rs",
        "gpt-4o_GtR_mq_3",
        "gpt-4_MQ4CS_persq",
        "gpt-3.5_MQ4CS_persq",
        "llama3.1_MQ4CS_persq",
        "mistral_MQ4CS_persq",
        "llama3.1_fengran_10_qr",
        "result_topic_entropy",
        "DEPS"
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
        cache_dir = args.cache_dir
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

    # print the arguments
    print(args)

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


    if "fengran_10_qr" in reformulation_name:
        prompter = Fengran_10QR_Prompter(phi=10)

    if "GtR_mq" in reformulation_name:
        splits = reformulation_name.split("_")
        number = splits[-1]
        if number.isdigit():
            number = int(number)
            prompter = GtR_RW(phi = number)
        else:
            prompter = GtR_RW(phi = 2)
    
    if "GtR_rs" in reformulation_name:
        prompter = GtR_RS() 
    if "MQ4CS_mq" in reformulation_name:
        splits = reformulation_name.split("_")
        number = splits[-1]
        if number.isdigit():
            number = int(number)
            prompter = MQ4CSPrompter(phi = number)
        else:
            prompter = MQ4CSPrompter(phi = 2)

    if "MQ4CS_persq" in reformulation_name:
        prompter = MQ4CSRWPrompter()

    if "judge_and_rewrite" in reformulation_name:
        prompter =  JudgeThenRewritePromptor(
            enable_cot=True,
            enable_demo=True,
            demo_file = demo_file 
        )
    if "jtr_wo_cot" in reformulation_name:
        prompter =  JudgeThenRewritePromptor(
            enable_cot=False,
            enable_demo=True,
            demo_file = demo_file
            )
    if "jtr_wo_in_context" in reformulation_name:
        prompter =  JudgeThenRewritePromptor(
            enable_cot=True,
            enable_demo=False,
            demo_file = demo_file
            )
        
    if "personalization_level" in reformulation_name:
        prompter = JudgePersonalizeLevelPromptor(
            enable_cot=True
        )

    if (reformulation_name == "rar" or 
        reformulation_name == "gpt-4o_rar" or
        reformulation_name == "llama3.1_rar" or
        reformulation_name == "mistral_rar"
        ):
        prompter = RewriteAndResponsePromptor(
            demo_file = demo_file, 
            enable_cot = False
        )

    if reformulation_name == "rar_cot" or reformulation_name == "gpt-4o_rar_cot":
        prompter = RewriteAndResponsePromptor(
            demo_file = demo_file, 
            enable_cot = True
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

    if ("rar_personalized_cot" in reformulation_name) or ("rar_non_personalized_cot" in reformulation_name):
        enable_cot = True
        zero_shot_cot = False
        one_shot_cot = False
        if reformulation_name[-1] == "0":
            zero_shot_cot = True
        if reformulation_name[-1] == "1":
            one_shot_cot = True
        if reformulation_name[-1] == "N":
            enable_cot = False
        
        if "non" in reformulation_name:
            prompter = RARNonPersonalizedCoTPromptor(
                demo_file = demo_file,
                enable_cot = enable_cot,
                zero_shot_cot = zero_shot_cot,
                one_shot_cot = one_shot_cot,
                cot_format="cot_seperate"
            )
        else:
            prompter = RARPersonalizedCoTPromptor(
                demo_file = demo_file,
                enable_cot = enable_cot,
                zero_shot_cot = zero_shot_cot,
                one_shot_cot = one_shot_cot,
                cot_format="cot_seperate"
            )




    ###########################
    ## load langauge model
    ###########################

    if "gpt" in rewrite_model:
        rewriter = OpenAILM(
        api_key = os.environ['openai_key'],
        model_name = rewrite_model,
        n = 1,
        max_tokens=2048,
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
            cache_dir=args.cache_dir,
            accelerator = None,
            load_in_8bit = False,
            load_in_4bit = False,
        )
    
    if "mistral" in rewrite_model:

        rewriter = LM(
        model_name_or_path="mistralai/Ministral-8B-Instruct-2410",
        tokenizer_name_or_path="mistralai/Ministral-8B-Instruct-2410",
        padding_side="left",
        dtype="bf16",
        device_map= "cuda:0",
        attn_implementation="flash_attention_2",
        #use_flash_attention_2=False, (deprecated)
        access_token=None,
        cache_dir=args.cache_dir,
        accelerator = None
        )
    
    if "llama" in rewrite_model:
        rewriter = LM(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        padding_side="left",
        dtype="bf16",
        device_map= "cuda:0",
        attn_implementation="flash_attention_2",
        access_token=None,
        cache_dir=args.cache_dir,
        accelerator = None,
        load_in_8bit = False,
        load_in_4bit = False,
        )

    #################################
    ## load query encoder
    #################################

    if "result_topic_entropy" in reformulation_name:
        model_name = "castorini/ance-msmarco-passage" 
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
        query_encoder = ANCE.from_pretrained(model_name, cache_dir=args.cache_dir).to("cuda:0")
        query_encoder.eval()

        turn_list = load_turns_from_json(input_query_path)

        # query_encoder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/data/rech/huiyuche/huggingface')
        # tokenizer = None

        qid_personalized_weight_dict = calculate_topic_entrop(
                query_encoder,
                tokenizer,
                turn_list
            ) 
        
        print("the normalized qid_personalized_weight_dict is: ", qid_personalized_weight_dict)    

    
    if "DEPS" in reformulation_name:
        qid_personalized_weight_dict = calculate_std_top_k_list(
            args.result_file,
            args.deps_entropy_top_k
            )
        print("the qid_personalized_weight_dict is: ", qid_personalized_weight_dict)    

    #################################
    ## load topic file and rewrite
    #################################

    turn_list = load_turns_from_json(input_query_path)

    for index, turn in tqdm(enumerate(turn_list), total=len(turn_list), desc="Rewriting"):

        context = get_context_by_qid(turn.turn_id,turn_list)
            
        if "DEPS" in reformulation_name:
            
            if turn.turn_id in qid_personalized_weight_dict:
                deps_score = qid_personalized_weight_dict[turn.turn_id]
                assert len(deps_score) == 3, f"the deps score for turn id {turn.turn_id} is {deps_score}, which is not 3"
                
                # add the DEPS score to the turn
                turn.add_reformulation(
                    reformulation_name = reformulation_name,
                    reformulated_query = deps_score,
                    ptkb_provenance = []
                )



        if "result_topic_entropy" in reformulation_name:

            # scores = calculate_topic_entrop(
            #     query_encoder,
            #     tokenizer,
            #     turn,
            #     args.result_file,
            #     args.deps_entropy_top_k
            # ) 

            # print("the scores are: ", scores)

            scores = qid_personalized_weight_dict[turn.turn_id]
            
            turn.add_reformulation(
                reformulation_name = reformulation_name,
                reformulated_query = scores,
                ptkb_provenance = []
            )

            assert turn.find_reformulation(reformulation_name).reformulated_query == scores
        

        if "GtR_rs" in reformulation_name:
            context = get_context_by_qid(turn.turn_id,turn_list)
            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)

            response = rewriter.generate_text(prompt)[0]

            if type(response) != type("yuchen"):
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue
            else:
                print(f"response: {response}")

            turn.add_reformulation(
                reformulation_name = reformulation_name,
                reformulated_query = response,
                ptkb_provenance = []
            )

        if "GtR_mq" in reformulation_name:

            splits = reformulation_name.split("_")
            number = splits[-1]
            if number.isdigit():
                number = int(number)
            else:
                number = 2

            current_turn_ptkb_dict = turn.ptkb
            response = turn.find_reformulation(reformulation_name.replace(reformulation_name[-4:],"rs")).reformulated_query
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn,response)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])

            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            if len(liste) < number:
                # append " " to the end of the list
                for i in range(number - len(liste)):
                    liste.append("GG")

            for i in range(len(liste)):
                query = liste[i]
                turn.add_reformulation(
                    reformulation_name = reformulation_name+f"_{i+1}",
                    reformulated_query = query,
                    ptkb_provenance = []
                )

            try:
                print("#########################")
                print("this is turn: ", turn.turn_id)
                print(f"personalized query: {turn.find_reformulation('gpt-4o_judge_and_rewrite_rw').reformulated_query}")
                for i in range(len(liste)):
                    print(f"MQ4CS query_{i+1}: {liste[i]}")
            except e:
                print(f"print error with turn id {turn.turn_id}")
                continue
                

        if "jtr_wo_cot" in reformulation_name:
            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])
            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            cot = liste[0]
            level = liste[1]
            rewrite = liste[2]
            response = liste[3]

            print(f"cot: {cot}")
            print(f"level: {level}")
            print(f"rewrite: {rewrite}")
            print(f"response: {response}")

            turn.add_reformulation(
                reformulation_name = reformulation_name+"_lv",
                reformulated_query = level,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rw",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rs",
                reformulated_query = response,
                ptkb_provenance = []
            )

        if "jtr_wo_in_context" in reformulation_name:

            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])
            liste = prompter.parse_returned_text(response[0])

            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            cot = liste[0]
            level = liste[1]
            rewrite = liste[2]
            response = liste[3]

            print(f"cot: {cot}")
            print(f"level: {level}")
            print(f"rewrite: {rewrite}")
            print(f"response: {response}")

            turn.add_reformulation(
                reformulation_name = reformulation_name+"_cot",
                reformulated_query = cot,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_lv",
                reformulated_query = level,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rw",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rs",
                reformulated_query = response,
                ptkb_provenance = []
            )


        if "MQ4CS_persq" in reformulation_name:
            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)
            
            # rewrite the prompt
            if "mistral" in reformulation_name or "llama" in reformulation_name:
                messages = [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
                response = rewriter.hf_llm_generate(
                    messages,
                    temperature = 0,
                    top_p = 0.9,
                    max_new_tokens = 4096,
                    do_sample = False,
                    num_beams = 1,
                    num_return_sequences = 1
                )[0]
            else:
                response = rewriter.generate_text(prompt)
            
            query = prompter.parse_returned_text(response[0])
            
            if query == None:
                print(f"error with turn id {turn.turn_id}")
                print(response[0])
                continue

            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rw",
                reformulated_query = query,
                ptkb_provenance = []
            )

            try:
                print("#########################")
                print("this is turn: ", turn.turn_id)
                print(f"personalized query: {turn.find_reformulation('gpt-4o_judge_and_rewrite_rw').reformulated_query}")
                print(f"MQ4CS Pers query: {query}")
            except e:
                print(f"print error with turn id {turn.turn_id}")
                continue


        if "MQ4CS_mq" in reformulation_name:
            splits = reformulation_name.split("_")
            number = splits[-1]
            if number.isdigit():
                number = int(number)
            else:
                number = 2

            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])

            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            if len(liste) < number:
                # append " " to the end of the list
                for i in range(number - len(liste)):
                    liste.append(" ")


            for i in range(len(liste)):
                query = liste[i]
                turn.add_reformulation(
                    reformulation_name = reformulation_name+f"_{i+1}",
                    reformulated_query = query,
                    ptkb_provenance = []
                )

            try:
                print("#########################")
                print("this is turn: ", turn.turn_id)
                print(f"personalized query: {turn.find_reformulation('gpt-4o_judge_and_rewrite_rw').reformulated_query}")
                for i in range(len(liste)):
                    print(f"MQ4CS query_{i+1}: {liste[i]}")
            except e:
                print(f"print error with turn id {turn.turn_id}")
                continue
           
        

        if "judge_and_rewrite" in reformulation_name:
            current_turn_ptkb_dict = turn.ptkb

            # run the 2nd times to deal with some bugs.
            if turn.turn_id not in ["21-1-10"]:
                continue
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)

            # rewrite the prompt
            if "mistral" in reformulation_name or "llama" in reformulation_name:
                messages = [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
                response = rewriter.hf_llm_generate(
                    messages,
                    temperature = 0,
                    top_p = 0.9,
                    max_new_tokens = 4096,
                    do_sample = False,
                    num_beams = 1,
                    num_return_sequences = 1
                )[0]
            else:
                response = rewriter.generate_text(prompt)
            
            print(response)
            liste = prompter.parse_returned_text(response[0])
            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            cot = liste[0]
            level = liste[1]
            rewrite = liste[2]
            response = liste[3]

            print(f"cot: {cot}")
            print(f"level: {level}")
            print(f"rewrite: {rewrite}")
            print(f"response: {response}")
            


            turn.add_reformulation(
                reformulation_name = reformulation_name+"_cot",
                reformulated_query = cot,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_lv",
                reformulated_query = level,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rw",
                reformulated_query = rewrite,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_rs",
                reformulated_query = response,
                ptkb_provenance = []
            )

        if "personalization_level" in reformulation_name:
            current_turn_ptkb_dict = turn.ptkb
            prompt = prompter.build_turn_prompt(context,current_turn_ptkb_dict,turn)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])
            level = liste[0]
            cot = liste[1]

            turn.add_reformulation(
                reformulation_name = reformulation_name+"_lv",
                reformulated_query = level,
                ptkb_provenance = []
            )
            turn.add_reformulation(
                reformulation_name = reformulation_name+"_cot",
                reformulated_query = cot,
                ptkb_provenance = []
            )



            
        # copy the rewrite named "gpt-4o_rar_personalized_cot1_rw"
        if "gpt-4o_rar_manual_depersonalized_cot1" == reformulation_name:
            print("ok")
            original_reformulation = turn.find_reformulation("gpt-4o_rar_personalized_cot1_rw")
            reformulated_query = original_reformulation.reformulated_query

            assert reformulated_query != None, f"helas, reformulated_query named gpt-4o_rar_personalized_cot1_rw for turn id {turn.turn_id} is None"

            # an interactive way to remove personalization
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口

            # 弹出对话框，提示用户输入内容
            user_input = simpledialog.askstring(title="输入框", prompt="请输入内容:")

            # 输出用户输入的内容
            if user_input is not None:
                print(f"用户输入的内容是: {user_input}")
            else:
                print("用户取消了输入")

            root.quit()

            print(f"############################################")
            print(f"This is turn id {turn.turn_id}")
            print(f"Original reformulated query: {reformulated_query}")
            print(f"##############")
            print("now please provide the depersonalized query, if you want to keep the original query, just press enter")
            print("############################################")
            depersonalized_query = input("Depersonalized query: ")


            # add with new name
            turn.add_reformulation(
                reformulation_name = reformulation_name + "_rw",
                reformulated_query = reformulated_query,
                ptkb_provenance = []
            )


        if "rar_personalized_cot" in reformulation_name or "rar_non_personalized_cot" in reformulation_name:

            prompt = prompter.build_turn_prompt(
                context,
                turn.ptkb,
                turn)
            response = rewriter.generate_text(prompt)
            liste = prompter.parse_returned_text(response[0])
        
            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            rewrite = liste[0]
            response = liste[1]
            cot = liste[2]

            if "N" not in reformulation_name:
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






        elif "_ptkb_sum" in reformulation_name:

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
        elif (
                reformulation_name == "rar" or 
                reformulation_name == "gpt-4o_rar" or
                reformulation_name == "llama3.1_rar" or
                reformulation_name == "mistral_rar"
              ):
            # generate prompt for the current turn
            prompt = prompter.build_turn_prompt(context,turn)

            # rewrite the prompt
            if "mistral" in reformulation_name or "llama" in reformulation_name:
                messages = [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
                responses = rewriter.hf_llm_generate(
                    messages,
                    temperature = 0,
                    top_p = 0.9,
                    max_new_tokens = 4096,
                    do_sample = False,
                    num_beams = 1,
                    num_return_sequences = 1
                )[0]
            else:
                responses = rewriter.generate_text(prompt)

            rewrite_resposne_cot = prompter.parse_returned_text(responses[0]) 
            if rewrite_resposne_cot == None:
                print(f"error with turn id {turn.turn_id}")
                print(turn)
                continue
            rewrite = rewrite_resposne_cot[0]
            response = rewrite_resposne_cot[1]

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
        

        elif reformulation_name == "rar_cot" or reformulation_name == "gpt-4o_rar_cot":
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
            cot = rewrite_resposne_cot[2]

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
        
            turn.add_reformulation(
                reformulation_name = reformulation_name + "_cot",
                reformulated_query = cot,
                ptkb_provenance = []
            )
    
            

    
    #################################
    ## save turn list
    #################################

    save_turns_to_json(turn_list, args.output_query_path)
    


         