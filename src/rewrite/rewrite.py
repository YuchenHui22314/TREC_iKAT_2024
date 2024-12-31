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
import tkinter as tk
from tkinter import simpledialog
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
from promptor import (
    RewriteAndResponsePromptor,
    PersonalizedCIRQueryExpansionPromptor,
    SummarizePTKBPromptor,
    PersonalizeViaPTKBSummaryPrompter,
    RARPersonalizedCoTPromptor,
    RARNonPersonalizedCoTPromptor,
    JudgePersonalizeLevelPromptor, 
    JudgeThenRewritePromptor
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
    LM,
    OpenAILM
)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/ikat_2023_test.json")

    parser.add_argument("--output_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2024_test.json")

    parser.add_argument("--demo_file", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json")

    parser.add_argument("--cache_dir", type=str, default="/data/rech/huiyuche/huggingface")

    parser.add_argument("--rewrite_model", type=str, default="gpt-3.5-turbo", choices=[
        "gpt-3.5-turbo", 
        "gpt-3.5-turbo-16k", 
        "gpt-4-0613",
        "gpt-4o-2024-08-06",
        "mistral-8b",
        "llama3-8b"])

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

    if "judge_and_rewrite" in reformulation_name:
        prompter =  JudgeThenRewritePromptor(
            enable_cot=True,
            demo_file = demo_file 
        )
        
    if "personalization_level" in reformulation_name:
        prompter = JudgePersonalizeLevelPromptor(
            enable_cot=True
        )

    if reformulation_name == "rar" or reformulation_name == "gpt-4o_rar":
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
            cache_dir="/data/rech/huiyuche/huggingface",
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
        device_map= "auto",
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
        device_map= "auto",
        attn_implementation="flash_attention_2",
        access_token=None,
        cache_dir=args.cache_dir,
        accelerator = None,
        load_in_8bit = False,
        load_in_4bit = False,
        )

    #################################
    ## load topic file and rewrite
    #################################

    turn_list = load_turns_from_json(input_query_path)
    for index, turn in tqdm(enumerate(turn_list), total=len(turn_list), desc="Rewriting"):
        context = get_context_by_qid(turn.turn_id,turn_list)
        if "judge_and_rewrite" in reformulation_name:
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
            
            liste = prompter.parse_returned_text(response[0])
            if liste == None:
                print(f"error with turn id {turn.turn_id}")
                print(response)
                continue

            cot = liste[0]
            level = liste[1]
            rewrite = liste[2]
            response = liste[3]


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
                reformulation_name == "llama3_rar" or
                reformulation_name == "mistral2_rar"
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
    


         