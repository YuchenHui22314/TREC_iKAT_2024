#!/bin/bash

# Default argument values
input_query_path="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/ikat_2023_test.json"
output_query_path="/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2023_test.json"
#demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/demonstration_using_ikat24_level.json"
demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json"
#demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/demonstration_using_ikat24.json"
# "mistral-8b",
# "llama3-8b"
rewrite_model="mistral-8b"
#rewrite_model="gpt-4o-2024-08-06"
# P -> personalize, D -> demo, C -> cot, Re -> rel explain
# O -> oracle, Rf -> rel feedback
#reformulaton_name="raw_llm_rm_PDCReORf"
#reformulation_name="raw_llm_rm____Re___"
#reformulation_name="raw_llm_rm_P__Re___"
#reformulation_name="ptkb_summarize"
#reformulation_name="rar_ptkb_sum"
#reformulation_name="rar_ptkb_sum_cot0"
#reformulation_name="rar_ptkb_sum_cot1"
#reformulation_name="rar_personalized_cotN"
#reformulation_name="gpt-4o_rar_personalized_cot1"
#reformulation_name="gpt-4o_rar"
#reformulation_name="gpt-4o_rar_manual_depersonalized_cot1"
#reformulation_name="gpt-3.5_judge_and_rewrite"
reformulation_name="mistral_rar"

LOG_FILE="/data/rech/huiyuche/TREC_iKAT_2024/logs/rewrite_log_2023.txt"

# Run the program with the specified arguments
python3 rewrite.py \
  --input_query_path $input_query_path \
  --output_query_path $output_query_path \
  --demo_file $demo_file \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE
