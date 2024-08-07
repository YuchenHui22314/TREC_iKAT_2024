#!/bin/bash

# Default argument values
input_query_path="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/ikat_2023_test.json"
demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json"
rewrite_model="gpt-3.5-turbo"
# P -> personalize, D -> demo, C -> cot, Re -> rel explain
# O -> oracle, Rf -> rel feedback
# "raw_llm_rm_PDCReORf"
#reformulation_name="raw_llm_rm____Re___"
reformulation_name="rar"

LOG_FILE="/data/rech/huiyuche/TREC_iKAT_2024/logs/rewrite_log.txt"

# Run the program with the specified arguments
python3 rewrite.py \
  --input_query_path $input_query_path \
  --demo_file $demo_file \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE