#!/bin/bash

# Default argument values
cache_dir=""        # where you store the hugingface models
output_query_path="../../test/ikat_2023_test.json"
input_query_path="../../data/topics/ikat23/ikat_2023_test.json"
demo_file="../../data/topics/ikat24/demonstration_using_ikat24_level.json"
rewrite_model="gpt-4o-2024-08-06"  
reformulation_name="gpt-4o_judge_and_rewrite"

LOG_FILE="/data/rech/huiyuche/TREC_iKAT_2024/logs/rewrite_log_2023.txt"

# Run the program with the specified arguments
python3 rewrite.py \
  --input_query_path $input_query_path \
  --output_query_path $output_query_path \
  --cache_dir $cache_dir \
  --demo_file $demo_file \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE
