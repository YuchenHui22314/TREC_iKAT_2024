#!/bin/bash

# Default argument values
cache_dir=""        # where you store the hugingface models
input_query_path="../../data/topics/ikat24/ikat_2024_test.json"
output_query_path="../../data/topics/ikat24/ikat_2024_test.json"
demo_file="../../data/topics/ikat23/non_personalized_demonstration_using_ikat23_level.json"
rewrite_model="llama3-8b"
reformulation_name="llama3.1_MQ4CS_persq"

LOG_FILE="../../logs/rewrite_log_2024.txt"

# Run the program with the specified arguments
python3 rewrite.py \
  --input_query_path $input_query_path \
  --output_query_path $output_query_path \
  --demo_file $demo_file \
  --cache_dir $cache_dir \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE