#!/bin/bash

# Default argument values
cache_dir="/data/rech/huiyuche/huggingface"
output_query_path="/data/rech/huiyuche/TREC_iKAT_2024/test/fengran_10_qr_prag_GRF.json"
input_query_path="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/fengran_10_qr_GRF_mistral.json"
# not important
demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/demonstration_using_ikat24_level.json"
rewrite_model="vllm_mistral"
# reformulation_name="llama3.1_fengran_10_GRF"
# reformulation_name="llama3.1_fengran_10_qr_ctx"
# reformulation_name="llama3.1_fengran_10_qr"
reformulation_name="vllm_mistral_fengran_prag_10_GRF"


LOG_FILE="/data/rech/huiyuche/TREC_iKAT_2024/logs/rewrite_log_topiocqa.txt"

cd ../../
# Run the program with the specified arguments
python -m apcir.rewrite.rewrite_topiocqa_vllm \
  --input_query_path $input_query_path \
  --output_query_path $output_query_path \
  --cache_dir $cache_dir \
  --demo_file $demo_file \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE
