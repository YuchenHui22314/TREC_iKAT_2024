#!/bin/bash

# Default argument values
cache_dir="/data/rech/huiyuche/huggingface"
input_query_path="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/ikat_2024_test.json"
#output_query_path="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/ikat_2024_test.json"
output_query_path="/data/rech/huiyuche/TREC_iKAT_2024/test/ikat_2024_test.json"
#demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/original_demonstration.json"
#demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat24/demonstration_using_ikat24.json"
#demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/demonstration_using_ikat23.json"
demo_file="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/non_personalized_demonstration_using_ikat23_level.json"

result_file="/data/rech/huiyuche/TREC_iKAT_2024/results/ClueWeb_ikat/ikat_24_test/ranking/S1[gpt-4o_rar_rw]-S2[none]-g[none]-[splade_v3]-[none_4_1_none]-[s2_top50].txt"
deps_entropy_top_k=10

#rewrite_model="gpt-3.5-turbo" # gpt-3.5-turbo-0125 "gpt-4o-2024-08-06"
#rewrite_model="gpt-4o-2024-08-06"
#rewrite_model="gpt-4-0613"
#rewrite_model="mistral-8b"
rewrite_model="llama3-8b"
rewrite_model="none"
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
#reformulation_name="rar_personalized_cot1"
#reformulation_name="gpt-4o_rar_personalized_cot1"
#reformulation_name="rar_cot"
#reformulation_name="gpt-4o_rar"
#reformulation_name="gpt-4o_rar_non_personalized_cot1"
#reformulation_name="gpt-3.5_judge_and_rewrite"
#reformulation_name="mistral_rar"
#reformulation_name="llama3.1_judge_and_rewrite"
#reformulation_name="gpt-4o_MQ4CS_mq"
#reformulation_name="gpt-4o_MQ4CS_persq"
#reformulation_name="gpt-4o_jtr_wo_cot"
#reformulation_name="gpt-4o_jtr_wo_in_context"
#reformulation_name="gpt-4o_MQ4CS_mq_3"
#reformulation_name="gpt-4o_GtR_rs"
#reformulation_name="gpt-4o_GtR_mq_3"
#reformulation_name="llama3.1_MQ4CS_persq"
# reformulation_name="DEPS"
reformulation_name="result_topic_entropy"

LOG_FILE="/data/rech/huiyuche/TREC_iKAT_2024/logs/rewrite_log_2024.txt"

cd ../../
# Run the program with the specified arguments
python -m apcir.rewrite.rewrite \
  --input_query_path $input_query_path \
  --output_query_path $output_query_path \
  --demo_file $demo_file \
  --result_file $result_file \
  --cache_dir $cache_dir \
  --rewrite_model $rewrite_model \
  --reformulation_name $reformulation_name &>> $LOG_FILE