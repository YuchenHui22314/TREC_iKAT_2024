#!/bin/bash

# Default argument values
collection="ClueWeb_ikat"
topics="ikat_23_test"
input_query_path="../../data/topics/ikat_2023_test.json"
index_dir_path="/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2/" # Please use local disk index to achieve the fastest access
output_dir_path="../../results"
qrel_file_path="../../data/qrels/ikat_23_qrel.txt"
retrieval_model="BM25"
cache_dir="/data/rech/huiyuche/huggingface"
dense_query_encoder_path="castorini/ance-msmarco-passage"
reranker="rankgpt"
rankgpt_llm="gpt-3.5-turbo"
window_size=3
step=1
bm25_k1=0.9
bm25_b=0.4
top_k=3
metrics="map,ndcg_cut.1,ndcg_cut.3,P.1,P.3,recall.3"
save_metrics_to_object=false
rewrite_model="no_rewrite"
retrieval_query_type="oracle_utterance"
reranking_query_type="oracle_utterance"
generation_query_type="oracle_utterance"
prompt_type="no_prompt"

LOG_FILE=/data/rech/huiyuche/TREC_iKAT_2024/logs/evaluation_log.txt

# Run the program with default arguments
python3 evaluation.py \
  --collection $collection \
  --topics $topics \
  --input_query_path $input_query_path \
  --index_dir_path $index_dir_path \
  --output_dir_path $output_dir_path \
  --qrel_file_path $qrel_file_path \
  --retrieval_model $retrieval_model \
  --cache_dir $cache_dir \
  --dense_query_encoder_path $dense_query_encoder_path \
  --reranker $reranker \
  --rankgpt_llm $rankgpt_llm \
  --window_size $window_size \
  --step $step \
  --bm25_k1 $bm25_k1 \
  --bm25_b $bm25_b \
  --top_k $top_k \
  --metrics $metrics \
  ${save_metrics_to_object:+--save_metrics_to_object} \
  --rewrite_model $rewrite_model \
  --retrieval_query_type $retrieval_query_type \
  --reranking_query_type $reranking_query_type \
  --generation_query_type $generation_query_type \
  --prompt_type $prompt_type &>> $LOG_FILE