#!/bin/bash

# Default argument values
collection="ClueWeb_ikat"
topics="ikat_23_test"
input_query_path="../../data/topics/ikat_2023_test.json"
index_dir_path="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2"
output_dir_path="../../results"
qrel_file_path="../../data/qrels/ikat_23_qrel.txt"
retrieval_model="BM25"
reranker="None"
dense_query_encoder_path="castorini/ance-msmarco-passage"
bm25_k1=0.9
bm25_b=0.4
top_k=1000
metrics="map,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100"
save_metrics_to_object=false
rewrite_model="no_rewrite"
retrieval_query_type="oracle_utterance"
reranking_query_type="oracle_utterance"
generation_query_type="oracle_utterance"
prompt_type="no_prompt"

# Run the program with default arguments
python3 your_program.py \
  --collection $collection \
  --topics $topics \
  --input_query_path $input_query_path \
  --index_dir_path $index_dir_path \
  --output_dir_path $output_dir_path \
  --qrel_file_path $qrel_file_path \
  --retrieval_model $retrieval_model \
  --reranker $reranker \
  --dense_query_encoder_path $dense_query_encoder_path \
  --bm25_k1 $bm25_k1 \
  --bm25_b $bm25_b \
  --top_k $top_k \
  --metrics $metrics \
  ${save_metrics_to_object:+--save_metrics_to_object} \
  --rewrite_model $rewrite_model \
  --retrieval_query_type $retrieval_query_type \
  --reranking_query_type $reranking_query_type \
  --generation_query_type $generation_query_type \
  --prompt_type $prompt_type
