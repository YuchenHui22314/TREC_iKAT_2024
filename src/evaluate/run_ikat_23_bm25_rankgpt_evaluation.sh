#!/bin/bash

# Default argument values
collection="ClueWeb_ikat"
topics="ikat_23_test"
input_query_path="../../data/topics/ikat_2023_test.json"
index_dir_path="/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2/" # Please use local disk index to achieve the fastest access
index_dir_path="/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_official_sparse_index/" # Please use local disk index to achieve the fastest access
output_dir_path="../../results"
qrel_file_path="../../data/qrels/ikat_23_qrel.txt"
retrieval_model="BM25"
cache_dir="/data/rech/huiyuche/huggingface"
dense_query_encoder_path="castorini/ance-msmarco-passage"
# rankllama, rankgpt, monot5_base, monot5_base_10k
reranker="none"
#rankllama
rerank_quant="none" # can be "none" ,"8b", "4b"
#rankgpt
rankgpt_llm="gpt-3.5-turbo"
window_size=4
step=1
#response generation:
generation_model="none"
#BM25
bm25_k1=0.9
bm25_b=0.4
#rm3
fb_terms=10
fb_docs=10
original_query_weight=0.9
retrieval_top_k=1000
rerank_top_k=50
generation_top_k=3
metrics="map,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,recall.5,recip_rank"
save_metrics_to_object=false
# project specific
run_name="none"
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
  --rerank_quant $rerank_quant \
  --rankgpt_llm $rankgpt_llm \
  --window_size $window_size \
  --step $step \
  --generation_model $generation_model \
  --bm25_k1 $bm25_k1 \
  --bm25_b $bm25_b \
  --fb_terms $fb_terms \
  --fb_docs $fb_docs \
  --original_query_weight $original_query_weight \
  --retrieval_top_k $retrieval_top_k \
  --rerank_top_k $rerank_top_k \
  --generation_top_k $generation_top_k \
  --metrics $metrics \
  ${save_metrics_to_object:+--save_metrics_to_object} \
  --run_name $run_name \
  --rewrite_model $rewrite_model \
  --retrieval_query_type $retrieval_query_type \
  --reranking_query_type $reranking_query_type \
  --generation_query_type $generation_query_type \
  --prompt_type $prompt_type &>> $LOG_FILE
  #--use_rm3 \
  #--just_run_no_evaluate \