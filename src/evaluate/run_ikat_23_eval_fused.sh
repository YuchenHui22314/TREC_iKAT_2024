#!/bin/bash
# rewrites choices
# P -> personalize, D -> demo, C -> cot, Re -> rel explain
# O -> oracle, Rf -> rel feedback
# Default argument values

#### modify (1) given rankinglist path (2) retrieval_query_types (3) retrieval model to none (4) write query type to evaluation.py if never run 

collection="ClueWeb_ikat"
topics="ikat_23_test"
input_query_path="../../data/topics/ikat23/ikat_2023_test.json"
index_dir_path="/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2/" # Please use local disk index to achieve the fastest access
#index_dir_path="/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_official_sparse_index/" # Please use local disk index to achieve the fastest access
output_dir_path="../../results"
qrel_file_path="../../data/qrels/ikat_23_qrel.txt"
retrieval_model="none"
cache_dir="/data/rech/huiyuche/huggingface"
dense_query_encoder_path="castorini/ance-msmarco-passage"
# none, rankllama, rankgpt, monot5_base, monot5_base_10k
reranker="none"
#rankllama
rerank_quant="none" # can be "none" ,"8b", "4b"
#rankgpt
rankgpt_llm="gpt-3.5-turbo"
window_size=4
step=1
#BM25
bm25_k1=0.9
bm25_b=0.4
#qe
qe_type="none"
fb_terms=20
fb_docs=10
original_query_weight=0.5
#response generation:
generation_model="none"
generation_prompt="none"
#number of documents to consider at each step.
retrieval_top_k=1000
rerank_top_k=50
generation_top_k=3
metrics="map,ndcg,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,P.20,recall.5,recall.10,recall.20,recall.50,recall.100,recall.1000,recip_rank"
given_ranking_list_path="/data/rech/huiyuche/TREC_iKAT_2024/results/ClueWeb_ikat/ikat_23_test/ranking/S1[gpt-4o_rar_rwrs_fuse_personalized_cot1_rw]-S2[none]-g[none]-[none]-[none_4_1_none]-[s2_top50].txt"
# project specific
run_name="none"
# raw_llm_rm_PDCReORf
retrieval_query_type="none"
reranking_query_type="none"
generation_query_type="none"
prompt_type="no_prompt"

LOG_FILE=/data/rech/huiyuche/TREC_iKAT_2024/logs/evaluation_log_2023.txt

#retrieval_query_types=("raw" "rar_rw" "rar_rwrs" "oracle_utterance" "raw_llm_rm_P__Re___" "raw_llm_rm____Re___" "rar_ptkb_sum_cot0_rw" "rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rw" "rar_ptkb_sum_rwrs")
#reranking_query_types=("raw" "rar_rw" "rar_rwrs" "oracle_utterance" "rar_ptkb_sum_cot0_rw" "rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rw" "rar_ptkb_sum_rwrs" "raw_llm_rm_P__Re___" "raw_llm_rm____Re___")
#reranking_query_types=("raw" "rar_rw" "rar_rwrs" "oracle_utterance" "rar_ptkb_sum_cot0_rw" "rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rw" "rar_ptkb_sum_rwrs" "raw_llm_rm_P__Re___" "raw_llm_rm____Re___")
#reranking_query_types=("rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rwrs" "raw_llm_rm_P__Re___" "raw_llm_rm____Re___")
# reranking_query_types=("rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rwrs")
#retrieval_query_types=("raw" "rar_rw" "rar_rwrs" "oracle_utterance" "raw_llm_rm_P__Re___" "raw_llm_rm____Re___" "rar_ptkb_sum_cot0_rw" "rar_ptkb_sum_cot0_rwrs" "rar_ptkb_sum_rw" "rar_ptkb_sum_rwrs")
#retrieval_query_types=("rar_personalized_cot1_rw" "rar_personalized_cotN_rw")
#retrieval_query_types=("rar_rw_fuse_rar_personcot1_rw")
# retrieval_query_types=("rar_rw_fuse_rar_rwrs")
# reranking_query_types=("rar_personalized_cot1_rw")
#retrieval_query_types=("gpt-4o_rar_rw" "gpt-4o_rar_rwrs" "gpt-4o_rar_personalized_cot1_rw")
# retrieval_query_types=("gpt-4o_rar_rw_fuse_rar_rwrs_fuse_personalized_cot1_rw") 
#retrieval_query_types=("gpt-4o_rar_rw_fuse_rar_rwrs_fuse_manual_depersonalized_cot1_rw")
retrieval_query_types=("gpt-4o_rar_rwrs_fuse_personalized_cot1_rw") 
reranking_query_types=("none")
generation_query_types=("none")


LOG_FILE=/data/rech/huiyuche/TREC_iKAT_2024/logs/evaluation_log_2023.txt

function run_evaluation() {
    local retrieval_query_type="$1"
    local reranking_query_type="$2"
    local generation_query_type="$3"

    echo "Running with retrieval_query_type: $retrieval_query_type, reranking_query_type: $reranking_query_type", generation_query_type: $generation_query_type

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
    --given_ranking_list_path $given_ranking_list_path \
    --rerank_quant $rerank_quant \
    --rankgpt_llm $rankgpt_llm \
    --window_size $window_size \
    --step $step \
    --generation_model $generation_model \
    --qe_type $qe_type \
    --bm25_k1 $bm25_k1 \
    --bm25_b $bm25_b \
    --fb_terms $fb_terms \
    --fb_docs $fb_docs \
    --original_query_weight $original_query_weight \
    --generation_prompt $generation_prompt \
    --generation_model $generation_model \
    --retrieval_top_k $retrieval_top_k \
    --rerank_top_k $rerank_top_k \
    --generation_top_k $generation_top_k \
    --metrics $metrics \
    --save_ranking_list \
    --run_rag \
    --run_eval \
    --run_name $run_name \
    --save_results_to_object \
    --retrieval_query_type $retrieval_query_type \
    --reranking_query_type $reranking_query_type \
    --generation_query_type $generation_query_type &>> $LOG_FILE
}

    # --save_results_to_object \


for retrieval_query_type in "${retrieval_query_types[@]}"
do
    for reranking_query_type in "${reranking_query_types[@]}"
    do
        for generation_query_type in "${generation_query_types[@]}"
        do
            run_evaluation $retrieval_query_type $reranking_query_type $generation_query_type
        done
    done
done