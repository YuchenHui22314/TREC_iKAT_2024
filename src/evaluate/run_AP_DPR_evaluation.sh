#!/bin/bash

# Define fixed arguments
collection="AP"
input_query_path="../../data/topics/trec_fire_covid_data.json"
index_dir_path="../../data/indexes/AP_sparse_index"
ann_index_dir_path="../../data/indexes/AP_DPR_Faiss_index"
output_dir_path="../../results"
qrel_file_path="../../data/qrels/AP_qrels.1-150.txt"
eval_type="dpr"
dense_query_encoder_path="facebook/dpr-question_encoder-multiset-base"
bm25_k1=0.9
bm25_b=0.4
top_k=1000
metrics="map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100"

# List of all query_types to loop through
query_types=("title" "description" "narrative" "title+description" "title+narrative" "description+narrative" "title+description+narrative")

# Loop through each query_type and run the Python program
for query_type in "${query_types[@]}"
do
    echo "Running with query_type: $query_type"
    python evaluation.py \
        --collection "$collection" \
        --input_query_path "$input_query_path" \
        --index_dir_path "$index_dir_path" \
        --ann_index_dir_path "$ann_index_dir_path" \
        --output_dir_path "$output_dir_path" \
        --qrel_file_path "$qrel_file_path" \
        --query_type "$query_type" \
        --eval_type "$eval_type" \
        --dense_query_encoder_path "$dense_query_encoder_path" \
        --bm25_k1 "$bm25_k1" \
        --bm25_b "$bm25_b" \
        --top_k "$top_k" \
        --metrics "$metrics"
done
