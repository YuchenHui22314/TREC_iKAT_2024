#!/bin/bash

# Define fixed arguments
collection="AP"
input_query_path="../../data/topics/trec_fire_covid_data.json"
index_dir_path="../../data/indexes/AP_sparse_index"
output_dir_path="../../results"
qrel_file_path="../../data/qrels/AP_qrels.1-150.txt"
retrieval_model="BM25"
rewrite_model="gpt-4-turbo"
bm25_k1=0.9
bm25_b=0.4
top_k=1000
metrics="map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.10,recall.20,recall.100,recall.1000"


# List of all query_types to loop through
#query_types=("title" "description" "narrative" "title+description" "title+narrative" "description+narrative" "title+description+narrative" "reformulation" "pseudo_narrative")
query_types=("reformulation" "pseudo_narrative")

# List of all prompt types for reformulation query_type
#reformulation_prompt_types=("few_shot_narrative_prompt" "complex_few_shot_narrative_prompt" "real_narrative_prompt" "complex_real_narrative_prompt" "few_shot_pseudo_narrative_only_prompt" "complex_few_shot_pseudo_narrative_only_prompt")
reformulation_prompt_types=("few_shot_pseudo_narrative_only_prompt" "complex_few_shot_pseudo_narrative_only_prompt")

# List of all prompt types for pseduo_narrative query_type
 pseduo_narrative_prompt_types=("few_shot_pseudo_narrative_only_prompt" "complex_few_shot_pseudo_narrative_only_prompt")

function run_evaluation() {
    local query_type="$1"
    local prompt_type="$2"

    echo "Running with query_type: $query_type and prompt_type: $prompt_type"
    python evaluation.py \
        --collection "$collection" \
        --input_query_path "$input_query_path" \
        --index_dir_path "$index_dir_path" \
        --output_dir_path "$output_dir_path" \
        --qrel_file_path "$qrel_file_path" \
        --query_type "$query_type" \
        --retrieval_model "$retrieval_model" \
        --rewrite_model "$rewrite_model" \
        --prompt_type "$prompt_type" \
        --dense_query_encoder_path "$dense_query_encoder_path" \
        --bm25_k1 "$bm25_k1" \
        --bm25_b "$bm25_b" \
        --top_k "$top_k" \
        --metrics "$metrics" \
        --run_reformulate \
        --save_metrics_to_object
}


# Loop through each query_type and run the Python program
for query_type in "${query_types[@]}"
do
    if [ "$query_type" == "reformulation" ] ; then
        for prompt_type in "${reformulation_prompt_types[@]}"
        do
            run_evaluation "$query_type" "$prompt_type"
        done
    elif [ "$query_type" == "pseudo_narrative" ] ; then
        for prompt_type in "${pseduo_narrative_prompt_types[@]}"
        do
            run_evaluation "$query_type" "$prompt_type"
        done
    else
        run_evaluation "$query_type" "few_shot_narrative_prompt"
    fi
done