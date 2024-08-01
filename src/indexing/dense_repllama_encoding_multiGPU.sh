#!/bin/bash

SHARD_NUM=1
INPUT="/part/01/Tmp/yuchen/ikat_official_collection/"
OUTPUT="/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/repllama_clueweb22B"
LOG="/data/rech/huiyuche/TREC_iKAT_2024/logs/indexing_log.txt"
ENCODER="/home/yuchenxi/projects/def-jynie/yuchenxi/huggingface_models/ance-msmarco-passage"

# if output directory does not exist, create it
if [-d ${OUTPUT}]; then
    echo "Output directory ${OUTPUT} already exists."
else 
    mkdir -p ${OUTPUT}
fi

# Function to run encoding for each shard
encode_shard() {
    local shard=$1
    echo "Creating embedding for shard ${shard}..."

    python /data/rech/huiyuche/TREC_iKAT_2024/src/indexing/indexing_repllama.py \
            input   --corpus ${INPUT} \
                    --fields text \
                    --shard-id ${shard} \
                    --shard-num ${SHARD_NUM} \
            output  --embeddings ${OUTPUT}/shard_${shard} \
            encoder --encoder ${ENCODER} \
                    --cache_dir /data/rech/huiyuche/huggingface \
                    --encoder-class repllama \
                    --fields text \
                    --batch 25 \
                    --max-length 2048 \
                    --dimension 4096 \
                    --device cuda:${shard} &>> ${LOG} &
}

# Run encoding for each shard concurrently
for shard in $(seq 0 $((SHARD_NUM - 1)))
do
    encode_shard "${shard}"
done

# Wait for all background processes to finish
wait

# Merge all shard embeddings into a single file
find ${OUTPUT} -type f -name 'embeddings.jsonl' ! -path "${OUTPUT}/embeddings.jsonl" -exec cat {} + > ${OUTPUT}/embeddings.jsonl

# Remove temporary shard embeddings
rm -r ${OUTPUT}/shard_*



