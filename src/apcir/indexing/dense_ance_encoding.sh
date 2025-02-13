#!/bin/bash

# Check if input parameters are provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <shard-num> <input> <output> <log>"
    exit 1
fi

SHARD_NUM=$1
INPUT=$2
OUTPUT=$3
LOG=$4
ENCODER=/home/yuchenxi/projects/def-jynie/yuchenxi/huggingface_models/ance-msmarco-passage

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

    python -m pyserini.encode \
            input   --corpus ${INPUT} \
                    --fields text \
                    --delimiter shutongyu \
                    --shard-id ${shard} \
                    --shard-num ${SHARD_NUM} \
            output  --embeddings ${OUTPUT}/shard_${shard} \
            encoder --encoder ${ENCODER} \
                    --encoder-class ance \
                    --fields text \
                    --batch 16 \
                    --fp16 \
                    --max-length 512 \
                    --dimension 768 \
                    --device cuda:${shard} \
                    > ${LOG} &
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



