#!/bin/bash
INPUT=../../data/collections/AP_jsonl
OUTPUT=../../data/indexes/AP_DPR_embeddings

if [ ! -f "$OUTPUT" ]; then
    echo "Creating embedding..."
    python -m pyserini.encode \
    input   --corpus ${INPUT} \
            --fields text \
            --delimiter shutongyu \
            --shard-id 0 \
            --shard-num 1 \
    output  --embeddings ${OUTPUT} \
    encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
            --encoder-class dpr \
            --fields text \
            --batch 16 \
            --fp16 \
            --max-length 256 \
            --dimension 768 \
            --device cuda:0 \


fi