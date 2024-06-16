#!/bin/bash

INPUT=../../data/indexes/AP_ANCE_512_embeddings
OUTPUT=../../data/indexes/AP_ANCE_512_Faiss_index

python -m pyserini.index.faiss \
  --input ${INPUT} \
  --output ${OUTPUT} \