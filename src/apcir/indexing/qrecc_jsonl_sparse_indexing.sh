#!/bin/bash

set -euo pipefail

# QReCC BM25 indexing.
# Input should be the extracted official passages.zip directory after
# paragraph_chunker.py, containing commoncrawl/, wayback/, and wayback-backfill/.

INPUT=${INPUT:-/part/01/Tmp/yuchenhui/qrecc_raw/collection-paragraph}
OUTPUT=${OUTPUT:-/data/rech/huiyuche/TREC_iKAT_2024/data/indexes/qrecc_official_sparse_index}
LOG_FILE=${LOG_FILE:-/data/rech/huiyuche/TREC_iKAT_2024/logs/indexing_qrecc_sparse.log}
THREADS=${THREADS:-40}

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${OUTPUT}")"

{
    echo "[$(date '+%F %T %Z')] Start QReCC sparse indexing"
    echo "INPUT=${INPUT}"
    echo "OUTPUT=${OUTPUT}"
    echo "THREADS=${THREADS}"

    if [ ! -d "${INPUT}" ]; then
        echo "ERROR: input collection directory does not exist: ${INPUT}"
        exit 1
    fi

    if [ -d "${OUTPUT}" ]; then
        echo "Skip: output index already exists: ${OUTPUT}"
        echo "Set OUTPUT to a new directory if you want to rebuild. No files are deleted by this script."
        exit 0
    fi

    python -m pyserini.index \
        -collection JsonCollection \
        -generator DefaultLuceneDocumentGenerator \
        -threads "${THREADS}" \
        -input "${INPUT}" \
        -index "${OUTPUT}" \
        -storePositions -storeDocvectors -storeRaw

    echo "[$(date '+%F %T %Z')] QReCC sparse indexing done"
} 2>&1 | tee -a "${LOG_FILE}"
