OUTPUT=/data/rech/huiyuche/TREC_iKAT_2024/data/indexes/clueweb22b_ikat23_fengran_sparse_index_2/
INPUT=/data/rech/huiyuche/TREC_iKAT_2024/data/collections/ikat_23/
LOG_FILE=/data/rech/huiyuche/TREC_iKAT_2024/logs/indexing_log.txt

if [ ! -f "$OUTPUT" ]; then
    echo "Creating index..."
    python -m pyserini.index -collection JsonCollection \
                            -generator DefaultLuceneDocumentGenerator \
                            -threads 40 \
                            -input ${INPUT} \
                            -index ${OUTPUT} \
							-storePositions -storeDocvectors -storeRaw &>> $LOG_FILE
fi