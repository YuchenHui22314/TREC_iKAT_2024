OUTPUT=/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_official_sparse_index/
INPUT=/part/01/Tmp/yuchen/ikat_official_collectioin/
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