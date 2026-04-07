#!/usr/bin/env bash
set -euo pipefail

LOG='/data/rech/huiyuche/TREC_iKAT_2024/logs/beir_qwen_emb_0.6b_log.txt'
RANK_DIR='/data/rech/huiyuche/TREC_iKAT_2024/results/beir/ranking'
METRIC_DIR='/data/rech/huiyuche/TREC_iKAT_2024/results/beir/metrics'
EMB_DIR='/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B'

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "[$ts] SIDECAR_STATUS_BEGIN"
    echo -n "[$ts] GPU_STATUS "
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' '; '
    echo
    echo -n "[$ts] DONE_RANKINGS "
    ls "$RANK_DIR"/beir_qwen3_emb_*_gpu*.pkl 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '
    echo
    echo -n "[$ts] WORKER_JSONS "
    ls "$METRIC_DIR"/beir_qwen3_emb_worker_*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '
    echo
    for d in msmarco scifact trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs nq hotpotqa dbpedia-entity fever climate-fever cqadupstack; do
      if [ -d "$EMB_DIR/$d" ]; then
        c="$(find "$EMB_DIR/$d" -maxdepth 1 -type f 2>/dev/null | wc -l)"
        echo "[$ts] EMB_FILES dataset=$d files=$c"
      fi
    done
    echo "[$ts] SIDECAR_STATUS_END"
  } >> "$LOG"
  sleep 60
done
