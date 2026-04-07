#!/usr/bin/env bash
set -euo pipefail

cd /data/rech/huiyuche/TREC_iKAT_2024/src

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/data/rech/huiyuche/envs/trec_ikat/bin/python -u -m apcir.evaluate.evaluate_BEIR_qwen3_emb \
  --worker_gpu_id 0 \
  --worker_index 0 \
  --worker_datasets fiqa scidocs fever \
  --batch_size 512 \
  --encode_only 2>&1 | tee -a /data/rech/huiyuche/TREC_iKAT_2024/logs/beir_qwen_emb_0.6b_log.txt
