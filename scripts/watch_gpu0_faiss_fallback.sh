#!/usr/bin/env bash
set -euo pipefail

MAIN_LOG="/data/rech/huiyuche/TREC_iKAT_2024/logs/beir_qwen_emb_0.6b_log.txt"
WATCHDOG_LOG="/data/rech/huiyuche/TREC_iKAT_2024/logs/beir_qwen_gpu0_watchdog.txt"
WORKDIR="/data/rech/huiyuche/TREC_iKAT_2024/src"
PYTHON_BIN="/data/rech/huiyuche/envs/trec_ikat/bin/python"
TMUX_SESSION="beir_qwen_gpu0_encode_only"
EMBEDDING_BASE_PATH="/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
CORPUS_CHUNK_SIZE=50000

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_line() {
  printf '[%s] %s\n' "$(timestamp)" "$1" | tee -a "$WATCHDOG_LOG"
}

count_shards() {
  local dataset="$1"
  find "${EMBEDDING_BASE_PATH}/${dataset}" -maxdepth 1 -type f -name 'corpus.*.pkl' | wc -l
}

expected_shards() {
  local dataset="$1"
  DATASET_NAME="$dataset" CORPUS_CHUNK_SIZE="$CORPUS_CHUNK_SIZE" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

dataset = os.environ["DATASET_NAME"]
chunk_size = int(os.environ["CORPUS_CHUNK_SIZE"])
base = Path("/data/rech/huiyuche/beir") / dataset / dataset
split = "dev" if dataset == "msmarco" else "test"
corpus, _, _ = GenericDataLoader(str(base)).load(split=split)
print((len(corpus) + chunk_size - 1) // chunk_size)
PY
}

is_dataset_encoded() {
  local dataset="$1"
  local existing expected
  existing="$(count_shards "$dataset" | tr -d '[:space:]')"
  expected="$(expected_shards "$dataset" | tail -n 1 | tr -d '[:space:]')"
  [[ -n "$existing" && -n "$expected" && "$existing" = "$expected" ]]
}

build_remaining_datasets() {
  local datasets=()

  if ! is_dataset_encoded "msmarco"; then
    datasets+=("msmarco")
  fi
  if ! is_dataset_encoded "fiqa"; then
    datasets+=("fiqa")
  fi
  if ! is_dataset_encoded "scidocs"; then
    datasets+=("scidocs")
  fi
  if ! is_dataset_encoded "fever"; then
    datasets+=("fever")
  fi

  printf '%s\n' "${datasets[@]}"
}

mkdir -p "$(dirname "$WATCHDOG_LOG")"
touch "$WATCHDOG_LOG"

log_line "watchdog_start session=${TMUX_SESSION}"

while true; do
  if grep -q "WORKER_DONE worker=0" "$MAIN_LOG"; then
    log_line "worker0_done_detected exiting_watchdog"
    exit 0
  fi

  if grep -q "WORKER_ERROR worker=0" "$MAIN_LOG"; then
    log_line "worker0_error_detected evaluating_remaining_encode_only_datasets"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
      log_line "encode_only_session_already_exists session=${TMUX_SESSION}"
      exit 0
    fi

    mapfile -t remaining_datasets < <(build_remaining_datasets)
    if [[ "${#remaining_datasets[@]}" -eq 0 ]]; then
      log_line "no_remaining_encode_datasets_detected exiting_watchdog"
      exit 0
    fi

    log_line "remaining_encode_datasets=${remaining_datasets[*]}"
    tmux new-session -d -s "$TMUX_SESSION" \
      "cd '$WORKDIR' && CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_NO_TF=1 USE_TF=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PYTHON_BIN' -u -m apcir.evaluate.evaluate_BEIR_qwen3_emb --worker_gpu_id 0 --worker_index 0 --worker_datasets ${remaining_datasets[*]} --batch_size 512 --encode_only 2>&1 | tee -a '$MAIN_LOG'"

    log_line "encode_only_session_started session=${TMUX_SESSION}"
    exit 0
  fi

  sleep 30
done
