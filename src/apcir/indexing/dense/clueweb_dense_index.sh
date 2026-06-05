#!/bin/bash
# Dense-encode the iKAT-2023 ClueWeb22-B (v2) collection with a dense model
# (qwen3-emb-0.6B or ANCE) on octal31's 4x RTX A5000.
# Keeps UNMERGED per-rank blocks (same layout as data/embeddings/ance_clueweb22B),
# stages on local /part/01 then copies to /data/rech.
#
# Usage:
#   PER_GPU_BATCH_SIZE=256 bash clueweb_dense_index.sh qwen
#   bash clueweb_dense_index.sh ance
set -uo pipefail

MODEL=${1:-qwen}

REPO_ROOT=/data/rech/huiyuche/TREC_iKAT_2024
SCRIPT_DIR=${REPO_ROOT}/src/apcir/indexing/dense
COLLECTION=${COLLECTION:-${REPO_ROOT}/data/collections/ikat_23/cluweb22B_ikat_v2.tsv}

# Absolute torchrun (conda-activate in a non-interactive tmux shell sometimes
# does not put the env bin on PATH -> "torchrun: command not found"). Hardcode it.
TORCHRUN=${TORCHRUN:-/data/rech/huiyuche/envs/trec_ikat/bin/torchrun}

N_GPU=${N_GPU:-4}
CUDA_VISIBLE_DEVICES_ARG=${CUDA_VISIBLE_DEVICES_ARG:-0,1,2,3}
TOTAL_NUM_DOCS=${TOTAL_NUM_DOCS:-116838987}
NUM_DOCS_PER_BLOCK=${NUM_DOCS_PER_BLOCK:-1000000}
MAX_DOC_LENGTH=${MAX_DOC_LENGTH:-512}
DO_MERGE=${DO_MERGE:-0}                 # keep unmerged per-rank blocks
COPY_TO_FINAL=${COPY_TO_FINAL:-1}
SKIP_EXISTING_BLOCKS=${SKIP_EXISTING_BLOCKS:-1}

QWEN_MODEL=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
ANCE_MODEL=/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234

if [ "$MODEL" = "qwen" ] || [ "$MODEL" = "qwen3" ]; then
    MODEL_TYPE=qwen-embedding
    MODEL_PATH=$QWEN_MODEL
    PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-256}
    LOCAL_OUT=/part/01/Tmp/yuchenhui/indexes/clueweb22B_qwen_emb_0.6
    FINAL_OUT=${REPO_ROOT}/data/embeddings/qwen_clueweb22B
    LOG=${REPO_ROOT}/logs/indexing_clueweb22B_qwen_emb_0.6.log
elif [ "$MODEL" = "ance" ]; then
    MODEL_TYPE=ance
    MODEL_PATH=$ANCE_MODEL
    PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-600}
    LOCAL_OUT=/part/01/Tmp/yuchenhui/indexes/clueweb22B_ance
    FINAL_OUT=${REPO_ROOT}/data/embeddings/ance_clueweb22B_v2
    LOG=${REPO_ROOT}/logs/indexing_clueweb22B_ance.log
else
    echo "unknown model '$MODEL' (use qwen|ance)"; exit 1
fi

mkdir -p "$LOCAL_OUT" "$(dirname "$LOG")"

{
echo "================ $(date) ClueWeb22-B dense encode ================"
echo "model=$MODEL type=$MODEL_TYPE bs=$PER_GPU_BATCH_SIZE n_gpu=$N_GPU"
echo "collection=$COLLECTION"
echo "local_out=$LOCAL_OUT  final_out=$FINAL_OUT  do_merge=$DO_MERGE"
echo "total_docs=$TOTAL_NUM_DOCS  per_block=$NUM_DOCS_PER_BLOCK  max_len=$MAX_DOC_LENGTH"

EXTRA=""
[ "$SKIP_EXISTING_BLOCKS" = "1" ] && EXTRA="--skip_existing_blocks"

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --nproc_per_node "$N_GPU" "${SCRIPT_DIR}/distributed_dense_index.py" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES_ARG" --n_gpu "$N_GPU" \
    --model_type "$MODEL_TYPE" \
    --collection_path "$COLLECTION" \
    --pretrained_doc_encoder_path "$MODEL_PATH" \
    --output_index_dir_path "$LOCAL_OUT" \
    --seed 42 --use_data_percent 1.0 \
    --per_gpu_index_batch_size "$PER_GPU_BATCH_SIZE" \
    --num_docs_per_block "$NUM_DOCS_PER_BLOCK" \
    --total_num_docs "$TOTAL_NUM_DOCS" \
    --max_doc_length "$MAX_DOC_LENGTH" \
    --do_dense_indexing $EXTRA
RC=$?
echo "encode rc=$RC"
[ $RC -ne 0 ] && { echo "ENCODE FAILED, abort before copy"; exit $RC; }

if [ "$DO_MERGE" = "1" ]; then
    echo "merging..."
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    "$TORCHRUN" --nproc_per_node "$N_GPU" "${SCRIPT_DIR}/distributed_dense_index.py" \
        --cuda_visible_devices "$CUDA_VISIBLE_DEVICES_ARG" --n_gpu "$N_GPU" \
        --output_index_dir_path "$LOCAL_OUT" --seed 42 --use_data_percent 1.0 \
        --num_docs_per_block "$NUM_DOCS_PER_BLOCK" --total_num_docs "$TOTAL_NUM_DOCS" \
        --max_doc_length "$MAX_DOC_LENGTH" --do_merge
    SRC="${LOCAL_OUT}_merged"
else
    SRC="$LOCAL_OUT"
fi

if [ "$COPY_TO_FINAL" = "1" ]; then
    echo "copying $SRC -> $FINAL_OUT"
    mkdir -p "$FINAL_OUT"
    cp -n "$SRC"/doc_emb_block.*.pb "$FINAL_OUT"/ 2>/dev/null
    cp -n "$SRC"/doc_embid_block.*.pb "$FINAL_OUT"/ 2>/dev/null
    echo "final block count: $(ls "$FINAL_OUT"/doc_emb_block.*.pb 2>/dev/null | wc -l)"
fi
echo "================ $(date) DONE ================"
} 2>&1 | tee -a "$LOG"
