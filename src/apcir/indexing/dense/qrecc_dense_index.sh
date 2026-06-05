#!/bin/bash

set -euo pipefail

# Dense indexing for QReCC with distributed_dense_index.py.
# Run this inside the GPU job. For long runs, start it in tmux session "yuchen":
#   tmux has-session -t yuchen 2>/dev/null || tmux new-session -d -s yuchen
#   tmux new-window -t yuchen -n qrecc_qwen
#   bash qrecc_dense_index.sh qwen
#
# Before full indexing, do a small end-to-end run by setting:
#   TOTAL_NUM_DOCS=1000 NUM_DOCS_PER_BLOCK=1000 COPY_TO_FINAL=0 bash qrecc_dense_index.sh qwen

MODEL_KIND=${1:-qwen}

REPO_ROOT=${REPO_ROOT:-/data/rech/huiyuche/TREC_iKAT_2024}
SCRIPT_DIR=${SCRIPT_DIR:-${REPO_ROOT}/src/apcir/indexing/dense}
COLLECTION_TSV=${COLLECTION_TSV:-/part/01/Tmp/yuchenhui/qrecc/qrecc_collection.tsv}
CUDA_VISIBLE_DEVICES_ARG=${CUDA_VISIBLE_DEVICES_ARG:-0,1,2,3}
N_GPU=${N_GPU:-4}
TOTAL_NUM_DOCS=${TOTAL_NUM_DOCS:-54573064}
NUM_DOCS_PER_BLOCK=${NUM_DOCS_PER_BLOCK:-1000000}
MAX_DOC_LENGTH=${MAX_DOC_LENGTH:-512}
SEED=${SEED:-42}
COPY_TO_FINAL=${COPY_TO_FINAL:-1}
SKIP_EXISTING_BLOCKS=${SKIP_EXISTING_BLOCKS:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

ANCE_MODEL=${ANCE_MODEL:-/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234}
QWEN_MODEL=${QWEN_MODEL:-/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418}

if [ "${MODEL_KIND}" = "ance" ]; then
    MODEL_TYPE="ance"
    MODEL_PATH="${ANCE_MODEL}"
    # On octal31 RTX A5000, direct single-GPU probe reached bs=1000, but
    # torchrun/DDP with NCCL workaround OOMed at bs=1000. Use a conservative
    # full-run default and override PER_GPU_BATCH_SIZE if probing a new node.
    PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-600}
    LOCAL_OUTPUT=${LOCAL_OUTPUT:-/part/01/Tmp/yuchenhui/indexes/qrecc_ance}
    FINAL_OUTPUT=${FINAL_OUTPUT:-${REPO_ROOT}/data/embeddings/qrecc_ance_merged}
    LOG_FILE=${LOG_FILE:-${REPO_ROOT}/logs/indexing_qrecc_ance.log}
elif [ "${MODEL_KIND}" = "qwen" ] || [ "${MODEL_KIND}" = "qwen3" ]; then
    MODEL_TYPE="qwen-embedding"
    MODEL_PATH="${QWEN_MODEL}"
    PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-600}
    LOCAL_OUTPUT=${LOCAL_OUTPUT:-/part/01/Tmp/yuchenhui/indexes/qrecc_qwen_emb_0.6}
    FINAL_OUTPUT=${FINAL_OUTPUT:-${REPO_ROOT}/data/embeddings/qrecc_qwen_merged}
    LOG_FILE=${LOG_FILE:-${REPO_ROOT}/logs/indexing_qrecc_qwen_emb_0.6.log}
else
    echo "Usage: bash qrecc_dense_index.sh [ance|qwen]"
    exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")" "${LOCAL_OUTPUT}" "$(dirname "${FINAL_OUTPUT}")"

{
    echo "[$(date '+%F %T %Z')] Start QReCC dense indexing"
    echo "hostname=$(hostname)"
    echo "MODEL_KIND=${MODEL_KIND}"
    echo "MODEL_TYPE=${MODEL_TYPE}"
    echo "MODEL_PATH=${MODEL_PATH}"
    echo "COLLECTION_TSV=${COLLECTION_TSV}"
    echo "LOCAL_OUTPUT=${LOCAL_OUTPUT}"
    echo "FINAL_OUTPUT=${FINAL_OUTPUT}"
    echo "CUDA_VISIBLE_DEVICES_ARG=${CUDA_VISIBLE_DEVICES_ARG}"
    echo "N_GPU=${N_GPU}"
    echo "TOTAL_NUM_DOCS=${TOTAL_NUM_DOCS}"
    echo "NUM_DOCS_PER_BLOCK=${NUM_DOCS_PER_BLOCK}"
    echo "PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE}"
    echo "SKIP_EXISTING_BLOCKS=${SKIP_EXISTING_BLOCKS}"
    echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
    echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
    echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

    if [ ! -f "${COLLECTION_TSV}" ]; then
        echo "ERROR: collection TSV does not exist: ${COLLECTION_TSV}"
        exit 1
    fi

    echo "GPU status before indexing:"
    nvidia-smi || true

    EXTRA_ARGS=()
    if [ "${SKIP_EXISTING_BLOCKS}" = "1" ]; then
        EXTRA_ARGS+=(--skip_existing_blocks)
    fi

    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    torchrun --nproc_per_node "${N_GPU}" "${SCRIPT_DIR}/distributed_dense_index.py" \
        --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_ARG}" \
        --n_gpu "${N_GPU}" \
        --model_type "${MODEL_TYPE}" \
        --collection_path "${COLLECTION_TSV}" \
        --pretrained_doc_encoder_path "${MODEL_PATH}" \
        --output_index_dir_path "${LOCAL_OUTPUT}" \
        --seed "${SEED}" \
        --use_data_percent 1.0 \
        --per_gpu_index_batch_size "${PER_GPU_BATCH_SIZE}" \
        --num_docs_per_block "${NUM_DOCS_PER_BLOCK}" \
        --total_num_docs "${TOTAL_NUM_DOCS}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --do_dense_indexing \
        "${EXTRA_ARGS[@]}"

    echo "[$(date '+%F %T %Z')] Dense encoding done, start merge"
    mkdir -p "${LOCAL_OUTPUT}_merged"

    NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}" \
    NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    torchrun --nproc_per_node "${N_GPU}" "${SCRIPT_DIR}/distributed_dense_index.py" \
        --cuda_visible_devices "${CUDA_VISIBLE_DEVICES_ARG}" \
        --n_gpu "${N_GPU}" \
        --output_index_dir_path "${LOCAL_OUTPUT}" \
        --seed "${SEED}" \
        --use_data_percent 1.0 \
        --num_docs_per_block "${NUM_DOCS_PER_BLOCK}" \
        --total_num_docs "${TOTAL_NUM_DOCS}" \
        --max_doc_length "${MAX_DOC_LENGTH}" \
        --do_merge

    if [ "${COPY_TO_FINAL}" = "1" ]; then
        echo "[$(date '+%F %T %Z')] Copy merged blocks to ${FINAL_OUTPUT}"
        mkdir -p "${FINAL_OUTPUT}"
        cp -n "${LOCAL_OUTPUT}_merged"/doc_emb_block.*.pb "${FINAL_OUTPUT}/"
        cp -n "${LOCAL_OUTPUT}_merged"/doc_embid_block.*.pb "${FINAL_OUTPUT}/"
    else
        echo "COPY_TO_FINAL=${COPY_TO_FINAL}; keep merged blocks only at ${LOCAL_OUTPUT}_merged"
    fi

    echo "[$(date '+%F %T %Z')] QReCC dense indexing pipeline done"
} 2>&1 | tee -a "${LOG_FILE}"
