#!/bin/bash
# Probe per-GPU batch size for qwen3-emb-0.6B on this node's 4x RTX A5000 (24G),
# using a small slice of the ClueWeb22-B collection. Tries 600 first, then 512.
# Records PEAK GPU memory per batch size so we can extrapolate the max safe bs.
# Writes only to a throwaway dir; does NOT touch final.
set -uo pipefail

REPO_ROOT=/data/rech/huiyuche/TREC_iKAT_2024
SCRIPT_DIR=${REPO_ROOT}/src/apcir/indexing/dense
COLLECTION=${REPO_ROOT}/data/collections/ikat_23/cluweb22B_ikat_v2.tsv
QWEN=/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418
TOTAL=116838987
PROBE_PCT=${PROBE_PCT:-0.0005}   # ~58k docs, enough to fill batches and hit peak mem
TMP=/part/01/Tmp/yuchenhui/indexes/cw_qwen_bsprobe
CANDIDATES=${CANDIDATES:-"600 512"}
LOG=${REPO_ROOT}/logs/clueweb_qwen_bsprobe.log

: > "$LOG"
echo "GPU_TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" | tee -a "$LOG"

for BS in $CANDIDATES; do
    rm -rf "$TMP"; mkdir -p "$TMP"
    echo "================ per_gpu_batch_size=$BS ================" | tee -a "$LOG"

    # background peak-memory sampler (max over all 4 GPUs, sampled every 1s)
    PEAK_FILE=$(mktemp)
    echo 0 > "$PEAK_FILE"
    (
        while true; do
            cur=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
            prev=$(cat "$PEAK_FILE" 2>/dev/null || echo 0)
            if [ -n "$cur" ] && [ "$cur" -gt "$prev" ] 2>/dev/null; then echo "$cur" > "$PEAK_FILE"; fi
            sleep 1
        done
    ) &
    SAMPLER=$!

    START=$(date +%s)
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node 4 "${SCRIPT_DIR}/distributed_dense_index.py" \
        --cuda_visible_devices 0,1,2,3 --n_gpu 4 \
        --model_type qwen-embedding \
        --collection_path "$COLLECTION" \
        --pretrained_doc_encoder_path "$QWEN" \
        --output_index_dir_path "$TMP" \
        --seed 42 --use_data_percent "$PROBE_PCT" --total_num_docs "$TOTAL" \
        --per_gpu_index_batch_size "$BS" --num_docs_per_block 1000000 \
        --max_doc_length 512 --do_dense_indexing >> "$LOG" 2>&1
    RC=$?
    END=$(date +%s)
    kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
    PEAK=$(cat "$PEAK_FILE"); rm -f "$PEAK_FILE"

    if [ $RC -eq 0 ]; then
        echo "RESULT bs=$BS status=OK peak_mem_MiB=$PEAK elapsed=${END}s_$((END-START))s" | tee -a "$LOG"
    elif grep -qiE "out of memory|CUDA out of memory" "$LOG"; then
        echo "RESULT bs=$BS status=OOM peak_mem_MiB=$PEAK" | tee -a "$LOG"
    else
        echo "RESULT bs=$BS status=FAIL_nonOOM rc=$RC peak_mem_MiB=$PEAK (see log)" | tee -a "$LOG"
    fi
done
rm -rf "$TMP"
echo "==== PROBE DONE ====" | tee -a "$LOG"
grep "^RESULT" "$LOG"
