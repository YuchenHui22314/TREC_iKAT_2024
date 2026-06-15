# ClueWeb22-B × Qwen3-Embedding-0.6B Dense Encoding — Report

Last updated: 2026-05-29. Host: **octal31** (octal31.iro.umontreal.ca).

## 0. Goal (the whole task)
Dense-encode the iKAT-2023 **ClueWeb22-B (v2)** collection with **Qwen3-Embedding-0.6B**,
storing embeddings **unmerged** (per-rank blocks, same layout as the existing
`ance_clueweb22B`), staged on local disk then copied to `/data/rech`.
That is all — QReCC was already done separately and is not touched here.

## 1. Environment
- conda env: **`trec_ikat`** at `/data/rech/huiyuche/envs/trec_ikat` (python 3.12).
  - ⚠️ `conda activate trec_ikat_2024` FAILS — the env is named `trec_ikat`.
- GPUs: **4 × RTX A5000 (24564 MiB each)**.
- ⚠️ In a non-interactive tmux shell, `conda activate` does not always put the env's
  `bin/` on PATH → `torchrun: command not found`. The script therefore calls torchrun by
  absolute path (`/data/rech/huiyuche/envs/trec_ikat/bin/torchrun`).
- `expandable_segments:True` warns "not supported on this platform" (harmless).

## 2. Key paths
| What | Path |
|---|---|
| Collection (tsv, `id\ttext`, tab-sep) | `data/collections/ikat_23/cluweb22B_ikat_v2.tsv` (145 GB) |
| Docs count | **116,838,987** (117 blocks @ 1M, last partial) |
| Qwen model snapshot | `/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418` |
| Existing ClueWeb ANCE emb (layout template) | `data/embeddings/ance_clueweb22B` (336 G, rank_0..3 × 117 blocks, fp32) |
| Qwen emb FINAL target | `data/embeddings/qwen_clueweb22B` |
| Local scratch (stage) | `/part/01/Tmp/yuchenhui/indexes/clueweb22B_qwen_emb_0.6` |
| Log | `logs/indexing_clueweb22B_qwen_emb_0.6.log` |

## 3. Model details (`models.py` `QwenEmbedding`, UNMODIFIED)
- last-token pooling + L2 normalize, dim **1024**.
- loaded bf16 + flash_attention_2; output cast to **fp32** on save (FAISS-compatible,
  matches `ance_clueweb22B`).

## 4. Scripts (under `src/apcir/indexing/dense/`)
- **`clueweb_dense_index.sh`** (NEW) — main wrapper. `bash clueweb_dense_index.sh qwen|ance`.
  - Defaults: N_GPU=4, CUDA 0,1,2,3, TOTAL_NUM_DOCS=116838987, NUM_DOCS_PER_BLOCK=1000000,
    MAX_DOC_LENGTH=512, **DO_MERGE=0** (keep unmerged), COPY_TO_FINAL=1, SKIP_EXISTING_BLOCKS=1,
    qwen PER_GPU_BATCH_SIZE default 256.
  - Override via env, e.g. `PER_GPU_BATCH_SIZE=256 bash clueweb_dense_index.sh qwen`.
- **`probe_batch_size.sh`** (NEW) — batch-size probe on ~58k docs, samples peak GPU mem.
  `CANDIDATES="256" bash probe_batch_size.sh`, log `logs/clueweb_qwen_bsprobe.log`.
- **`distributed_dense_index.py`**, **`utils.py`**, **`models.py`** — **UNMODIFIED originals.**
  `dense_indexing()` streams the collection via `distributed_index_dataset_generator`
  (1M-doc blocks) and splits each block across the 4 ranks with `DistributedSampler` —
  already memory-safe, no patch needed. (Earlier I thought I'd added a `load_collection_shard`
  helper; those edits were only ever in cancelled tool batches and never landed. Do NOT add it.)

## 5. Batch-size probe results (4×A5000, 24564 MiB)
| per_gpu_batch_size | status | peak MiB |
|---|---|---|
| 600 | OOM | 23297 |
| 512 | OOM | 23999 |
| 256 | **OK** | ~18119 |

OOM site = qwen3 MLP `down_proj`. Qwen3-0.6B is a decoder LLM (MLP intermediate 3072,
~5× ANCE), so batch must be small. **Chosen: 256.**

## 6. Disk (/part/01, shared, 2.7T)
- Freed ~380 G on 2026-05-29 by deleting QReCC **unmerged transit** dirs (finals already on
  /data/rech): `indexes/qrecc_qwen_emb_0.6`, `indexes/qrecc_ance`, `indexes/qrecc_ance_repair_20260519`.
- KEPT: `indexes/qrecc_*_merged` (final copies) + mini/probe dirs.
- After cleanup ~852 G free. Full qwen fp32 unmerged ≈ **479 G** → fits.

## 7. Run status
- **LAUNCHED 2026-05-29 23:37** in tmux session **`yuchen`**, bs=256, 4 workers, all GPUs 100%,
  ~20.7 G/GPU. Speed ≈ **3.3 s/it**, **977 it/block** → ~54 min/block → full 117 blocks ≈ **4–4.5 days**.
- Output: local `…/clueweb22B_qwen_emb_0.6` → script auto-`cp` to `data/embeddings/qwen_clueweb22B`
  on completion. Expect 4 ranks × 117 = 468 `doc_emb_block.*.pb` (+ matching `doc_embid_block.*.pb`).
- **Resume if interrupted:** rerun `PER_GPU_BATCH_SIZE=256 bash clueweb_dense_index.sh qwen`
  (`--skip_existing_blocks` skips finished blocks).
- Monitor: `tmux attach -t yuchen` or `tail -f logs/indexing_clueweb22B_qwen_emb_0.6.log`.

## 8. Done-check
`ls data/embeddings/qwen_clueweb22B/doc_emb_block.*.pb | wc -l` → should be 468; log ends with "DONE".

## 9. Gotchas
- Call torchrun by absolute path in tmux (see §1).
- Send Bash calls **one at a time** — a non-zero exit (e.g. `pkill` no-match) in a parallel
  batch cancels the whole batch.
- Don't run two probes at once — they share the TMP dir / GPUs and corrupt each other's results.
