"""Prebuild the IVF-PQ64 candidate index for the interactive `pq_refine` load mode.

Reads a MERGED dense index (doc_emb_block.{i}.pb float32 blocks), trains IVF-PQ64 (GPU k-means),
adds all vectors block-by-block, and writes a CPU faiss file next to the blocks
({index_dir}/ivfpq64.faiss by default) — the file the server clones to ONE GPU at activate time.

PQ64 (not PQ96/128) on purpose: 64 sub-quantizers x 8 bits = 32KB shared memory, the largest PQ
that runs on SM86 (A5000, 48KB); measured NDCG@3 after fp32 refine ties PQ128 (see
docs/dense_search_benchmark_report.md §5.1b). Add runs on ONE GPU (PQ64 codes are tiny), the
serialized index is CPU.

Usage (from src/, trec_ikat env; ~1h for ClueWeb-Qwen at /part disk speed):
  python -m apcir.indexing.build_ivfpq_index \
      --index-dir /part/01/Tmp/yuchenhui/indexes/qrecc_qwen_emb_0.6_merged \
      --dim 1024 --blocks 55
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import time

import numpy as np


def log(msg: str):
    print(f"[build_ivfpq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--out", default=None, help="default {index-dir}/ivfpq64.faiss")
    ap.add_argument("--nlist", type=int, default=None, help="default 4*sqrt(N), power-of-2-ish")
    ap.add_argument("--train-n", type=int, default=2_000_000)
    ap.add_argument("--train-gpus", type=int, default=4)
    args = ap.parse_args()
    import faiss

    out = args.out or os.path.join(args.index_dir, "ivfpq64.faiss")
    assert not os.path.exists(out), f"{out} already exists — delete it to rebuild"

    # pass 1: read only the first blocks — enough for the training sample; ESTIMATE N from their
    # mean size (merged blocks are uniform) so we don't burn a full extra disk pass just to count.
    sample, need, sizes = [], args.train_n, []
    for i in range(args.blocks):
        with open(os.path.join(args.index_dir, f"doc_emb_block.{i}.pb"), "rb") as fh:
            emb = pickle.load(fh)
        sizes.append(emb.shape[0])
        take = min(need, emb.shape[0])
        sample.append(np.ascontiguousarray(emb[:take], dtype=np.float32))
        need -= take
        del emb
        if need <= 0:
            break
    est_total = int(np.mean(sizes) * args.blocks)
    nlist = args.nlist or 1 << int(round(math.log2(4 * math.sqrt(est_total))))
    log(f"N~{est_total:,} (estimated from {len(sizes)} blocks)  nlist={nlist}")

    xt = np.concatenate(sample)
    del sample
    quant = faiss.IndexFlatIP(args.dim)
    cpu = faiss.IndexIVFPQ(quant, args.dim, nlist, 64, 8, faiss.METRIC_INNER_PRODUCT)
    cpu.clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(args.dim),
                                                       ngpu=args.train_gpus)
    t0 = time.time()
    cpu.train(xt)
    log(f"trained in {time.time() - t0:.0f}s")
    del xt

    # add on ONE GPU (fast), block order = global row order (the pq_refine contract)
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True                      # for IVFPQ the cloner's useFloat16 = fp16 LUTs
                                              # (GpuClonerOptions has NO useFloat16LookupTables)
    gpu = faiss.index_cpu_to_gpu(res, 0, cpu, co)
    t0 = time.time()
    for i in range(args.blocks):
        with open(os.path.join(args.index_dir, f"doc_emb_block.{i}.pb"), "rb") as fh:
            emb = pickle.load(fh)
        gpu.add(np.ascontiguousarray(emb, dtype=np.float32))
        del emb
        log(f"added block {i + 1}/{args.blocks} (ntotal={gpu.ntotal:,})")
    log(f"add done in {time.time() - t0:.0f}s")
    assert gpu.ntotal > 0

    final = faiss.index_gpu_to_cpu(gpu)
    faiss.write_index(final, out)
    # docid fingerprint: lets pq_refine verify the PQ row order matches the RAM store's block
    # order (ntotal alone can't catch a same-size different-order build)
    import json
    with open(os.path.join(args.index_dir, "doc_embid_block.0.pb"), "rb") as fh:
        first_ids = pickle.load(fh)
    with open(os.path.join(args.index_dir, f"doc_embid_block.{args.blocks - 1}.pb"), "rb") as fh:
        last_ids = pickle.load(fh)
    meta = dict(ntotal=int(gpu.ntotal), dim=args.dim, blocks=args.blocks, nlist=int(nlist),
                first_docid=str(first_ids[0]), last_docid=str(last_ids[-1]))
    with open(out + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    log(f"wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB) + meta")


if __name__ == "__main__":
    main()
