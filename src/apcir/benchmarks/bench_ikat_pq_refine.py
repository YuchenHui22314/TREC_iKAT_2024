"""Validate pq_refine (IVF-PQ64 candidates + INT8-store fp32 rescore) on the HARD setting:
iKAT'23 oracle rewrites over the full ClueWeb22-B Qwen index (116M docs, graded qrels).

One 450G disk pass produces BOTH the exact-fp32 ground-truth run AND the int8 rescore store:
  per block: load fp32 -> sharded GPU IndexFlatIP add/search/reset (exact GT) -> quantize to
  int8 (+ per-row scales, kept) -> free fp32.  RAM peak ~ int8_total(119G) + one fp32 block(80G).
Then: prebuilt ivfpq64.faiss -> ONE GPU (fp16 LUTs) -> PQ-only run + PQ+refine-int8 run.
Metrics: pytrec_eval GRADED ndcg_cut_3 / ndcg_cut_10 / recall_100 vs data/qrels/ikat_23_qrel.txt.

Run (from src/, trec_ikat env, AFTER the PQ prebuild finished; ~1h, disk-bound):
  nohup python -m apcir.benchmarks.bench_ikat_pq_refine > ../logs/bench_ikat_pq.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import pickle
import time
from os.path import join as oj

import numpy as np

from apcir.interactive.ram_index import _reclaim_libc_heap, _rss_gb, quantize_int8, dequantize_int8
from apcir.search.grouped.merge import merge_topk

IDX = "/part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged"
PQ = oj(IDX, "ivfpq64.faiss")
TOPICS = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/2023_ikat_test_topics_flattened.json"
QREL = "/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/ikat_23_qrel.txt"
ENC = ("/data/rech/huiyuche/huggingface/models--Qwen--Qwen3-Embedding-0.6B/"
       "snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418")
OUT_DIR = "/part/01/Tmp/yuchenhui/bench_dense"
BLOCKS, DIM, K = 6, 1024, 1000


def log(m):
    print(f"[ikat_pq {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_queries():
    cache = oj(OUT_DIR, "queries_ikat23_oracle_qwen.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["q"], [str(x) for x in z["qid"]]
    topics = json.load(open(TOPICS))
    qids = [t["sample_id"] for t in topics]
    texts = [t["oracle_utt_text"] for t in topics]
    import types
    from apcir.search.dense_search import get_test_query_embedding
    args = types.SimpleNamespace(retrieval_model="qwen3", retrieval_query_type="raw",
                                 dense_query_encoder_path=ENC, query_gpu_id=0,
                                 query_encoder_batch_size=32, seed=42,
                                 retrieval_query_list=texts, qid_list_string=qids)
    q, _ = get_test_query_embedding(args)
    q = np.ascontiguousarray(q, dtype=np.float32)
    np.savez(cache, q=q, qid=np.array(qids))
    import torch
    torch.cuda.empty_cache()
    _reclaim_libc_heap()
    log(f"encoded {q.shape} ikat23 oracle queries")
    return q, qids


def metrics(run_D, run_I, qids, tag):
    import pytrec_eval
    qrels = {}
    with open(QREL) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                qrels.setdefault(p[0], {})[p[2]] = int(p[3])
    run = {}
    for r, qid in enumerate(qids):
        if qid not in qrels:
            continue
        run[qid] = {str(d): float(K - i) for i, d in enumerate(run_I[r]) if d is not None}
    ev = pytrec_eval.RelevanceEvaluator(
        {q: qrels[q] for q in run}, {"ndcg_cut.3", "ndcg_cut.10", "recall.100", "recip_rank"})
    res = ev.evaluate(run)
    agg = {m: float(np.mean([v[m.replace(".", "_")] for v in res.values()]))
           for m in ("ndcg_cut.3", "ndcg_cut.10", "recall.100", "recip_rank")}
    log(f"{tag:28s} judged={len(res)} " +
        " ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    with open(oj(OUT_DIR, "results.jsonl"), "a") as f:
        f.write(json.dumps(dict(unit="clueweb_qwen_ikat23", config="_ikat_ndcg", param=tag,
                                batch=None, judged=len(res),
                                **{k.replace(".", ""): round(v, 4) for k, v in agg.items()},
                                ts=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    return agg


def main():
    import faiss
    assert os.path.exists(PQ), f"{PQ} missing — wait for build_ivfpq_index to finish"
    Q, qids = load_queries()

    # ---- pass 1: exact GT + int8 store in ONE disk sweep --------------------
    gt_cache = oj(OUT_DIR, "gt_ikat23_clueweb_qwen.npz")
    int8_blocks, scales, ids_all, offs = [], [], [], [0]
    need_gt = not os.path.exists(gt_cache)
    if need_gt:
        res = [faiss.StandardGpuResources() for _ in range(4)]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        vres, vdev = faiss.GpuResourcesVector(), faiss.Int32Vector()
        for i in range(4):
            vdev.push_back(i)
            vres.push_back(res[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, faiss.IndexFlatIP(DIM), co)
    per_block = []
    for b in range(BLOCKS):
        t0 = time.time()
        with open(oj(IDX, f"doc_emb_block.{b}.pb"), "rb") as fh:
            emb = np.ascontiguousarray(pickle.load(fh), dtype=np.float32)
        with open(oj(IDX, f"doc_embid_block.{b}.pb"), "rb") as fh:
            ids = np.array(pickle.load(fh), dtype=object)
        if need_gt:
            index.add(emb)
            D, I = index.search(Q, K)
            per_block.append((D, ids[I]))
            index.reset()
        qb, sc = quantize_int8(emb)
        int8_blocks.append(qb)
        scales.append(sc)
        ids_all.append(ids)
        offs.append(offs[-1] + emb.shape[0])
        del emb
        _reclaim_libc_heap()
        log(f"block {b + 1}/{BLOCKS}: GT+int8 in {time.time() - t0:.0f}s RSS={_rss_gb():.0f}G")
    ids_cat = np.concatenate(ids_all)
    offs = np.array(offs)
    if need_gt:
        gt_D, gt_I = merge_topk(per_block, K)
        np.savez(gt_cache, D=np.asarray(gt_D), ids=np.asarray(gt_I).astype(str))
        index.reset()
        del per_block
    z = np.load(gt_cache, allow_pickle=True)
    metrics(z["D"], z["ids"], qids, "exact_fp32")

    # ---- PQ index on one GPU -------------------------------------------------
    cpu = faiss.read_index(PQ)
    co1 = faiss.GpuClonerOptions()
    co1.useFloat16 = True
    gres = faiss.StandardGpuResources()
    gpu = faiss.index_cpu_to_gpu(gres, 0, cpu, co1)
    del cpu
    _reclaim_libc_heap()
    log(f"PQ index on cuda:0 ntotal={gpu.ntotal:,}")
    assert gpu.ntotal == offs[-1], "PQ/store row mismatch"

    for nprobe in (64, 128):
        gpu.nprobe = nprobe
        _D, I = gpu.search(Q, K)
        I = np.where(I < 0, 0, I)
        metrics(_D, ids_cat[I], qids, f"pq64_only_np{nprobe}")

        # refine: exact fp32 rescore of the top-K candidate rows from the INT8 store
        out_D = np.empty((len(Q), K), dtype=np.float32)
        out_I = np.empty((len(Q), K), dtype=object)
        t0 = time.time()
        for r in range(len(Q)):
            rows = I[r]
            bidx = np.searchsorted(offs, rows, side="right") - 1
            cand = np.empty((len(rows), DIM), dtype=np.float32)
            for bb in np.unique(bidx):
                m = bidx == bb
                loc = rows[m] - offs[bb]
                cand[m] = dequantize_int8(int8_blocks[bb][loc], scales[bb][loc])
            sc = cand @ Q[r]
            top = np.argsort(-sc)
            out_D[r] = sc[top]
            out_I[r] = ids_cat[rows[top]]
        log(f"refine np{nprobe}: {(time.time() - t0) / len(Q) * 1000:.1f}ms/query rescore")
        metrics(out_D, out_I, qids, f"pq64_refine_int8_np{nprobe}")

    log("DONE")


if __name__ == "__main__":
    main()
