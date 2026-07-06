"""Micro-benchmark: the production dense-search paths vs professional vector-search configs.

Measures, on a REAL index unit (default: first 26 blocks of qrecc_ance_merged, 26M x 768-d):
  RAM (RSS) / VRAM, index build+load time, per-request latency (p50/p95) at batch sizes,
  and recall@10/@100 vs the exact fp32 ground truth — for:

  a  stream_fp16_torch    production interactive path (fp16 blocks re-streamed H2D per request)
  b  stream_faiss_fp32    production offline/faiss path (per-request add->search->reset); rep#1=GT
  c  resident_faiss_fp16  sharded GpuIndexFlat useFloat16, built ONCE, search-only per request
  d  resident_torch_fp16  fp16 shards resident on GPUs, GEMM+topk per request
  e  gpu_ivf_sqfp16       IVF-Flat-fp16 (ScalarQuantizer QT_fp16), nprobe sweep     [ANN]
  f  gpu_ivf_sq8 / ivf_pq96  IVF-SQ8 + IVF-PQ (memory headline), nprobe sweep       [ANN]
  g  cpu_hnsw32           faiss IndexHNSWFlat on a subset (CPU-ANN family)          [ANN, --with-hnsw]

Baselines (a)/(b) call the REAL production functions (ram_index.search_query_against_ram,
dense_search.build_faiss_index) — no reimplementation. Results append to results.jsonl
(one row per config x param x batch), rendered by report_table.py.

Run (from src/, trec_ikat env; smoke first!):
  python -m apcir.benchmarks.bench_dense --smoke
  nohup python -m apcir.benchmarks.bench_dense --unit qrecc_ance --blocks 26 \
      > ../logs/bench_dense_$(date +%m%d).log 2>&1 &
  python -m apcir.benchmarks.bench_dense --unit qrecc_qwen --blocks 10 --configs a,c,d
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from os.path import join as oj
from typing import Dict, List, Optional, Tuple

import numpy as np

from apcir.interactive.ram_index import (RamBlockSource, _reclaim_libc_heap, _rss_gb,
                                         search_query_against_ram)
from apcir.search.grouped.merge import merge_topk

# --------------------------------------------------------------------------- #
# Units (paths verified on octal31; /part local SSD)
# --------------------------------------------------------------------------- #
IDX = "/part/01/Tmp/yuchenhui/indexes"
HF = "/data/rech/huiyuche/huggingface"
UNITS = {
    "mini": dict(index_dir=f"{IDX}/qrecc_ance_mini_merged", blocks=1, dim=768, model="ance",
                 encoder=f"{HF}/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234"),
    "qrecc_ance": dict(index_dir=f"{IDX}/qrecc_ance_merged", blocks=26, dim=768, model="ance",
                       encoder=f"{HF}/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234"),
    "qrecc_qwen": dict(index_dir=f"{IDX}/qrecc_qwen_emb_0.6_merged", blocks=10, dim=1024, model="qwen3",
                       encoder=f"{HF}/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"),
}
QUERY_FILE = "/data/rech/huiyuche/TREC_iKAT_2024/data/topics/qrecc/qrecc_valid.jsonl"
QREL_FILE = "/data/rech/huiyuche/TREC_iKAT_2024/data/qrels/qrecc_qrel.trec"
GPUS = [0, 1, 2, 3]
K = 100


def load_qrels() -> Dict[str, Dict[str, int]]:
    """TREC qrels: qid Q0 docid rel — qids are qrecc sample_ids ('conv-turn')."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(QREL_FILE) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[docid] = rel
    return qrels


def log(msg: str):
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vram_used_gb() -> List[float]:
    """Per-GPU used GB via NVML-equivalent (total-free), covers faiss allocations too."""
    import torch
    out = []
    for g in GPUS:
        try:
            free, total = torch.cuda.mem_get_info(g)
            out.append(round((total - free) / 1e9, 2))
        except Exception:
            out.append(-1.0)
    return out


def record(out_path: str, row: dict):
    row = dict(row)
    row["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    log(f"RECORD {row.get('config')}/{row.get('param', '-')}/b{row.get('batch', '-')}: "
        f"p50={row.get('p50_s')} recall@10={row.get('recall10')}")


def done_keys(out_path: str) -> set:
    keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r.get("unit"), r.get("config"), r.get("param"), r.get("batch")))
                except Exception:
                    pass
    return keys


# --------------------------------------------------------------------------- #
# Stage 1 — queries (encode once, cache)
# --------------------------------------------------------------------------- #
def load_queries(unit: dict, out_dir: str, n: int = 128) -> Tuple[np.ndarray, List[str]]:
    """Encode n JUDGED queries (sample_id has >=1 qrel) so NDCG is computable. Cache: _j suffix
    (the original unjudged 128-query cache/GT/ids files are a different query set — kept apart)."""
    cache = oj(out_dir, f"queries_{unit['model']}_j.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["q"], [str(x) for x in z["qid"]]
    import types
    judged = set(load_qrels().keys())
    texts, qids = [], []
    with open(QUERY_FILE) as f:
        for line in f:
            d = json.loads(line)
            t = (d.get("Truth_rewrite") or "").strip()
            if t and d.get("sample_id") in judged:
                texts.append(t)
                qids.append(str(d["sample_id"]))
            if len(texts) >= n:
                break
    assert len(texts) == n, f"only {len(texts)} judged queries found"
    from apcir.search.dense_search import get_test_query_embedding
    args = types.SimpleNamespace(
        retrieval_model=unit["model"], retrieval_query_type="raw",
        dense_query_encoder_path=unit["encoder"], query_gpu_id=0,
        query_encoder_batch_size=32, seed=42,
        retrieval_query_list=texts, qid_list_string=qids,
    )
    q, _ = get_test_query_embedding(args)
    q = np.ascontiguousarray(q, dtype=np.float32)
    np.savez(cache, q=q, qid=np.array(qids))
    import torch
    torch.cuda.empty_cache()
    _reclaim_libc_heap()
    log(f"encoded+cached {q.shape} JUDGED queries -> {cache}")
    return q, qids


def save_ids(out_dir: str, unit_name: str, config: str, param, I: np.ndarray):
    tag = f"{config}" + (f"_{param}" if param not in (None, "exact") else "")
    tag = tag.replace("=", "").replace("(", "_").replace(")", "")
    np.savez(oj(out_dir, f"run_{unit_name}_{tag}_j.npz"), ids=I[:, :K].astype(str))


# --------------------------------------------------------------------------- #
# Block sources
# --------------------------------------------------------------------------- #
class PreloadedBlockSource:
    """Duck-types RamBlockSource for search_query_against_ram: fp32 blocks already in RAM."""

    def __init__(self, blocks: List[Tuple[int, np.ndarray, np.ndarray]]):
        self._blocks = blocks
        self.store_dtype = np.dtype("float32")
        self.num_blocks = len(blocks)
        self.total_vecs = sum(b[1].shape[0] for b in blocks)

    def iter_blocks(self):
        yield from self._blocks


def collapse_to_single(src32: "PreloadedBlockSource") -> "PreloadedBlockSource":
    """Merge all blocks into ONE (N, dim) array. Sharded GPU faiss (IndexShards) only accepts a
    SINGLE add() pass, so the resident-index configs (c/e/f) must add all vectors at once. Copies
    block-by-block into a pre-allocated array and frees each source block right after, so the peak
    is ~ one full array + one block (not 2x)."""
    N = src32.total_vecs
    dim = src32._blocks[0][1].shape[1]
    big = np.empty((N, dim), dtype=np.float32)
    ids_parts, off = [], 0
    while src32._blocks:
        _b, emb, ids = src32._blocks.pop(0)
        big[off:off + emb.shape[0]] = emb
        ids_parts.append(ids)
        off += emb.shape[0]
        del emb
        _reclaim_libc_heap()
    big_ids = np.concatenate(ids_parts)
    log(f"collapsed to single block ({N:,}, {dim}) RSS={_rss_gb():.1f}G")
    return PreloadedBlockSource([(0, big, big_ids)])


def load_fp32_blocks(unit: dict, n_blocks: int) -> PreloadedBlockSource:
    t0 = time.time()
    blocks = []
    for i in range(n_blocks):
        with open(oj(unit["index_dir"], f"doc_emb_block.{i}.pb"), "rb") as h:
            emb = pickle.load(h)
        with open(oj(unit["index_dir"], f"doc_embid_block.{i}.pb"), "rb") as h:
            ids = pickle.load(h)
        if isinstance(ids, list):
            ids = np.array(ids, dtype=object)
        blocks.append((i, np.ascontiguousarray(emb, dtype=np.float32), ids))
        _reclaim_libc_heap()
        log(f"fp32 block {i}: {emb.shape} RSS={_rss_gb():.1f}G")
    src = PreloadedBlockSource(blocks)
    log(f"fp32 load: {src.total_vecs:,} vecs in {time.time()-t0:.0f}s RSS={_rss_gb():.1f}G")
    return src


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #
def recall_vs(gt_ids: np.ndarray, ids: np.ndarray, k: int) -> float:
    """Mean per-query |top-k(ids) ∩ top-k(gt)| / k (docid sets)."""
    vals = []
    for r in range(gt_ids.shape[0]):
        g = set(gt_ids[r, :k].tolist())
        s = set(ids[r, :k].tolist())
        vals.append(len(g & s) / k)
    return round(float(np.mean(vals)), 4)


def time_config(search_fn, Q: np.ndarray, batches: List[int], warmup: int, reps: int):
    """search_fn(Qb) -> (D, I). Returns {batch: (p50, p95, qps)} + one full-Q run for recall."""
    out = {}
    for b in batches:
        # warmup
        for i in range(warmup):
            search_fn(Q[(i * b) % len(Q):(i * b) % len(Q) + b])
        ts = []
        for i in range(reps):
            s = (i * b) % max(1, len(Q) - b)
            t0 = time.time()
            search_fn(Q[s:s + b])
            ts.append(time.time() - t0)
        ts = np.array(ts)
        out[b] = (round(float(np.percentile(ts, 50)), 4), round(float(np.percentile(ts, 95)), 4),
                  round(b / float(np.percentile(ts, 50)), 2))
    D, I = search_fn(Q)                      # full pass for recall
    return out, (D, I)


def bench_rows(out_path, unit_name, config, param, mem_row, timing, rec10, rec100, skip):
    for b, (p50, p95, qps) in timing.items():
        key = (unit_name, config, param, b)
        if key in skip:
            continue
        record(out_path, dict(unit=unit_name, config=config, param=param, batch=b,
                              p50_s=p50, p95_s=p95, qps=qps, recall10=rec10, recall100=rec100,
                              **mem_row))


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def cfg_a_stream_fp16(ram16, Q, gpus):
    def fn(Qb):
        return search_query_against_ram(Qb, ram16, index=None, topN=K, gpus=gpus)
    return fn


def cfg_b_stream_faiss(src32, Q, n_gpu):
    import types
    from apcir.search.dense_search import build_faiss_index
    args = types.SimpleNamespace(faiss_n_gpu=n_gpu, use_gpu_for_faiss=True,
                                 embed_dim=src32._blocks[0][1].shape[1], tempmem=-1)
    index = build_faiss_index(args)

    def fn(Qb):
        return search_query_against_ram(Qb, src32, index=index, topN=K)
    return fn, index


def cfg_c_resident_faiss_fp16(src32, n_gpu):
    """Sharded GPU flat index with fp16 storage, built ONCE; per request only .search()."""
    import faiss
    dim = src32._blocks[0][1].shape[1]
    res = [faiss.StandardGpuResources() for _ in range(n_gpu)]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    co.useFloat16 = True
    vres, vdev = faiss.GpuResourcesVector(), faiss.Int32Vector()
    for i in range(n_gpu):
        vdev.push_back(i)
        vres.push_back(res[i])
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, faiss.IndexFlatIP(dim), co)
    t0 = time.time()
    ids_all = []
    for _bid, emb, ids in src32.iter_blocks():
        index.add(emb)
        ids_all.append(ids)
    ids_cat = np.concatenate(ids_all)
    build_s = round(time.time() - t0, 1)

    def fn(Qb):
        D, I = index.search(np.ascontiguousarray(Qb, dtype=np.float32), K)
        return D, ids_cat[I]
    return fn, index, build_s, res


def cfg_d_resident_torch_fp16(src16_blocks, gpus):
    """fp16 shards uploaded ONCE to the GPUs; per request GEMM + running topk + id map."""
    import torch
    devs = [f"cuda:{g}" for g in gpus]
    shards = []                              # (dev, tensor, global_row_offset)
    ids_all = []
    off = 0
    t0 = time.time()
    for _bid, emb, ids in src16_blocks:
        n = emb.shape[0]
        edges = np.linspace(0, n, len(devs) + 1, dtype=np.int64)
        for gi, d in enumerate(devs):
            s0, s1 = int(edges[gi]), int(edges[gi + 1])
            if s1 > s0:
                shards.append((d, torch.from_numpy(np.ascontiguousarray(emb[s0:s1])).to(d), off + s0))
        ids_all.append(ids)
        off += n
    ids_cat = np.concatenate(ids_all)
    build_s = round(time.time() - t0, 1)

    def fn(Qb, chunk=2_000_000):
        # FIXED semantics (matches production after the fp32-score fix): docs resident fp16,
        # chunks upcast on-GPU, fp32 GEMM vs fp32 query -> fp32 scores.
        with torch.no_grad():
            Qf = np.ascontiguousarray(Qb, dtype=np.float32)
            Qt = {d: torch.from_numpy(Qf).to(d) for d in devs}
            parts = []
            for d, t, goff in shards:        # issue all GPUs' work before any sync
                bv = bi = None
                for s in range(0, t.shape[0], chunk):
                    sc = Qt[d] @ t[s:s + chunk].float().T
                    v, i = torch.topk(sc, min(K, sc.shape[1]), dim=1)
                    i = i + s
                    if bv is None:
                        bv, bi = v, i
                    else:
                        cat_v = torch.cat([bv, v], dim=1)
                        cat_i = torch.cat([bi, i], dim=1)
                        bv, sel = torch.topk(cat_v, min(K, cat_v.shape[1]), dim=1)
                        bi = torch.gather(cat_i, 1, sel)
                parts.append((bv, bi + goff))
            per = [(v.cpu().numpy(), ids_cat[i.cpu().numpy()]) for v, i in parts]
        return merge_topk(per, K)
    return fn, shards, build_s


def run_legacy_fp16out(ram16, Q, gpus):
    """The PRE-FIX scoring (fp16-output GEMM, fp16 query) — bench-only replica for the 'before'
    NDCG row. Accuracy only; latency for this path is already in results.jsonl (old run)."""
    import torch
    devs = [f"cuda:{g}" for g in gpus]
    Qf = np.ascontiguousarray(Q, dtype=np.float16)
    Qt = {d: torch.from_numpy(Qf).to(d) for d in devs}
    sources = []
    with torch.no_grad():
        for _bid, emb, ids in ram16.iter_blocks():
            n = emb.shape[0]
            edges = np.linspace(0, n, len(devs) + 1, dtype=np.int64)
            for gi, d in enumerate(devs):
                s0, s1 = int(edges[gi]), int(edges[gi + 1])
                bv = bi = None
                for s in range(s0, s1, 2_000_000):
                    e = min(s + 2_000_000, s1)
                    ct = torch.from_numpy(emb[s:e]).to(d)
                    sc = (Qt[d] @ ct.T).float()          # fp16-OUT GEMM then upcast = OLD behavior
                    v, i = torch.topk(sc, min(K, sc.shape[1]), dim=1)
                    i = i + s
                    if bv is None:
                        bv, bi = v, i
                    else:
                        cat_v = torch.cat([bv, v], dim=1)
                        cat_i = torch.cat([bi, i], dim=1)
                        bv, sel = torch.topk(cat_v, min(K, cat_v.shape[1]), dim=1)
                        bi = torch.gather(cat_i, 1, sel)
                    del ct, sc
                if bv is not None:
                    sources.append((bv.cpu().numpy(), ids[bi.cpu().numpy()]))
    return merge_topk(sources, K)


def build_ivf(src32, factory_kind: str, nlist: int, n_gpu: int, train_n: int = 2_000_000):
    """Train (GPU k-means) + build an IVF index, sharded across GPUs. factory_kind in
    {sqfp16, sq8, pq96, pq64}. Returns (search_fn(nprobe), index, build_s, mem_note, where)."""
    import faiss
    dim = src32._blocks[0][1].shape[1]
    # training sample from the first blocks
    need, sample = train_n, []
    for _bid, emb, _ids in src32.iter_blocks():
        take = min(need, emb.shape[0])
        sample.append(emb[:take])
        need -= take
        if need <= 0:
            break
    xt = np.ascontiguousarray(np.concatenate(sample), dtype=np.float32)
    quant = faiss.IndexFlatIP(dim)
    if factory_kind == "sqfp16":
        cpu = faiss.IndexIVFScalarQuantizer(quant, dim, nlist, faiss.ScalarQuantizer.QT_fp16,
                                            faiss.METRIC_INNER_PRODUCT)
    elif factory_kind == "sq8":
        cpu = faiss.IndexIVFScalarQuantizer(quant, dim, nlist, faiss.ScalarQuantizer.QT_8bit,
                                            faiss.METRIC_INNER_PRODUCT)
    elif factory_kind == "sq4":
        cpu = faiss.IndexIVFScalarQuantizer(quant, dim, nlist, faiss.ScalarQuantizer.QT_4bit,
                                            faiss.METRIC_INNER_PRODUCT)
    elif factory_kind == "pq96":
        cpu = faiss.IndexIVFPQ(quant, dim, nlist, 96, 8, faiss.METRIC_INNER_PRODUCT)
    elif factory_kind == "pq128":
        cpu = faiss.IndexIVFPQ(quant, dim, nlist, 128, 8, faiss.METRIC_INNER_PRODUCT)
    elif factory_kind == "pq64":
        cpu = faiss.IndexIVFPQ(quant, dim, nlist, 64, 8, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(factory_kind)
    # k-means on GPU (canonical clustering_index pattern)
    cpu.clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(dim), ngpu=n_gpu)
    t0 = time.time()
    cpu.train(xt)
    train_s = round(time.time() - t0, 1)
    del xt, sample
    _reclaim_libc_heap()
    # clone to GPUs sharded, then add on GPU; fall back to CPU add+search if the clone rejects
    where = "gpu"
    try:
        res = [faiss.StandardGpuResources() for _ in range(n_gpu)]
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        co.useFloat16CoarseQuantizer = False
        vres, vdev = faiss.GpuResourcesVector(), faiss.Int32Vector()
        for i in range(n_gpu):
            vdev.push_back(i)
            vres.push_back(res[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, cpu, co)
    except Exception as e:  # noqa: BLE001
        log(f"IVF {factory_kind}: GPU clone failed ({e}) -> CPU (ANN only; exact stays GPU)")
        index, where, res = cpu, "cpu", None
    t0 = time.time()
    ids_all = []
    for _bid, emb, ids in src32.iter_blocks():
        index.add(emb)
        ids_all.append(ids)
    ids_cat = np.concatenate(ids_all)
    add_s = round(time.time() - t0, 1)

    def make_fn(nprobe: int):
        try:
            faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        except Exception:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)

        def fn(Qb):
            D, I = index.search(np.ascontiguousarray(Qb, dtype=np.float32), K)
            I = np.where(I < 0, 0, I)        # pad misses (rare at high nprobe)
            return D, ids_cat[I]
        return fn
    return make_fn, index, dict(train_s=train_s, add_s=add_s, where=where), res


def make_refine_fn(ivf_index, src32_single, cand_k: int = 512):
    """Two-stage search: IVF candidates (top-cand_k ROW indices) -> exact fp32 rescore of those
    rows from the collapsed CPU block -> top-K. Candidate search runs on a CPU clone of the IVF:
    this faiss 1.8 build's GPU interleaved-scan kernel ABORTS (uncatchable C++ assert, SM86) for
    SQ8/SQ4 at k>~100, and refine needs k=512. CPU nprobe=128 candidate scan is seconds-class for
    a query batch; the refine row measures feasibility+quality, labeled where=cpu-cand.
    Requires src32_single to be a SINGLE collapsed block (row index == global index)."""
    import faiss
    assert len(src32_single._blocks) == 1, "refine requires the collapsed single block"
    big = src32_single._blocks[0][1]
    ids_cat = src32_single._blocks[0][2]
    try:
        cpu_index = faiss.index_gpu_to_cpu(ivf_index)
    except Exception:
        cpu_index = ivf_index                       # already CPU (the build fell back)
    faiss.ParameterSpace().set_index_parameter(cpu_index, "nprobe", 128)
    faiss.omp_set_num_threads(os.cpu_count() or 16)

    def fn(Qb):
        Qb = np.ascontiguousarray(Qb, dtype=np.float32)
        _D, I = cpu_index.search(Qb, cand_k)
        I = np.where(I < 0, 0, I)
        out_D = np.empty((Qb.shape[0], K), dtype=np.float32)
        out_I = np.empty((Qb.shape[0], K), dtype=object)
        for r in range(Qb.shape[0]):
            cand = big[I[r]]                          # (cand_k, dim) fp32 gather from RAM
            sc = cand @ Qb[r]
            top = np.argpartition(-sc, K - 1)[:K]
            top = top[np.argsort(-sc[top])]
            out_D[r] = sc[top]
            out_I[r] = ids_cat[I[r][top]]
        return out_D, out_I
    return fn


def cfg_g_hnsw(src32, max_docs: int):
    import faiss
    dim = src32._blocks[0][1].shape[1]
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 80
    faiss.omp_set_num_threads(os.cpu_count() or 16)
    ids_all, used = [], 0
    t0 = time.time()
    for _bid, emb, ids in src32.iter_blocks():
        take = min(emb.shape[0], max_docs - used)
        if take <= 0:
            break
        index.add(np.ascontiguousarray(emb[:take], dtype=np.float32))
        ids_all.append(ids[:take])
        used += take
    ids_cat = np.concatenate(ids_all)
    build_s = round(time.time() - t0, 1)

    def make_fn(ef: int):
        index.hnsw.efSearch = ef

        def fn(Qb):
            D, I = index.search(np.ascontiguousarray(Qb, dtype=np.float32), K)
            I = np.where(I < 0, 0, I)
            return D, ids_cat[I]
        return fn
    return make_fn, index, build_s, used


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default="qrecc_ance", choices=list(UNITS))
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--configs", default="a,b,c,d,e,f")
    ap.add_argument("--with-hnsw", action="store_true")
    ap.add_argument("--hnsw-docs", type=int, default=5_000_000)
    ap.add_argument("--smoke", action="store_true", help="whole matrix on the mini unit")
    ap.add_argument("--out-dir", default="/part/01/Tmp/yuchenhui/bench_dense")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--n-gpu", type=int, default=4)
    ap.add_argument("--f-kinds", default="sq8,pq96", help="comma IVF kinds for config f")
    ap.add_argument("--refine", action="store_true", help="two-stage refine per f-kind")
    ap.add_argument("--ndcg", action="store_true", help="score all saved runs against qrels")
    args = ap.parse_args()

    if args.smoke:
        args.unit, args.blocks, args.configs = "mini", 1, "a,b,c,d,e,f"
        args.with_hnsw = True
        args.hnsw_docs = 1000
    unit = dict(UNITS[args.unit])
    n_blocks = args.blocks or unit["blocks"]
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = oj(args.out_dir, "results.jsonl")
    skip = done_keys(out_path) if args.skip_done else set()
    configs = set(args.configs.split(","))
    nlist = 256 if args.smoke else 16384
    warm, reps_stream, reps_fast = (1, 2, 5) if args.smoke else (3, 8, 40)
    batches = [1, 16]

    # ---- Stage 0: preflight -------------------------------------------------
    import torch
    assert torch.cuda.device_count() >= args.n_gpu, "not enough GPUs visible"
    for g in range(args.n_gpu):
        free, total = torch.cuda.mem_get_info(g)
        assert total - free < 2e9, f"GPU{g} busy ({(total-free)/1e9:.1f}G used) — refusing"
    for i in range(n_blocks):
        p = oj(unit["index_dir"], f"doc_emb_block.{i}.pb")
        assert os.path.exists(p), f"missing {p}"
    import faiss
    record(out_path, dict(unit=args.unit, config="_env", param=None, batch=None,
                          faiss=faiss.__version__, torch=torch.__version__,
                          blocks=n_blocks, host=os.uname().nodename))
    rss0 = _rss_gb()

    # ---- Stage 1: queries (judged -> NDCG computable) -----------------------
    Q, qids = load_queries(unit, args.out_dir)
    gpus = GPUS[: args.n_gpu]

    gt_path = oj(args.out_dir, f"gt_{args.unit}_{n_blocks}b_j.npz")
    gt_ids = None
    if os.path.exists(gt_path):
        gt_ids = np.load(gt_path, allow_pickle=True)["ids"]

    # ---- Stage 2-3: fp16 load + configs (a), (d) ----------------------------
    need_fp16 = configs & {"a", "d"}
    d_state = None
    if need_fp16:
        t0 = time.time()
        ram16 = RamBlockSource(unit["index_dir"], n_blocks, dim=unit["dim"],
                               verbose=False, store_dtype="float16")
        load16_s = round(time.time() - t0, 1)
        rss16 = round(_rss_gb() - rss0, 1)
        id_bytes = sum(b[2].nbytes + sum(len(str(x)) + 49 for x in b[2][:1000]) / 1000 * len(b[2])
                       for b in ram16._blocks)
        record(out_path, dict(unit=args.unit, config="_fp16_load", param=None, batch=None,
                              load_s=load16_s, rss_gb=rss16, docid_gb=round(id_bytes / 1e9, 2),
                              vecs=ram16.total_vecs))
        if "a" in configs:
            fn = cfg_a_stream_fp16(ram16, Q, gpus)
            timing, (D, I) = time_config(fn, Q, batches, warm, reps_stream)
            a_ids = I
            mem = dict(rss_extra_gb=rss16, vram_gb=max(vram_used_gb()), build_s=load16_s)
            # recall filled after GT exists (streaming exact — expect ~1.0); store ids for later
            save_ids(args.out_dir, args.unit, "a_stream_fp16_torch", None, I)
            bench_rows(out_path, args.unit, "a_stream_fp16_torch", "exact", mem, timing,
                       None, None, skip)
            # legacy PRE-FIX scoring (fp16-output GEMM), NDCG "before" row — accuracy only
            _D, I = run_legacy_fp16out(ram16, Q, gpus)
            save_ids(args.out_dir, args.unit, "legacy_fp16out", None, I)
            log("legacy fp16-out run saved (accuracy-only row)")
        if "d" in configs:
            fn, shards, build_s = cfg_d_resident_torch_fp16(ram16._blocks, gpus)
            del ram16                        # CPU fp16 no longer needed once shards are on GPU
            _reclaim_libc_heap()
            timing, (D, I) = time_config(fn, Q, batches, warm, reps_fast)
            save_ids(args.out_dir, args.unit, "d_resident_torch_fp16", None, I)
            bench_rows(out_path, args.unit, "d_resident_torch_fp16", "exact",
                       dict(rss_extra_gb=round(_rss_gb() - rss0, 1), vram_gb=max(vram_used_gb()),
                            build_s=build_s), timing, None, None, skip)
            del fn, shards
            import torch as _t
            for g in gpus:
                with _t.cuda.device(g):
                    _t.cuda.empty_cache()
        _reclaim_libc_heap()

    # ---- Stage 4-5: fp32 load + GT + (b), (c), (e), (f), (g) ----------------
    src32 = None
    if configs & {"b", "c", "e", "f"} or args.with_hnsw or gt_ids is None:
        src32 = load_fp32_blocks(unit, n_blocks)
        rss32 = round(_rss_gb() - rss0, 1)

    if "b" in configs or gt_ids is None:
        fn, index = cfg_b_stream_faiss(src32, Q, args.n_gpu)
        timing, (D, I) = time_config(fn, Q, batches, warm, reps_stream)
        if gt_ids is None:
            gt_ids = I[:, :K].astype(str)
            np.savez(gt_path, ids=gt_ids)
            save_ids(args.out_dir, args.unit, "b_exact_fp32", None, I)
            log(f"ground truth saved -> {gt_path}")
        if "b" in configs:
            bench_rows(out_path, args.unit, "b_stream_faiss_fp32", "exact",
                       dict(rss_extra_gb=rss32, vram_gb=max(vram_used_gb()), build_s=0), timing,
                       recall_vs(gt_ids, I.astype(str), 10), recall_vs(gt_ids, I.astype(str), 100),
                       skip)
        index.reset()
        del fn, index

    # backfill recall for (a)/(d)/legacy now that GT exists
    for c in ("a_stream_fp16_torch", "d_resident_torch_fp16", "legacy_fp16out"):
        p = oj(args.out_dir, f"run_{args.unit}_{c}_j.npz")
        if os.path.exists(p) and gt_ids is not None:
            ids = np.load(p, allow_pickle=True)["ids"]
            record(out_path, dict(unit=args.unit, config=f"{c}_recall_backfill", param="exact",
                                  batch=None, recall10=recall_vs(gt_ids, ids, 10),
                                  recall100=recall_vs(gt_ids, ids, 100)))

    # resident GPU-faiss configs need a single add() pass -> one merged block
    if src32 is not None and (configs & {"c", "e", "f"} or args.with_hnsw):
        src32 = collapse_to_single(src32)

    if "c" in configs:
        v0 = vram_used_gb()
        fn, index, build_s, _res = cfg_c_resident_faiss_fp16(src32, args.n_gpu)
        timing, (D, I) = time_config(fn, Q, batches, warm, reps_fast)
        save_ids(args.out_dir, args.unit, "c_resident_faiss_fp16", None, I)
        bench_rows(out_path, args.unit, "c_resident_faiss_fp16", "exact",
                   dict(rss_extra_gb=round(_rss_gb() - rss0, 1),
                        vram_gb=round(max(vram_used_gb()) - min(v0), 2), build_s=build_s), timing,
                   recall_vs(gt_ids, I.astype(str), 10), recall_vs(gt_ids, I.astype(str), 100),
                   skip)
        index.reset()
        del fn, index

    ivf_matrix = []
    if "e" in configs:
        ivf_matrix.append(("e_gpu_ivf_sqfp16", "sqfp16"))
    if "f" in configs:
        ivf_matrix += [(f"f_ivf_{k}", k) for k in args.f_kinds.split(",") if k]
    for cname, kind in ivf_matrix:
        try:
            v0 = vram_used_gb()
            make_fn, index, meta, _res = build_ivf(src32, kind, nlist, args.n_gpu)
        except Exception as e:  # noqa: BLE001
            if kind == "pq96":               # SM86 shared-mem fallback
                log(f"pq96 failed ({e}) -> retry pq64")
                try:
                    make_fn, index, meta, _res = build_ivf(src32, "pq64", nlist, args.n_gpu)
                    cname = "f_ivf_pq64"
                except Exception as e2:  # noqa: BLE001
                    log(f"pq64 also failed ({e2}) — skipping")
                    continue
            else:
                log(f"{cname} failed ({e}) — skipping")
                continue
        for nprobe in (8, 32, 128):
            fn = make_fn(nprobe)
            timing, (D, I) = time_config(fn, Q, batches, warm, reps_fast)
            save_ids(args.out_dir, args.unit, cname, f"nprobe{nprobe}", I)
            bench_rows(out_path, args.unit, cname, f"nprobe={nprobe}",
                       dict(rss_extra_gb=round(_rss_gb() - rss0, 1),
                            vram_gb=round(max(vram_used_gb()) - min(v0), 2),
                            build_s=meta["train_s"] + meta["add_s"], where=meta["where"]),
                       timing, recall_vs(gt_ids, I.astype(str), 10),
                       recall_vs(gt_ids, I.astype(str), 100), skip)
        if args.refine:
            # two-stage: IVF candidates (nprobe=128, top-1000 rows) -> fp32 rescore from the
            # collapsed CPU block -> top-K. VRAM = the IVF index only; the fix for corpora whose
            # exact fp16 index cannot fit VRAM (ClueWeb-Qwen).
            fn = make_refine_fn(index, src32)
            timing, (D, I) = time_config(fn, Q, batches, warm, reps_fast)
            save_ids(args.out_dir, args.unit, cname, "refine512", I)
            bench_rows(out_path, args.unit, f"{cname}+refine", "cand512,nprobe=128",
                       dict(rss_extra_gb=round(_rss_gb() - rss0, 1),
                            vram_gb=round(max(vram_used_gb()) - min(v0), 2),
                            build_s=meta["train_s"] + meta["add_s"], where=meta["where"]),
                       timing, recall_vs(gt_ids, I.astype(str), 10),
                       recall_vs(gt_ids, I.astype(str), 100), skip)
        try:
            index.reset()
        except Exception:
            pass
        del make_fn, index
        _reclaim_libc_heap()

    if args.with_hnsw:
        make_fn, index, build_s, used = cfg_g_hnsw(src32, args.hnsw_docs)
        # subset GT: exact fp32 over the same subset
        sub = PreloadedBlockSource(
            [(i, b[1][: max(0, min(b[1].shape[0], args.hnsw_docs - sum(x[1].shape[0] for x in src32._blocks[:i])))],
              b[2][: max(0, min(b[1].shape[0], args.hnsw_docs - sum(x[1].shape[0] for x in src32._blocks[:i])))])
             for i, b in enumerate(src32._blocks)])
        fnb, idxb = cfg_b_stream_faiss(sub, Q, args.n_gpu)
        _D, gI = fnb(Q)
        idxb.reset()
        sub_gt = gI[:, :K].astype(str)
        for ef in (16, 64, 256):
            fn = make_fn(ef)
            timing, (D, I) = time_config(fn, Q, batches, warm, reps_fast)
            bench_rows(out_path, args.unit, "g_cpu_hnsw32", f"efSearch={ef}(n={used})",
                       dict(rss_extra_gb=round(_rss_gb() - rss0, 1), vram_gb=0, build_s=build_s,
                            where="cpu"), timing, recall_vs(sub_gt, I.astype(str), 10),
                       recall_vs(sub_gt, I.astype(str), 100), skip)
        del make_fn, index

    # ---- Stage 7: NDCG vs qrels over every saved run ------------------------
    if args.ndcg:
        import glob
        qrels = load_qrels()
        rows = []
        for p in sorted(glob.glob(oj(args.out_dir, f"run_{args.unit}_*_j.npz"))):
            tag = os.path.basename(p)[len(f"run_{args.unit}_"):-len("_j.npz")]
            ids = np.load(p, allow_pickle=True)["ids"]
            m = qrel_metrics(ids, qids, qrels)
            m.update(unit=args.unit, config="_ndcg", param=tag, batch=None)
            record(out_path, m)
            rows.append((tag, m))
        log("── NDCG table ──")
        for tag, m in sorted(rows, key=lambda r: -r[1]["ndcg3"]):
            log(f"  {tag:38s} ndcg@3={m['ndcg3']:.4f} ndcg@10={m['ndcg10']:.4f} "
                f"mrr@10={m['mrr10']:.4f} r@100={m['r100']:.4f} judged={m['n_q']}")

    log(f"DONE. results -> {out_path}")


def qrel_metrics(ids: np.ndarray, qids: List[str], qrels: Dict[str, Dict[str, int]]) -> dict:
    """NDCG@3/@10, MRR@10, Recall@100 for a (n_q, K) docid matrix (rank order = column order).
    qrecc rels are binary; ideal DCG uses the total number of judged positives."""
    n3 = n10 = mrr = r100 = 0.0
    n_q = 0
    for r, qid in enumerate(qids):
        rel = {d for d, v in qrels.get(qid, {}).items() if v > 0}
        if not rel:
            continue
        n_q += 1
        hits = [1.0 if str(d) in rel else 0.0 for d in ids[r][:100]]
        def dcg(g):
            return sum(h / np.log2(i + 2) for i, h in enumerate(g))
        for k, acc in ((3, "n3"), (10, "n10")):
            ideal = dcg([1.0] * min(len(rel), k))
            val = dcg(hits[:k]) / ideal if ideal > 0 else 0.0
            if k == 3:
                n3 += val
            else:
                n10 += val
        rr = 0.0
        for i, h in enumerate(hits[:10]):
            if h > 0:
                rr = 1.0 / (i + 1)
                break
        mrr += rr
        r100 += sum(hits) / len(rel)
    assert n_q > 0, "no judged queries — qid mismatch?"
    return dict(ndcg3=round(n3 / n_q, 4), ndcg10=round(n10 / n_q, 4),
                mrr10=round(mrr / n_q, 4), r100=round(r100 / n_q, 4), n_q=n_q)


if __name__ == "__main__":
    main()
