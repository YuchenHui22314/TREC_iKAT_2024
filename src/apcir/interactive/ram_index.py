"""RAM-resident dense index for the interactive search server.

The batch path (`dense_search.search_one_by_one_with_faiss`) re-loads each block
(`doc_emb_block.{i}.pb` + `doc_embid_block.{i}.pb`) from disk on EVERY query batch.
For a long-running server answering one turn at a time that would re-stream the whole
336 GB index per turn. Instead we load ALL blocks into RAM ONCE at startup
(`RamBlockSource`), build the GPU faiss index ONCE, and per query do the same
add -> search -> reset PHASE-B loop as `grouped/runner.py:78-92`, but reading blocks
from RAM. Only one ~28 GB block is resident on the GPUs at a time (add then reset).

Byte-faithful to the batch path: same block load (`PickleBlockSource`-style), same
`IndexFlatIP` add/search/reset, same id mapping `ids[I]`, then `merge_topk` (proper
global top-k; equivalent to the legacy two-pointer merge over the used top_k range —
see grouped/merge.py).
"""

from __future__ import annotations

import time
from os.path import join as oj
import os
import gc
import pickle
from typing import List, Optional, Tuple

import numpy as np

from apcir.search.grouped.merge import merge_topk


_LIBC = None


def _rss_gb() -> float:
    """Resident set size of this process in GB (via /proc/self/statm; no deps)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return -1.0


def _reclaim_libc_heap():
    """Return freed heap memory to the OS. NumPy/pickle free their large fp32 buffers via libc,
    but glibc may retain freed multi-GB chunks in its arenas, so RSS does NOT drop on `del` alone
    (this caused a ~235G RSS / OOM when loading a 169G-fp32 index as fp16). gc.collect() handles
    Python reachability; malloc_trim(0) asks glibc to return free top-of-heap memory to the OS.
    Best-effort: no-op if libc / malloc_trim is unavailable."""
    gc.collect()
    global _LIBC
    try:
        if _LIBC is None:
            import ctypes
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
    except Exception:
        pass


def quantize_int8(emb32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization for the pq_refine RESCORE store: halves the fp16
    footprint again (ClueWeb-Qwen 235G -> ~119G, which FITS octal31). Rescoring from int8 rows
    ranks essentially identically to fp32 for normalized embeddings (measured: IVF-SQ8's NDCG@3
    equals the refine ceiling). Returns (int8 codes, float32 (n,1) per-row scales)."""
    scales = np.abs(emb32).max(axis=1, keepdims=True).astype(np.float32) / 127.0
    scales[scales == 0] = 1.0
    q = np.clip(np.rint(emb32 / scales), -127, 127).astype(np.int8)
    return q, scales


def dequantize_int8(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scales


class RamBlockSource:
    """Loads all `num_blocks` (emb, ids) blocks into RAM at construction and keeps
    them resident. `iter_blocks()` yields from RAM (no disk touch after __init__).

    Layout matches PickleBlockSource: doc_emb_block.{i}.pb (np.ndarray (n_i, dim)
    float32) + doc_embid_block.{i}.pb (list/ndarray of docids), i in [0, num_blocks).
    """

    def __init__(self, index_dir: str, num_blocks: int, dim: Optional[int] = None,
                 verbose: bool = True, store_dtype: str = "float32", progress_cb=None):
        self.index_dir = index_dir
        self.num_blocks = num_blocks
        self.dim = dim
        # store_dtype="float16" halves RAM (e.g. the 491 GB qwen3 ClueWeb22-B index -> ~245 GB,
        # fitting a 503 GB node). Blocks are cast back to float32 per-block at search time for the
        # faiss add (faiss IndexFlatIP requires float32 input); fp16 storage for inner-product
        # retrieval is standard (cf. faiss GpuIndexFlatConfig.useFloat16) with negligible quality loss.
        self.store_dtype = np.dtype(store_dtype)
        self._blocks: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self._scales: List[np.ndarray] = []      # int8 store only: per-block (n,1) row scales
        self._load(verbose, progress_cb)

    def validate(self):
        """Pre-flight (per CLAUDE.md): all 2*num_blocks files exist and it's a MERGED
        index (consecutive doc_emb_block.{0..n-1}.pb, not raw per-rank rank_* blocks)."""
        for i in range(self.num_blocks):
            for pref in ("doc_emb_block", "doc_embid_block"):
                p = oj(self.index_dir, f"{pref}.{i}.pb")
                if not os.path.exists(p):
                    # an int8 store can load from its npy cache alone (fp32 block not needed)
                    if (pref == "doc_emb_block" and self.store_dtype == np.int8
                            and os.path.exists(oj(self.index_dir, f"doc_emb_int8_block.{i}.npy"))
                            and os.path.exists(oj(self.index_dir, f"doc_emb_int8_scale.{i}.npy"))):
                        continue
                    raise FileNotFoundError(
                        f"RamBlockSource missing {p} (merged index? right block_num?)")

    def _load(self, verbose: bool, progress_cb=None):
        self.validate()
        t0 = time.time()
        total_vecs = 0
        for block_id in range(self.num_blocks):
            tb = time.time()
            with open(oj(self.index_dir, f"doc_embid_block.{block_id}.pb"), "rb") as h:
                ids = pickle.load(h)
                if isinstance(ids, list):
                    # dtype=object keeps docids as pointers (~8MB/block, still supports ids[I]
                    # fancy-indexing). The DEFAULT builds a fixed-width UCS4 array padded to the
                    # LONGEST docid: qrecc URL docids reach 6335 chars -> ~25G/block, which (kept
                    # across 55 blocks) OOM'd the loader. The fp16 embeddings were never the issue.
                    ids = np.array(ids, dtype=object)
            if self.store_dtype == np.int8:      # pq_refine rescore store: int8 + per-row scales
                # disk cache: reading the 119G int8 store beats re-reading 450G fp32 + quantizing
                # (~13min vs ~70min for ClueWeb-Qwen). Written on the first (cold) load below.
                c_emb = oj(self.index_dir, f"doc_emb_int8_block.{block_id}.npy")
                c_sc = oj(self.index_dir, f"doc_emb_int8_scale.{block_id}.npy")
                if os.path.exists(c_emb) and os.path.exists(c_sc):
                    emb = np.load(c_emb)
                    sc = np.load(c_sc)
                else:
                    with open(oj(self.index_dir, f"doc_emb_block.{block_id}.pb"), "rb") as h:
                        emb32 = pickle.load(h)
                    emb, sc = quantize_int8(np.asarray(emb32, dtype=np.float32))
                    del emb32
                    try:
                        np.save(c_emb, emb)
                        np.save(c_sc, sc)
                    except OSError as e:         # cache is an optimization, never a load failure
                        print(f"[RamBlockSource] int8 cache write failed ({e}) — continuing",
                              flush=True)
                self._scales.append(sc)
            else:
                with open(oj(self.index_dir, f"doc_emb_block.{block_id}.pb"), "rb") as h:
                    emb32 = pickle.load(h)
                emb = np.ascontiguousarray(emb32, dtype=self.store_dtype)
                if emb is not emb32:
                    del emb32                # drop the fp32 source NOW (always a copy for fp16 store)
            if self.dim is not None and emb.shape[1] != self.dim:
                raise ValueError(
                    f"block {block_id} dim {emb.shape[1]} != expected {self.dim}")
            self._blocks.append((block_id, emb, ids))
            total_vecs += emb.shape[0]
            # return the freed fp32 buffer to the OS so RSS stays ~ (fp16 resident + one block),
            # not (fp16 resident + ALL fp32 temporaries) — the glibc-arena retention that caused
            # the ~235G RSS / OOM when loading qrecc_ance (169G fp32) as fp16.
            _reclaim_libc_heap()
            if progress_cb is not None:                 # per-block progress for the activate bar
                progress_cb(block_id + 1, self.num_blocks)
            if verbose:
                print(f"[RamBlockSource] block {block_id}: {emb.shape} "
                      f"({emb.nbytes/1e9:.1f} GB)  RSS={_rss_gb():.1f}G  in {time.time()-tb:.1f}s")
        if verbose:
            print(f"[RamBlockSource] loaded {self.num_blocks} blocks, "
                  f"{total_vecs:,} vectors in {time.time()-t0:.1f}s")
        self.total_vecs = total_vecs

    def iter_blocks(self):
        for block_id, emb, ids in self._blocks:
            yield block_id, emb, ids


def search_query_against_ram(query_embeddings: np.ndarray,
                             ram_src: RamBlockSource,
                             index,
                             topN: int,
                             merge_fn=merge_topk,
                             gpus=None):
    """PHASE-B for the RAM index: per block add -> search(Q, topN) -> map ids -> reset,
    then merge across blocks. Mirrors grouped/runner.py:78-92.

    query_embeddings: (num_query, dim) float32.
    index: a faiss IndexFlatIP (built once via build_faiss_index); reset between blocks.
    Returns (merged_D, merged_I) numpy arrays (num_query, topN) — feed to
    dense_search.get_dense_ranking_list.
    """
    # A non-streaming resident container (GpuResidentDense / PQRefineDense) answers directly.
    if hasattr(ram_src, "search_topn"):
        D, I = ram_src.search_topn(query_embeddings, topN)
        return np.asarray(D), np.asarray(I)

    # fp16-stored index (e.g. the 491G qwen3 ClueWeb22-B): faiss IndexFlatIP.add() takes ONLY
    # float32, so the faiss path would cast every 41G fp16 block back to an 82G fp32 array per
    # query AND shard-transfer it across GPUs over PCIe == ~180s/query. A fp16 block is 41G and
    # fits ONE GPU (<46G), so we search it NATIVELY in fp16 below (no cast, no sharding, half the
    # PCIe traffic) -> ~10x faster. Only the float32 path (ANCE) goes through faiss.
    if getattr(ram_src, "store_dtype", np.dtype("float32")) == np.float16:
        return _search_ram_fp16_gpu(query_embeddings, ram_src, topN, merge_fn, gpus=gpus)

    Q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    per_block = []
    try:
        for block_id, emb, ids in ram_src.iter_blocks():
            assert index.ntotal == 0, "index not empty before add (reset bug)"
            index.add(emb)                       # fp32 path (ANCE): emb is already float32
            D, I = index.search(Q, topN)
            per_block.append((D, ids[I]))
            index.reset()
            assert index.ntotal == 0, "index not reset after block (reset bug)"
    finally:
        index.reset()
    merged_D, merged_I = merge_fn(per_block, topN)
    return np.asarray(merged_D), np.asarray(merged_I)


def _search_ram_fp16_gpu(query_embeddings, ram_src, topN, merge_fn=merge_topk,
                         gpus=None, row_chunk=2_000_000):
    """fp16 dense search WITHOUT faiss, sharded across `gpus` (default [0]).

    Each block's rows are split evenly across the GPUs; on each GPU the shard is streamed in
    row_chunks (bounding GPU memory) while keeping a running per-shard top-k. The per-(block,
    shard) top-k's are merged by `merge_fn` into the global top-N. Sharding lets a 41G fp16 block
    fit even 24G cards and lets the GPUs work in parallel (each GPU's shard work is issued before
    any `.cpu()` sync); gpus=[0] is the legacy whole-block-on-one-card path. Same global top-N and
    the same (D, ids[I]) contract as the faiss path.

    Numerics: docs are STORED fp16 (RAM + the H2D copy stays half-sized) but each chunk is upcast
    on-GPU and scored with an fp32 GEMM against the fp32 query, so the SCORES are fp32. An
    fp16-output GEMM quantizes scores to ulp=0.5 at ANCE's ~700 magnitude, shuffling near-tied
    docs (measured recall@10 0.83 vs 0.98, see docs/dense_search_benchmark_report.md). row_chunk
    default is 2M (not 4M) to bound the transient fp32 chunk (2M x 1024 x 4B = 8G/GPU).
    """
    import torch
    gpus = list(gpus) if gpus else [0]
    devs = [(g, f"cuda:{g}") for g in gpus]
    Qf = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    Qt = {d: torch.from_numpy(Qf).to(d) for _, d in devs}
    sources = []
    with torch.no_grad():
        for _block_id, emb, ids in ram_src.iter_blocks():
            n = emb.shape[0]
            edges = np.linspace(0, n, len(devs) + 1, dtype=np.int64)
            staged = []
            for gi, (_g, d) in enumerate(devs):
                s0, s1 = int(edges[gi]), int(edges[gi + 1])
                if s1 <= s0:
                    continue
                bv = bi = None                                   # running per-shard top-k (on d)
                for s in range(s0, s1, row_chunk):
                    e = min(s + row_chunk, s1)
                    ct = torch.from_numpy(emb[s:e]).to(d)        # fp16 (m, dim) — half-size H2D
                    sc = Qt[d] @ ct.float().T                    # fp32 GEMM -> fp32 scores
                    cv, ci = torch.topk(sc, min(topN, sc.shape[1]), dim=1)   # (nq, k) sorted desc
                    ci = ci + s                                  # chunk-local -> block-row index
                    if bv is None:
                        bv, bi = cv, ci
                    else:
                        cat_v = torch.cat([bv, cv], dim=1)
                        cat_i = torch.cat([bi, ci], dim=1)
                        bv, sel = torch.topk(cat_v, min(topN, cat_v.shape[1]), dim=1)
                        bi = torch.gather(cat_i, 1, sel)
                    del ct, sc, cv, ci
                staged.append((bv, bi, ids))
            for bv, bi, ids_full in staged:                      # .cpu() here -> GPUs overlapped above
                sources.append((bv.cpu().numpy(), ids_full[bi.cpu().numpy()]))
                del bv, bi
    for g in gpus:
        with torch.cuda.device(g):
            torch.cuda.empty_cache()
    merged_D, merged_I = merge_fn(sources, topN)
    return np.asarray(merged_D), np.asarray(merged_I)


# --------------------------------------------------------------------------- #
# Resident containers (load modes beyond the default ram_fp16 streaming)
# --------------------------------------------------------------------------- #
class GpuResidentDense:
    """`gpu_resident` mode: fp16 doc shards uploaded ONCE to the assigned GPUs; per request only
    an fp32-score GEMM (docs upcast per chunk) + top-k + id map. ~100x lower latency than the
    streaming path (benchmarked 2.73s -> 0.03s at 26M docs) at identical retrieval quality.

    allocation: [(gpu_id, gb), ...] from CapacityManager.allocate_gpus — block rows are split
    across the GPUs proportionally to their granted VRAM."""

    def __init__(self, index_dir: str, num_blocks: int, dim: int, allocation,
                 verbose: bool = True, progress_cb=None, row_chunk: int = 2_000_000):
        import pickle
        import torch
        self.store_dtype = np.dtype("float16")   # keeps pipeline dtype checks meaningful
        self.load_mode = "gpu_resident"
        self._row_chunk = row_chunk
        gpus = [g for g, _gb in allocation]
        weights = np.array([gb for _g, gb in allocation], dtype=np.float64)
        weights = weights / weights.sum()
        self._devs = [f"cuda:{g}" for g in gpus]
        self._shards = []                        # (dev, fp16 tensor, global_row_offset)
        ids_all, off = [], 0
        for i in range(num_blocks):
            with open(os.path.join(index_dir, f"doc_emb_block.{i}.pb"), "rb") as fh:
                emb32 = pickle.load(fh)
            emb = np.ascontiguousarray(emb32, dtype=np.float16)
            del emb32
            _reclaim_libc_heap()
            n = emb.shape[0]
            edges = np.floor(np.concatenate([[0.0], np.cumsum(weights)]) * n).astype(np.int64)
            edges[-1] = n
            for gi, dev in enumerate(self._devs):
                s0, s1 = int(edges[gi]), int(edges[gi + 1])
                if s1 > s0:
                    self._shards.append(
                        (dev, torch.from_numpy(np.ascontiguousarray(emb[s0:s1])).to(dev),
                         off + s0))
            with open(os.path.join(index_dir, f"doc_embid_block.{i}.pb"), "rb") as fh:
                ids = pickle.load(fh)
            ids_all.append(np.array(ids, dtype=object))
            off += n
            del emb
            _reclaim_libc_heap()
            if progress_cb:
                progress_cb(i + 1, num_blocks)
            if verbose:
                print(f"[gpu_resident] block {i + 1}/{num_blocks} uploaded", flush=True)
        self._ids = np.concatenate(ids_all)
        self.total_vecs = off
        self.num_blocks = num_blocks

    def search_topn(self, Q: np.ndarray, topN: int):
        import torch
        with torch.no_grad():
            Qf = np.ascontiguousarray(Q, dtype=np.float32)
            Qt = {d: torch.from_numpy(Qf).to(d) for d in dict.fromkeys(self._devs)}
            parts = []
            for dev, t, goff in self._shards:    # issue all GPUs' work before any sync
                bv = bi = None
                for s in range(0, t.shape[0], self._row_chunk):
                    sc = Qt[dev] @ t[s:s + self._row_chunk].float().T   # fp32 scores
                    v, i = torch.topk(sc, min(topN, sc.shape[1]), dim=1)
                    i = i + s
                    if bv is None:
                        bv, bi = v, i
                    else:
                        cat_v = torch.cat([bv, v], dim=1)
                        cat_i = torch.cat([bi, i], dim=1)
                        bv, sel = torch.topk(cat_v, min(topN, cat_v.shape[1]), dim=1)
                        bi = torch.gather(cat_i, 1, sel)
                parts.append((bv, bi + goff))
            per = [(v.cpu().numpy(), self._ids[i.cpu().numpy()]) for v, i in parts]
        return merge_topk(per, topN)

    def close(self):
        import torch
        self._shards = []
        for d in dict.fromkeys(self._devs):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()


class PQRefineDense:
    """`pq_refine` mode: prebuilt IVF-PQ64 on ONE GPU proposes top-`cand_k` candidates; the fp16
    RAM store rescored in fp32 fixes their order (benchmarked NDCG@3 within ~2% of exact on
    normalized embeddings; PQ alone is NOT acceptable). VRAM = the small PQ index only."""

    def __init__(self, pq_path: str, ram_src: RamBlockSource, gpu_id: int,
                 nprobe: int = 64, cand_k: int = 512, verbose: bool = True):
        import faiss
        self.store_dtype = np.dtype("float16")
        self.load_mode = "pq_refine"
        self._ram = ram_src
        self._cand_k = cand_k
        cpu = faiss.read_index(pq_path)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True                  # IVFPQ: cloner useFloat16 == fp16 lookup tables
        self._res = faiss.StandardGpuResources()
        self._index = faiss.index_cpu_to_gpu(self._res, gpu_id, cpu, co)
        self._index.nprobe = nprobe
        # global row -> (block, local row) mapping via block offsets (add order == block order)
        offs, ids_all = [0], []
        for _bid, emb, ids in ram_src.iter_blocks():
            offs.append(offs[-1] + emb.shape[0])
            ids_all.append(ids)
        self._offs = np.array(offs)
        self._ids = np.concatenate(ids_all)
        self.total_vecs = int(self._offs[-1])
        self.num_blocks = ram_src.num_blocks
        assert self._index.ntotal == self._offs[-1], \
            f"PQ index ntotal {self._index.ntotal} != store vectors {self._offs[-1]} " \
            f"(index built from a different corpus/blocks?)"
        meta_path = pq_path + ".meta.json"
        if os.path.exists(meta_path):                       # docid-order fingerprint (build meta)
            import json
            with open(meta_path) as fh:
                meta = json.load(fh)
            assert (str(self._ids[0]) == meta["first_docid"]
                    and str(self._ids[-1]) == meta["last_docid"]), \
                f"PQ index {pq_path} docid fingerprint mismatch — built from a different " \
                f"block order than the RAM store"
        else:
            print(f"[pq_refine] WARNING: {meta_path} missing — row-order vs RAM store "
                  f"verified by ntotal only", flush=True)
        if verbose:
            print(f"[pq_refine] index {pq_path} on cuda:{gpu_id} ntotal={self._index.ntotal}",
                  flush=True)

    def search_topn(self, Q: np.ndarray, topN: int):
        Qf = np.ascontiguousarray(Q, dtype=np.float32)
        # candidates must cover topN (default retrieval_top_k is 1000 > the 512 default) — GPU
        # PQ64 handles k<=2048 (verified on SM86); beyond that we cap and TRIM the output.
        cand_k = min(max(self._cand_k, topN), 2048, self._index.ntotal)
        _D, I = self._index.search(Qf, cand_k)                # (nq, cand_k) global rows
        I = np.where(I < 0, 0, I)
        blocks = [b[1] for b in self._ram._blocks]
        int8_store = self._ram.store_dtype == np.int8
        k_out = min(topN, cand_k)
        out_D = np.empty((len(Qf), k_out), dtype=np.float32)
        out_I = np.empty((len(Qf), k_out), dtype=object)
        for r in range(len(Qf)):
            rows = I[r]
            bidx = np.searchsorted(self._offs, rows, side="right") - 1
            cand = np.empty((len(rows), Qf.shape[1]), dtype=np.float32)
            for b in np.unique(bidx):
                m = bidx == b
                loc = rows[m] - self._offs[b]
                if int8_store:                            # dequantize gathered rows (per-row scales)
                    cand[m] = dequantize_int8(blocks[b][loc], self._ram._scales[b][loc])
                else:
                    cand[m] = blocks[b][loc].astype(np.float32)
            sc = cand @ Qf[r]
            top = np.argpartition(-sc, k_out - 1)[:k_out]
            top = top[np.argsort(-sc[top])]
            out_D[r] = sc[top]
            out_I[r] = self._ids[rows[top]]
        return out_D, out_I

    def close(self):
        try:
            self._index.reset()
        except Exception:
            pass
        self._index = None
        self._res = None
