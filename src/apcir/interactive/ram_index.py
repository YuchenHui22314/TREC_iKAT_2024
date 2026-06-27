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


class RamBlockSource:
    """Loads all `num_blocks` (emb, ids) blocks into RAM at construction and keeps
    them resident. `iter_blocks()` yields from RAM (no disk touch after __init__).

    Layout matches PickleBlockSource: doc_emb_block.{i}.pb (np.ndarray (n_i, dim)
    float32) + doc_embid_block.{i}.pb (list/ndarray of docids), i in [0, num_blocks).
    """

    def __init__(self, index_dir: str, num_blocks: int, dim: Optional[int] = None,
                 verbose: bool = True, store_dtype: str = "float32"):
        self.index_dir = index_dir
        self.num_blocks = num_blocks
        self.dim = dim
        # store_dtype="float16" halves RAM (e.g. the 491 GB qwen3 ClueWeb22-B index -> ~245 GB,
        # fitting a 503 GB node). Blocks are cast back to float32 per-block at search time for the
        # faiss add (faiss IndexFlatIP requires float32 input); fp16 storage for inner-product
        # retrieval is standard (cf. faiss GpuIndexFlatConfig.useFloat16) with negligible quality loss.
        self.store_dtype = np.dtype(store_dtype)
        self._blocks: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self._load(verbose)

    def validate(self):
        """Pre-flight (per CLAUDE.md): all 2*num_blocks files exist and it's a MERGED
        index (consecutive doc_emb_block.{0..n-1}.pb, not raw per-rank rank_* blocks)."""
        for i in range(self.num_blocks):
            for pref in ("doc_emb_block", "doc_embid_block"):
                p = oj(self.index_dir, f"{pref}.{i}.pb")
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"RamBlockSource missing {p} (merged index? right block_num?)")

    def _load(self, verbose: bool):
        self.validate()
        t0 = time.time()
        total_vecs = 0
        for block_id in range(self.num_blocks):
            tb = time.time()
            with open(oj(self.index_dir, f"doc_emb_block.{block_id}.pb"), "rb") as h:
                emb32 = pickle.load(h)
            with open(oj(self.index_dir, f"doc_embid_block.{block_id}.pb"), "rb") as h:
                ids = pickle.load(h)
                if isinstance(ids, list):
                    # dtype=object keeps docids as pointers (~8MB/block, still supports ids[I]
                    # fancy-indexing). The DEFAULT builds a fixed-width UCS4 array padded to the
                    # LONGEST docid: qrecc URL docids reach 6335 chars -> ~25G/block, which (kept
                    # across 55 blocks) OOM'd the loader. The fp16 embeddings were never the issue.
                    ids = np.array(ids, dtype=object)
            emb = np.ascontiguousarray(emb32, dtype=self.store_dtype)
            if emb is not emb32:
                del emb32                    # drop the fp32 source NOW (always a copy for fp16 store)
            if self.dim is not None and emb.shape[1] != self.dim:
                raise ValueError(
                    f"block {block_id} dim {emb.shape[1]} != expected {self.dim}")
            self._blocks.append((block_id, emb, ids))
            total_vecs += emb.shape[0]
            # return the freed fp32 buffer to the OS so RSS stays ~ (fp16 resident + one block),
            # not (fp16 resident + ALL fp32 temporaries) — the glibc-arena retention that caused
            # the ~235G RSS / OOM when loading qrecc_ance (169G fp32) as fp16.
            _reclaim_libc_heap()
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
                         gpus=None, row_chunk=4_000_000):
    """fp16 dense search WITHOUT faiss, sharded across `gpus` (default [0]).

    Each block's rows are split evenly across the GPUs; on each GPU the shard is streamed in
    row_chunks (bounding GPU memory) while keeping a running per-shard top-k (scores = Q @ shard.T
    with tensor cores, fp16 in / fp32 accumulate, then top-k). The per-(block, shard) top-k's are
    merged by `merge_fn` into the global top-N. Sharding lets a 41G fp16 block fit even 24G cards
    and lets the GPUs work in parallel (each GPU's shard work is issued before any `.cpu()` sync);
    gpus=[0] is the legacy whole-block-on-one-card path. Same global top-N and the same
    (D, ids[I]) contract as the faiss path.
    """
    import torch
    gpus = list(gpus) if gpus else [0]
    devs = [(g, f"cuda:{g}") for g in gpus]
    Qf = np.ascontiguousarray(query_embeddings, dtype=np.float16)
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
                    ct = torch.from_numpy(emb[s:e]).to(d)        # fp16 (m, dim)
                    sc = (Qt[d] @ ct.T).float()                  # (nq, m) inner product
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
