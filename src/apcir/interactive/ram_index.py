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
import pickle
from typing import List, Optional, Tuple

import numpy as np

from apcir.search.grouped.merge import merge_topk


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
                emb = pickle.load(h)
            with open(oj(self.index_dir, f"doc_embid_block.{block_id}.pb"), "rb") as h:
                ids = pickle.load(h)
                if isinstance(ids, list):
                    ids = np.array(ids)
            emb = np.ascontiguousarray(emb, dtype=self.store_dtype)
            if self.dim is not None and emb.shape[1] != self.dim:
                raise ValueError(
                    f"block {block_id} dim {emb.shape[1]} != expected {self.dim}")
            self._blocks.append((block_id, emb, ids))
            total_vecs += emb.shape[0]
            if verbose:
                print(f"[RamBlockSource] block {block_id}: {emb.shape} "
                      f"({emb.nbytes/1e9:.1f} GB) in {time.time()-tb:.1f}s")
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
                             merge_fn=merge_topk):
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
        return _search_ram_fp16_gpu(query_embeddings, ram_src, topN, merge_fn)

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
                         device="cuda:0", row_chunk=10_000_000):
    """fp16 dense search WITHOUT faiss (no fp16->fp32 round-trip).

    A fp16 block (41G) fits one GPU, so we skip faiss's float32-only add + multi-GPU sharding
    (the ~180s/query bottleneck for the 491G qwen index). Per block: transfer the fp16 block
    (chunked, to bound GPU RAM) to ONE GPU, compute scores = Q @ block.T with tensor cores
    (fp16 inputs, fp32 accumulation), top-k on GPU, then merge across blocks. Same global top-k
    as the faiss path; identical (D, ids[I]) per-block contract for merge_fn.
    """
    import torch
    Qt = torch.from_numpy(np.ascontiguousarray(query_embeddings, dtype=np.float16)).to(device)
    per_block = []
    with torch.no_grad():
        for block_id, emb, ids in ram_src.iter_blocks():
            n = emb.shape[0]
            parts = []
            for s in range(0, n, row_chunk):
                ct = torch.from_numpy(emb[s:s + row_chunk]).to(device)   # fp16 (chunk, dim)
                parts.append((Qt @ ct.T).float())                       # (nq, chunk) IP scores
                del ct
            scores = torch.cat(parts, dim=1)                            # (nq, n)
            k = min(topN, n)
            topv, topi = torch.topk(scores, k, dim=1)                    # sorted desc, GPU
            per_block.append((topv.cpu().numpy(), ids[topi.cpu().numpy()]))
            del scores, topv, topi, parts
        torch.cuda.empty_cache()
    merged_D, merged_I = merge_fn(per_block, topN)
    return np.asarray(merged_D), np.asarray(merged_I)
