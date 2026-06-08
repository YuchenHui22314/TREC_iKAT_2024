"""BlockSource: the ONLY disk-touching abstraction. Streams a block-partitioned dense
index one block at a time. PickleBlockSource lifts the load logic from
dense_search.search_one_by_one_with_faiss:98-113 verbatim.
"""

import os
import pickle
from os.path import join as oj

import numpy as np


class PickleBlockSource:
    """doc_emb_block.{i}.pb (np.ndarray (n_i, dim) float32) + doc_embid_block.{i}.pb
    (list/ndarray of docids), i in [0, num_blocks)."""

    def __init__(self, index_dir, num_blocks, dim=None):
        self.index_dir = index_dir
        self.num_blocks = num_blocks
        self.dim = dim

    def validate(self):
        """Pre-flight (per CLAUDE.md): assert all 2*num_blocks files exist and it's a MERGED
        index (consecutive doc_emb_block.{0..n-1}.pb, not raw per-rank rank_* blocks)."""
        for i in range(self.num_blocks):
            for pref in ("doc_emb_block", "doc_embid_block"):
                p = oj(self.index_dir, f"{pref}.{i}.pb")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"BlockSource missing {p} (merged index? right block_num?)")

    def iter_blocks(self):
        """Yield (block_id, emb, ids) one at a time; caller must drop each before the next
        (one ~75G block resident at a time). Matches the legacy load exactly."""
        for block_id in range(self.num_blocks):
            with open(oj(self.index_dir, f"doc_emb_block.{block_id}.pb"), "rb") as h:
                emb = pickle.load(h)
            with open(oj(self.index_dir, f"doc_embid_block.{block_id}.pb"), "rb") as h:
                ids = pickle.load(h)
                if isinstance(ids, list):
                    ids = np.array(ids)
            yield block_id, emb, ids
