"""
Cross-block result merging for the shared-corpus dense-retrieval framework.

When we stream a corpus block-by-block and search ALL stacked queries against each
block, we get per-block (scores, docids) and must merge them into a global ranking
per query. Two implementations:

- `merge_compat`  : a BIT-IDENTICAL reproduction of the legacy
  `dense_search.search_one_by_one_with_faiss` two-pointer merge (lines 162-212),
  MINUS the `copy.deepcopy` (provably a no-op: the previous merged lists are only
  READ and their tuples are immutable). Used to prove the new path reproduces the
  paper's exact numbers (validation gate 1).

- `merge_topk`    : a fast numpy running-top-K merge. ~100x less Python overhead
  (no list-of-tuples, no per-element Python loop, no deepcopy). Returns the proper
  global top-K. For the used metric range (top `retrieval_top_k`), this is
  equivalent to `merge_compat` — see NOTE below — so it is the production default.

NOTE on the legacy "2*topN, drop the lower half each block" behaviour: the legacy
merge keeps `2*topN` candidates and, from block 3 on, only re-reads the FIRST
`topN` of the accumulated list (`while p1 < topN`), discarding accumulated ranks
[topN:2topN] each block. This does NOT affect the final top-`topN`: every candidate
is from some block's top-`topN`, the running top-`topN` already holds the global
best `topN`, and discarded ranks are all below it. `get_dense_ranking_list` then
takes only the top `retrieval_top_k (== topN)`. So `merge_topk` (proper top-K) and
`merge_compat` agree on the used range, modulo equal-score tie order (the two-pointer
breaks ties toward the earlier block via `>=`; numpy breaks ties by stable sort).
For the paper, validate with `merge_compat`; switch to `merge_topk` for speed once
gate 1 passes.
"""

import numpy as np


def merge_compat(per_block, topN):
    """Bit-identical reproduction of the legacy two-pointer merge (deepcopy removed).

    per_block: iterable of (D_block, P_block) where
        D_block : (q, topN) float scores for this block (descending per row, as FAISS returns)
        P_block : (q, topN) docids for this block (already mapped via passage_embedding2id[I])
    Returns (merged_D, merged_P), each a python list-of-lists, width up to 2*topN.
    """
    merged = None
    for D_block, P_block in per_block:
        # legacy builds candidate_matrix as a list per query of (score, pid) tuples
        candidate_matrix = [
            [(s, p) for s, p in zip(D_block[q], P_block[q])]
            for q in range(len(D_block))
        ]
        if merged is None:
            merged = candidate_matrix
            continue
        # legacy: merged_candidate_matrix_tmp = copy.deepcopy(merged); merged = []
        # deepcopy is a no-op here -> tmp is only read, tuples are immutable.
        tmp = merged
        merged = []
        for merged_list, cur_list in zip(tmp, candidate_matrix):
            p1, p2 = 0, 0
            out = []
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:   # ties -> earlier (merged) wins
                    out.append(merged_list[p1]); p1 += 1
                else:
                    out.append(cur_list[p2]); p2 += 1
            while p1 < topN:
                out.append(merged_list[p1]); p1 += 1
            while p2 < topN:
                out.append(cur_list[p2]); p2 += 1
            merged.append(out)
    merged_D = [[c[0] for c in ml] for ml in merged]
    merged_P = [[c[1] for c in ml] for ml in merged]
    return merged_D, merged_P


def merge_topk(per_block, top_k):
    """Fast numpy running-top-K merge. Returns (merged_D, merged_P) numpy arrays (q, top_k),
    each row sorted by score descending. Proper global top-k (no lossy 2*topN buffer).

    per_block: iterable of (D_block, P_block) numpy arrays, shapes (q, n_i).
    """
    acc_D = None   # (q, K) running best scores
    acc_P = None   # (q, K) running best docids
    for D_block, P_block in per_block:
        D_block = np.asarray(D_block)
        P_block = np.asarray(P_block)
        if acc_D is None:
            cat_D, cat_P = D_block, P_block
        else:
            cat_D = np.concatenate([acc_D, D_block], axis=1)
            cat_P = np.concatenate([acc_P, P_block], axis=1)
        k = min(top_k, cat_D.shape[1])
        # top-k per row by score (unordered), then sort those k descending
        part = np.argpartition(-cat_D, k - 1, axis=1)[:, :k]
        rows = np.arange(cat_D.shape[0])[:, None]
        sel_D = cat_D[rows, part]
        sel_P = cat_P[rows, part]
        order = np.argsort(-sel_D, axis=1, kind="stable")
        acc_D = sel_D[rows, order]
        acc_P = sel_P[rows, order]
    return acc_D, acc_P
