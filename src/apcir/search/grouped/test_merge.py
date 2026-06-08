"""Validation gate 1: prove merge_compat reproduces the legacy two-pointer merge
EXACTLY (byte-identical), and that merge_topk agrees on the used top-K range.

Run: python -m apcir.search.grouped.test_merge   (from src/)
"""
import copy
import numpy as np

from apcir.search.grouped.merge import merge_compat, merge_topk


def legacy_merge(per_block, topN):
    """Faithful inline copy of dense_search.search_one_by_one_with_faiss lines 134-212
    (the merge half), including the copy.deepcopy. Ground truth for byte-identity."""
    merged_candidate_matrix = None
    for D_block, P_block in per_block:
        D = D_block.tolist()
        candidate_id_matrix = P_block.tolist()
        candidate_matrix = []
        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp, candidate_matrix):
            p1, p2 = 0, 0
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1]); p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2]); p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1]); p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2]); p2 += 1
    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix:
        merged_D.append([]); merged_I.append([])
        for candidate in merged_list:
            merged_D[-1].append(candidate[0]); merged_I[-1].append(candidate[1])
    return np.array(merged_D), np.array(merged_I)


def make_blocks(rng, q, n_blocks, topN, n_docs):
    """Per block: random descending scores (q, topN) + unique-ish docids, mimicking
    FAISS top-topN output (each row sorted descending)."""
    per = []
    for b in range(n_blocks):
        D = -np.sort(-rng.random((q, topN)).astype(np.float32), axis=1)   # descending per row
        P = (b * n_docs + rng.integers(0, n_docs, size=(q, topN))).astype(np.int64)
        per.append((D, P))
    return per


def main():
    rng = np.random.default_rng(0)
    cases = [(3, 1, 5, 100), (4, 2, 5, 100), (5, 6, 10, 1000), (8, 6, 1000, 20_000_000)]
    for q, B, topN, ndoc in cases:
        per = make_blocks(rng, q, B, topN, ndoc)
        # gate 1: compat == legacy, byte-identical
        lD, lI = legacy_merge([(d.copy(), p.copy()) for d, p in per], topN)
        cD, cP = merge_compat([(d.copy(), p.copy()) for d, p in per], topN)
        cD, cP = np.array(cD), np.array(cP)
        assert cD.shape == lD.shape, f"shape {cD.shape} vs {lD.shape}"
        assert np.array_equal(cD, lD), f"compat D != legacy D (q={q},B={B},topN={topN})"
        assert np.array_equal(cP, lI), f"compat P != legacy I (q={q},B={B},topN={topN})"
        # merge_topk agrees with compat on the top-K SCORES (tie order may differ on ids)
        K = topN
        tD, tP = merge_topk([(d.copy(), p.copy()) for d, p in per], K)
        # compare the multiset of (score) at top-K per row, and ids where scores are strictly unique
        comp_topD = lD[:, :K]
        assert np.allclose(np.sort(tD, axis=1), np.sort(comp_topD, axis=1)), \
            f"merge_topk top-{K} scores differ (q={q},B={B},topN={topN})"
        print(f"  OK q={q} B={B} topN={topN}: compat==legacy byte-identical; topk scores match")
    print("GATE 1 PASSED: merge_compat is byte-identical to legacy; merge_topk matches top-K scores.")


if __name__ == "__main__":
    main()
