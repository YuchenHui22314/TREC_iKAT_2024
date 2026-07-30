"""Rebuild a reranking candidate pool the way the official CAsT baseline builds it.

The official baseline retrieves top-N DOCUMENTS with BM25, chunks each into passages, and sends
ALL of those passages to the reranker. We retrieve passages directly, so our pool has the same
size but a completely different shape. Measured on CAsT-2022 (per query):

    official : ~841 passages from   99 documents  = 8.47 passages/document
    ours     : 1000 passages from  765 documents  = 1.35 passages/document

That difference alone changes what the reranker can do: with 8.5 passages per document it can
pick the best passage *within* a document, whereas with 1.35 it mostly compares across
documents. It is a plausible cause of the residual NDCG@3 gap that survives on BOTH 2021 and
2022 — and it cannot be explained by chunking, since the 2022 collection was rebuilt with the
official chunker and verified at 100% qrel passage-id coverage.

This script converts a passage-level run into an official-shaped pool:
  1. collapse the run to documents by max passage score,
  2. keep the top-K documents,
  3. emit EVERY passage of those documents (probed from the Lucene index), ordered by document
     rank then passage number — the official run is "in document rank order".

The result is fed back through the pipeline with `--retrieval_model none
--given_ranking_list_path <this file>` plus a reranker, so only the candidate-pool shape
changes and everything else stays identical.

Usage
-----
    cd src
    python -m apcir.indexing.expand_docs_to_passages \
        --run   ../results_cast22/.../ranking/S1[oracle]-...txt \
        --index /part/01/Tmp/yuchen/indexes/index-cast2022 \
        --out   ../results_cast22/docpool_oracle.txt --top_docs 100 --sep -
"""
import argparse
import collections
import re

from pyserini.search.lucene import LuceneSearcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_docs", type=int, default=100)
    ap.add_argument("--sep", default="-", help="passage-id separator: '-' for 2022, '_' for 2021")
    ap.add_argument("--max_passages", type=int, default=60,
                    help="stop probing a document after this many passages")
    ap.add_argument("--tag", default="docpool")
    a = ap.parse_args()

    pat = re.compile(re.escape(a.sep) + r"\d+$")
    searcher = LuceneSearcher(a.index)

    run = collections.defaultdict(dict)
    for line in open(a.run):
        f = line.split()
        if len(f) >= 5:
            run[f[0]][f[2]] = float(f[4])

    n_q = n_out = 0
    tot_docs = tot_pas = 0
    with open(a.out, "w") as fo:
        for qid, docs in run.items():
            best = {}
            for pid, s in docs.items():
                did = pat.sub("", pid)
                if did not in best or s > best[did]:
                    best[did] = s
            ranked = sorted(best.items(), key=lambda x: -x[1])[:a.top_docs]
            rank = 0
            for di, (did, score) in enumerate(ranked):
                # passage numbering starts at 1 in our 2021 collection and at 1 in the official
                # 2022 chunker output; probe upward and stop at the first gap.
                for pn in range(1, a.max_passages + 1):
                    pid = f"{did}{a.sep}{pn}"
                    if searcher.doc(pid) is None:
                        if pn == 1:
                            continue          # some ids have no _1; try a couple more
                        break
                    rank += 1
                    # score must decrease so the pool keeps document-rank order
                    fo.write(f"{qid} Q0 {pid} {rank} {1.0 / rank:.6f} {a.tag}\n")
                    n_out += 1
                tot_docs += 1
            tot_pas += rank
            n_q += 1
    print(f"{n_q} queries, {n_out:,} passage rows -> {a.out}")
    print(f"  average per query: {tot_pas / max(n_q,1):.0f} passages from "
          f"{tot_docs / max(n_q,1):.0f} documents = {tot_pas / max(tot_docs,1):.2f} passages/doc")


if __name__ == "__main__":
    main()
