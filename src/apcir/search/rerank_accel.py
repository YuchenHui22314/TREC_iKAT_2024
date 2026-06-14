"""Multi-GPU reranking via HuggingFace accelerate — the RIGHT no-sync data parallelism
for inference (one model per GPU process, shard the queries, gather). Works for BOTH
the qwen3 reranker (CausalLM) and monot5 (T5), unlike DataParallel (re-replicates per
forward) or DDP (gradient-sync; training-only).

Run:
    accelerate launch --num_processes 4 --multi_gpu \
        -m apcir.search.rerank_accel \
        --ranking_in <first_stage.txt> --ranking_out <reranked.txt> \
        --reranker qwen3_reranker --reranking_query_type qwen_3_rerank_instruct_full \
        --topics ikat_23_test --input_query_path ../data/topics/ikat23/ikat_2023_test.json \
        --retrieval_model BM25 --sparse_index_dir_path <lucene_dir> \
        [--qrel_file_path <qrel> to also print metrics]

CORRECTNESS: queries are sharded WHOLE (each query's top-k docs are reranked as one
batch on one GPU), so the per-query scores are identical to single-GPU — the output
ranking is byte-identical to a 1-process run (verify with --num_processes 1 vs 4).
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
from accelerate import Accelerator
from accelerate.utils import gather_object
from pyserini.search.lucene import LuceneSearcher

from apcir.evaluate.evaluation_util import get_query_list, evaluate
from apcir.search.search import load_ranking_list_from_file
from apcir.search.rerank import (
    QwenReranker, _fetch_contents_cached,
    QWEN3_RERANK_CONV_INSTRUCTION, QWEN3_RERANK_DEFAULT_INSTRUCTION,
)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ranking_in", required=True)
    p.add_argument("--ranking_out", required=True)
    p.add_argument("--reranker", default="qwen3_reranker")
    p.add_argument("--qwen3_reranker_path", default="Qwen/Qwen3-Reranker-4B")
    p.add_argument("--cache_dir", default="/data/rech/huiyuche/huggingface")
    p.add_argument("--rerank_quant", default="none")
    p.add_argument("--rerank_top_k", type=int, default=50)
    p.add_argument("--rerank_batch_size", type=int, default=8)
    p.add_argument("--sparse_index_dir_path", required=True)
    # query building (get_query_list)
    p.add_argument("--topics", required=True)
    p.add_argument("--input_query_path", required=True)
    p.add_argument("--retrieval_model", default="BM25")
    p.add_argument("--retrieval_query_type", default="oracle")
    p.add_argument("--reranking_query_type", default="oracle")
    p.add_argument("--qrel_file_path", default=None)
    p.add_argument("--run_name", default="rerank_accel")
    return p


def make_query_args(a):
    return SimpleNamespace(
        topics=a.topics, input_query_path=a.input_query_path,
        retrieval_model=a.retrieval_model, retrieval_query_type=a.retrieval_query_type,
        reranking_query_type=a.reranking_query_type, generation_query_type="none",
        fb_terms=20, original_query_weight=0.5, fusion_type="none",
        QRs_to_rank=[], level_type="none", personalization_group="all",
        fuse_weights=[], fusion_query_lists=[], qid_personalized_level_dict={},
    )


def main():
    a = build_parser().parse_args()
    acc = Accelerator()
    device = str(acc.device)

    # ---- build queries + load first-stage ranking (every process; cheap) ----
    rql, reranking_query_list, gql, fql, qids, plvl, wts, turns = get_query_list(make_query_args(a))
    qid2query = {q: rq for q, rq in zip(qids, reranking_query_list)}
    hits = load_ranking_list_from_file(a.ranking_in)
    work = [q for q in qids if q in hits]   # stable order; one item per query

    # ---- load the reranker ONCE on THIS process's GPU ----
    if a.reranker == "qwen3_reranker":
        reranker = QwenReranker(model_path=a.qwen3_reranker_path, cache_dir=a.cache_dir,
                                quant=a.rerank_quant, device=device)
        instruction = (QWEN3_RERANK_CONV_INSTRUCTION
                       if a.reranking_query_type == "qwen_3_rerank_instruct_full"
                       else QWEN3_RERANK_DEFAULT_INSTRUCTION)
    else:
        raise NotImplementedError(
            f"reranker {a.reranker} not wired in rerank_accel yet (qwen3_reranker works; "
            "monot5 would load monoT5 single-GPU on `device` here)")

    searcher = LuceneSearcher(a.sparse_index_dir_path)
    doc_cache = {}

    # ---- shard the queries across processes; each reranks its slice ----
    local = {}
    with acc.split_between_processes(work) as shard:
        for qid in shard:
            top = hits[qid][:a.rerank_top_k]
            docs = [_fetch_contents_cached(searcher, d.docid, doc_cache) for d in top]
            scores = reranker.score(instruction, qid2query[qid], docs, a.rerank_batch_size)
            order = np.argsort(np.array(scores, dtype=np.float32))[::-1]
            # reranked head (1/rank score) + untouched tail, same convention as rerank.py
            reranked = [top[i].docid for i in order] + [d.docid for d in hits[qid][a.rerank_top_k:]]
            local[qid] = reranked

    # ---- gather all processes' results to rank 0 ----
    gathered = gather_object([local])
    if not acc.is_main_process:
        return
    merged = {}
    for d in gathered:
        merged.update(d)

    # ---- write reranked TREC ranking ----
    with open(a.ranking_out, "w") as f:
        for qid in qids:
            if qid not in merged:
                continue
            for rank, docid in enumerate(merged[qid]):
                f.write(f"{qid} Q0 {docid} {rank+1} {1.0/(rank+1)} {a.run_name}\n")
    print(f"[rerank_accel] wrote {len(merged)} reranked queries -> {a.ranking_out}")

    if a.qrel_file_path:
        run = {qid: {docid: 1.0/(r+1) for r, docid in enumerate(merged[qid])} for qid in merged}
        mlist = "recip_rank,ndcg_cut.3,recall.10,recall.100".split(",")
        keyf = [m.replace(".", "_") for m in mlist]
        _, avg = evaluate(run, a.qrel_file_path, a.ranking_out, mlist, keyf)
        print("[rerank_accel] metrics:", {k: round(avg[k]*100, 1) for k in keyf})


if __name__ == "__main__":
    main()
