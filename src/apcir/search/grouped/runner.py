"""GroupRunner: the core "stream once, fan out" loop for a group of ExperimentSpec sharing
one CorpusKey (same doc index, same dim — possibly different query encoders/checkpoints).

PHASE A  encode every spec's queries (GPU, cheap) -> stack into one Q, record row-slices.
PHASE B  ONE corpus stream: build_faiss_index once; per block add -> search(Q,topN) -> map
         ids -> accumulate; reset ONCE per block. (merge via merge_compat for byte-identity.)
PHASE C  per spec: slice D/I -> get_dense_ranking_list -> get_run_object_and_save_ranking_list
         (writes ranking file) -> evaluate -> save metrics JSON. Reproduces evaluation.py:455-564.

Validate G2/G3/G4 (byte-identical to legacy) before trusting for the paper. Uses merge_compat
(exact). Switch to merge.merge_topk for speed once gates pass.
"""

import os
import numpy as np

from apcir.evaluate.evaluation_util import get_query_list, evaluate
from apcir.search.dense_search import (
    build_faiss_index, get_test_query_embedding, get_dense_ranking_list,
)
from apcir.search.search import get_run_object_and_save_ranking_list
from apcir.search.grouped.merge import merge_compat
from apcir.search.grouped.block_source import PickleBlockSource


def _encode_spec(spec):
    """Phase A for one spec: get_query_list (routes full_conversation->_dense + populates
    retrieval_query_list/qid_list_string on args), then encode. Returns (emb (N,dim), qids)."""
    a = spec.args
    (rql, rrl, gql, fql, qids, plvl, wts, turns) = get_query_list(a)
    a.retrieval_query_list = rql
    a.reranking_query_list = rrl
    a.fusion_query_lists = fql
    a.qid_list_string = qids
    a.qid_personalized_level_dict = plvl
    a.qid_weights_dict = wts
    spec.turn_list = turns
    emb, ids = get_test_query_embedding(a)     # (N, dim) float32, ids == qids order
    return np.asarray(emb, dtype=np.float32), list(ids)


def _output_paths(spec, base_output_dir):
    """The ikat-results-layout paths for this spec (same as evaluation.py)."""
    stem = spec.file_name_stem
    coll, topics = spec.args.collection, spec.args.topics
    d = os.path.join(base_output_dir, coll, topics)
    return {
        "ranking": os.path.join(d, "ranking", stem + ".txt"),
        "metrics": os.path.join(d, "metrics", stem + ".json"),
        "per_query": os.path.join(d, "per_query_metrics", stem + "_dict.json"),
        "dir": d,
    }


def run_group(group, merge_fn=merge_compat):
    """group: list[ExperimentSpec] sharing one CorpusKey. Streams the corpus ONCE."""
    key = group[0].corpus_key
    src = PickleBlockSource(key.index_dir, key.block_num, key.embed_dim)
    src.validate()

    # ---- PHASE A: encode all specs, stack ----
    embs, slices = [], []
    cursor = 0
    per_spec_qids = []
    for spec in group:
        emb, qids = _encode_spec(spec)
        n = emb.shape[0]
        slices.append((cursor, cursor + n))
        per_spec_qids.append(qids)
        embs.append(emb)
        cursor += n
    Q = np.concatenate(embs, axis=0)            # (sum_N, dim)
    topN = max(int(s.args.retrieval_top_k) for s in group)
    print(f"[grouped] {len(group)} jobs, Q={Q.shape}, topN={topN}, blocks={key.block_num}, idx={key.index_dir}")

    # ---- PHASE B: ONE stream over blocks ----
    index = build_faiss_index(group[0].args)    # OPT-2: built once for the whole group
    per_block = []
    for block_id, emb, ids in src.iter_blocks():
        assert index.ntotal == 0, "index not empty before add (reset bug)"
        index.add(emb)
        D, I = index.search(Q, topN)            # ALL stacked queries in one GEMM
        P = ids[I]                              # map faiss idx -> docid
        per_block.append((D, P))
        index.reset()
        assert index.ntotal == 0, "index not reset after block (reset bug)"
        del emb, ids
    merged_D, merged_I = merge_fn(per_block, topN)   # compat: lists (q, up to 2*topN)
    merged_D = np.array(merged_D)
    merged_I = np.array(merged_I)

    # ---- PHASE C: slice + emit per spec ----
    for spec, (lo, hi), qids in zip(group, slices, per_spec_qids):
        a = spec.args
        paths = _output_paths(spec, a.output_dir_path)
        for sub in ("ranking", "metrics", "per_query_metrics", "ikat_format_output"):
            os.makedirs(os.path.join(paths["dir"], sub), exist_ok=True)
        a.ranking_list_path = paths["ranking"]
        a.file_name_stem = spec.file_name_stem
        # restore the per-spec query fields search()/save expect
        a.retrieval_query_list = a.retrieval_query_list  # already set in phase A
        hits = get_dense_ranking_list(qids, merged_D[lo:hi], merged_I[lo:hi], int(a.retrieval_top_k))
        # returns (hits, run); search() unpacks it the same way (search.py:517, evaluation.py:488)
        _hits2, run = get_run_object_and_save_ranking_list(hits, a)   # builds TREC run + writes ranking file
        if a.run_eval:
            metrics_list = a.metrics.split(",")
            key_form = [m.replace(".", "_") for m in metrics_list]
            query_metrics_dic, averaged_metrics = evaluate(
                run, a.qrel_file_path, paths["ranking"], metrics_list, key_form)
            _save_metrics(paths, spec, averaged_metrics, query_metrics_dic)
        print(f"[grouped]  emitted {spec.args.retrieval_model} x {spec.args.retrieval_query_type} x {spec.args.topics}")


def _save_metrics(paths, spec, averaged_metrics, query_metrics_dic):
    """Replicate evaluation.py's metrics-JSON dump (the "Print this line..." header + metrics dict
    + args dict). TODO: verify byte-format against evaluation.py:560-600 in G4."""
    import json
    a = spec.args
    args_view = {
        "collection": a.collection, "topics": a.topics,
        "retrieval_model": a.retrieval_model, "reranker": a.reranker,
        "generation_model": a.generation_model,
        "retrieval_query_type": a.retrieval_query_type,
        "reranking_query_type": a.reranking_query_type,
        "generation_query_type": a.generation_query_type,
    }
    with open(paths["metrics"], "w") as f:
        f.write("Print this line to your latex table\n")
        f.write(json.dumps(averaged_metrics) + "\n")
        f.write(json.dumps(args_view) + "\n")
    with open(paths["per_query"], "w") as f:
        json.dump(query_metrics_dic, f)
