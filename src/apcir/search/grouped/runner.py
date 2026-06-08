"""GroupRunner: the core "stream once, fan out" loop for a group of ExperimentSpec sharing
one CorpusKey (same doc index, same dim — possibly different query encoders/checkpoints).

PHASE A  encode every spec's queries (GPU, cheap) -> stack into one Q, record row-slices.
PHASE B  ONE corpus stream: build_faiss_index once; per block add -> search(Q,topN) -> map
         ids -> accumulate; reset ONCE per block. (merge via merge_compat for byte-identity.)
PHASE C  per spec: slice D/I -> get_dense_ranking_list -> get_run_object_and_save_ranking_list
         (writes ranking file) -> evaluate -> save metrics. BYTE-FAITHFUL to evaluation.py:467-648
         (verified by code review: arg set/del bookkeeping + the exact metrics-file format).

Validate G2/G3/G4 (byte-identical to legacy) before trusting for the paper. Uses merge_compat
(exact). Switch to merge.merge_topk for speed once gates pass.
"""

import os
import json
import numpy as np

from apcir.evaluate.evaluation_util import (
    get_query_list, evaluate, print_formatted_latex_metrics,
)
from apcir.functional.topics import load_turns_from_json, save_turns_to_json
from apcir.search.dense_search import (
    build_faiss_index, get_test_query_embedding, get_dense_ranking_list,
)
from apcir.search.search import get_run_object_and_save_ranking_list
from apcir.search.grouped.merge import merge_compat
from apcir.search.grouped.block_source import PickleBlockSource


def _encode_spec(spec):
    """Phase A for one spec: get_query_list (routes full_conversation->_dense + sets the query
    fields on args, exactly like evaluation.py:455-481), then encode. Returns (emb (N,dim), qids)."""
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


def _output_paths(spec):
    a = spec.args
    d = os.path.join(a.output_dir_path, a.collection, a.topics)
    stem = spec.file_name_stem
    return {
        "dir": d,
        "ranking": os.path.join(d, "ranking", stem + ".txt"),
        "metrics": os.path.join(d, "metrics", stem + ".json"),
        "per_query": os.path.join(d, "per_query_metrics", stem + "_dict.json"),
    }


def run_group(group, merge_fn=merge_compat):
    """group: list[ExperimentSpec] sharing one CorpusKey. Streams the corpus ONCE."""
    key = group[0].corpus_key
    src = PickleBlockSource(key.index_dir, key.block_num, key.embed_dim)
    src.validate()

    # ---- PHASE A: encode all specs, stack ----
    embs, slices, per_spec_qids, cursor = [], [], [], 0
    for spec in group:
        emb, qids = _encode_spec(spec)
        n = emb.shape[0]
        slices.append((cursor, cursor + n))
        per_spec_qids.append(qids)
        embs.append(emb)
        cursor += n
    Q = np.concatenate(embs, axis=0)            # (sum_N, dim)
    topN = max(int(s.args.retrieval_top_k) for s in group)
    print(f"[grouped] {len(group)} jobs, Q={Q.shape}, topN={topN}, blocks={key.block_num}")

    # ---- PHASE B: ONE stream over blocks ----
    index = build_faiss_index(group[0].args)    # OPT-2: built once for the whole group
    per_block = []
    try:
        for block_id, emb, ids in src.iter_blocks():
            assert index.ntotal == 0, "index not empty before add (reset bug)"
            index.add(emb)
            D, I = index.search(Q, topN)        # ALL stacked queries in one GEMM
            per_block.append((D, ids[I]))       # map faiss idx -> docid within this block
            index.reset()
            assert index.ntotal == 0, "index not reset after block (reset bug)"
            del emb, ids
    finally:
        index.reset()
    merged_D, merged_I = merge_fn(per_block, topN)
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)

    # ---- PHASE C: slice + emit per spec (reproduces evaluation.py:467-648) ----
    for spec, (lo, hi), qids in zip(group, slices, per_spec_qids):
        a = spec.args
        paths = _output_paths(spec)
        for sub in ("ranking", "metrics", "per_query_metrics", "ikat_format_output"):
            os.makedirs(os.path.join(paths["dir"], sub), exist_ok=True)
        a.ranking_list_path = paths["ranking"]
        a.file_name_stem = spec.file_name_stem
        a.file_name_stem_without_group = spec.file_name_stem   # B3 (== stem for personalization_group "all")
        a.save_ranking_list = True                             # B6: guarantee ranking is written

        hits = get_dense_ranking_list(qids, merged_D[lo:hi], merged_I[lo:hi], int(a.retrieval_top_k))
        _hits2, run = get_run_object_and_save_ranking_list(hits, a)   # (hits, run); writes ranking .txt

        # B2: delete the 6 query-list keys before vars(args) is dumped (evaluation.py:493-500)
        for k in ("retrieval_query_list", "reranking_query_list", "fusion_query_lists", "qid_list_string"):
            if hasattr(a, k):
                delattr(a, k)
        for k in ("qid_personalized_level_dict", "qid_weights_dict"):
            if hasattr(a, k):
                delattr(a, k)

        if a.run_eval:
            metrics_list = a.metrics.split(",")
            key_form = [m.replace(".", "_") for m in metrics_list]
            query_metrics_dic, averaged_metrics = evaluate(
                run, a.qrel_file_path, paths["ranking"], metrics_list, key_form)
            _save_metrics(paths, a, averaged_metrics, query_metrics_dic, key_form)
        print(f"[grouped]  emitted {a.retrieval_model} x {a.retrieval_query_type} x {a.topics}")


def _save_metrics(paths, a, averaged_metrics, query_metrics_dic, key_form):
    """Byte-faithful to evaluation.py:565-648 (save_results_to_object + the metrics files)."""
    # B7: write per-turn results back into the input topic json
    if a.save_results_to_object:
        turn_list = load_turns_from_json(input_topic_path=a.input_query_path, range_start=0, range_end=-1)
        for qid, result_dict in query_metrics_dic.items():
            response = "rag_not_run, no response."   # generation_model == "none" for these specs
            for turn in turn_list:
                if str(turn.turn_id) == qid:
                    turn.add_result(a.collection, a.retrieval_model, a.reranker, a.generation_model,
                                    a.retrieval_query_type, a.reranking_query_type, a.generation_query_type,
                                    result_dict, response)
        save_turns_to_json(turn_list, a.input_query_path)

    # B1: formatted-latex header + the two JSON dicts (exact evaluation.py:609-632)
    formatted_metrics = print_formatted_latex_metrics(averaged_metrics, a.metrics_to_print)
    mp = paths["metrics"]
    with open(mp, "w") as f:
        f.write("Print this line to your latex table:\n")
        f.write("-------------------------------------\n")
        f.write("    " + formatted_metrics + "\n")
        f.write("-------------------------------------\n")
    with open(mp, "a") as f:
        f.write("\n"); json.dump(averaged_metrics, f, indent=4)
    with open(mp, "a") as f:
        f.write("\n"); f.write(json.dumps(vars(a), indent=4))

    # B4: per-query dict with indent=4
    with open(paths["per_query"], "w") as f:
        json.dump(query_metrics_dic, f, indent=4)

    # B5: append one line per metric to metrics/{metric}.txt (append-only -> clean dir for byte-compare)
    for metric_name in key_form:
        with open(os.path.join(paths["dir"], "metrics", f"{metric_name}.txt"), "a") as f:
            f.write(a.file_name_stem + f"-[{averaged_metrics[metric_name]}]\n")
