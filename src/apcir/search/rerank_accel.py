"""Multi-GPU reranking via HuggingFace accelerate — the RIGHT no-sync data parallelism
for inference (one model per GPU process, shard the queries, gather). Works for the
qwen3 reranker (monot5 can be wired the same way). Unlike DataParallel (re-replicates
per forward) or DDP (gradient-sync; training-only).

TWO modes:
  --config <yaml> : process the WHOLE fuse_then_eval yaml (ALL iterate combos / settings)
                    in ONE launch -> the model is loaded ONCE per process for the whole
                    batch (N loads total, NOT N*num_settings). This is the mode to use when
                    you test many settings at once. RERANK-ONLY: each combo loads its
                    existing no-rerank first-stage ranking (disk-load), reranks top-k.
  --ranking_in/out: single first-stage ranking -> reranked ranking (one job).

Run:
    accelerate launch --num_processes 4 --multi_gpu -m apcir.search.rerank_accel \
        --config ./apcir/evaluate/fuse_then_eval_config_table13_qwen_rerank.yaml

CORRECTNESS: queries are sharded WHOLE (each query's top-k reranked as one batch on one
GPU), so the output ranking is BYTE-IDENTICAL to a 1-process run (verify --num_processes
1 vs N). The reranker model loads once per process for the entire --config batch.
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import yaml
from accelerate import Accelerator
from accelerate.utils import gather_object
from pyserini.search.lucene import LuceneSearcher

from apcir.evaluate.evaluation_util import get_query_list, evaluate, print_formatted_latex_metrics
from apcir.search.search import load_ranking_list_from_file
from apcir.search.rerank import (
    QwenReranker, _fetch_contents_cached,
    QWEN3_RERANK_CONV_INSTRUCTION, QWEN3_RERANK_DEFAULT_INSTRUCTION,
)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="fuse_then_eval yaml: rerank ALL combos in one launch")
    # single-job mode (used when --config is absent)
    p.add_argument("--ranking_in"); p.add_argument("--ranking_out")
    p.add_argument("--topics"); p.add_argument("--input_query_path")
    p.add_argument("--retrieval_model", default="BM25")
    p.add_argument("--retrieval_query_type", default="oracle")
    p.add_argument("--reranking_query_type", default="oracle")
    p.add_argument("--sparse_index_dir_path")
    p.add_argument("--qrel_file_path", default=None)
    # reranker
    p.add_argument("--reranker", default="qwen3_reranker")
    p.add_argument("--qwen3_reranker_path", default="Qwen/Qwen3-Reranker-4B")
    p.add_argument("--cache_dir", default="/data/rech/huiyuche/huggingface")
    p.add_argument("--rerank_quant", default="none")
    p.add_argument("--rerank_top_k", type=int, default=50)
    p.add_argument("--rerank_batch_size", type=int, default=8)
    return p


def file_name_stem(a):
    """Exactly evaluation.py:391 — so the input (no-rerank) and output (reranked) stems
    match the on-disk ranking files."""
    qe = "_rm3" if getattr(a, "qe_type", "none") == "rm3" else ""
    pg = "" if a.personalization_group == "all" else f"_{a.personalization_group}"
    return (f"S1[{a.retrieval_query_type}{pg}]-S2[{a.reranking_query_type}]-"
            f"g[{a.generation_query_type}]-[{a.retrieval_model}{qe}]-"
            f"[{a.reranker}_{a.window_size}_{a.step}_{a.rerank_quant}]-[s2_top{a.rerank_top_k}]")


def expand_config(path):
    """Reproduce run_experiments combo expansion -> list of flat param dicts (one per setting)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    fixed, it, pm = cfg["fixed"], cfg["iterate"], cfg["param_mapping"]
    keys = list(it.keys())
    import itertools
    out = []
    for combo in itertools.product(*[it[k] for k in keys]):
        d = deepcopy(fixed)
        pdict = dict(zip(keys, combo))
        for pname, mapping in pm.items():
            for pval, assoc in mapping.items():
                if pname in pdict and pdict[pname] == pval:
                    d.update(assoc)
        d.update(pdict)
        out.append(d)
    return out


def prep_combo(d):
    """From a flat param dict -> (args, qid2query, first_stage_hits, instruction, in_path, out_paths).
    Returns None if the no-rerank first-stage ranking is missing."""
    a = SimpleNamespace(**d)
    in_stem = file_name_stem(SimpleNamespace(**{**d, "reranking_query_type": "none", "reranker": "none"}))
    out_stem = file_name_stem(a)
    rdir = os.path.join(a.output_dir_path, a.collection, a.topics, "ranking")
    in_path = os.path.join(rdir, in_stem + ".txt")
    if not os.path.exists(in_path):
        return None
    rql, rrl, gql, fql, qids, plvl, wts, turns = get_query_list(a)
    qid2query = {q: rq for q, rq in zip(qids, rrl)}
    hits = load_ranking_list_from_file(in_path)
    instr = (QWEN3_RERANK_CONV_INSTRUCTION if a.reranking_query_type == "qwen_3_rerank_instruct_full"
             else QWEN3_RERANK_DEFAULT_INSTRUCTION)
    mdir = os.path.join(a.output_dir_path, a.collection, a.topics, "metrics")
    return {"args": a, "qid2query": qid2query, "hits": hits, "qids": [q for q in qids if q in hits],
            "instruction": instr, "out_ranking": os.path.join(rdir, out_stem + ".txt"),
            "out_metrics": os.path.join(mdir, out_stem + ".json"), "stem": out_stem,
            "rdir": rdir, "mdir": mdir}


def main():
    a = build_parser().parse_args()
    acc = Accelerator()
    device = str(acc.device)

    # ---- build the combo list (every process; cheap CPU work) ----
    if a.config:
        combos = [c for c in (prep_combo(d) for d in expand_config(a.config)) if c is not None]
    else:
        d = {k: v for k, v in vars(a).items()}
        d.update({"output_dir_path": os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(a.ranking_in)))) or ".",
                  "collection": "ClueWeb_ikat", "generation_query_type": "none",
                  "personalization_group": "all", "window_size": 4, "step": 1, "qe_type": "none",
                  "fb_terms": 20, "original_query_weight": 0.5, "fusion_type": "none",
                  "QRs_to_rank": [], "level_type": "none", "fuse_weights": [], "fusion_query_lists": [],
                  "qid_personalized_level_dict": {}})
        c = prep_combo(d) if False else None  # single-job uses a simpler path below
        combos = None
    if acc.is_main_process:
        n = len(combos) if combos is not None else 1
        print(f"[rerank_accel] {n} setting(s); loading {a.reranker} ONCE per process on {device}")

    # ---- load the reranker ONCE on THIS process's GPU (shared by ALL combos) ----
    if a.reranker == "qwen3_reranker":
        reranker = QwenReranker(model_path=a.qwen3_reranker_path, cache_dir=a.cache_dir,
                                quant=a.rerank_quant, device=device)
    else:
        raise NotImplementedError(f"{a.reranker} not wired (qwen3_reranker works)")

    # ---- flatten ALL (combo, qid) work across all settings ----
    if a.config:
        searcher = LuceneSearcher(combos[0]["args"].sparse_index_dir_path)
        work = [(ci, qid) for ci, c in enumerate(combos) for qid in c["qids"]]
    else:
        searcher = LuceneSearcher(a.sparse_index_dir_path)
        single = single_prep(a)
        combos = [single]
        work = [(0, qid) for qid in single["qids"]]

    doc_cache = {}
    local = {}
    with acc.split_between_processes(work) as shard:
        for ci, qid in shard:
            c = combos[ci]; a2 = c["args"]
            top = c["hits"][qid][:a2.rerank_top_k]
            docs = [_fetch_contents_cached(searcher, d.docid, doc_cache) for d in top]
            scores = reranker.score(c["instruction"], c["qid2query"][qid], docs, a.rerank_batch_size)
            order = np.argsort(np.array(scores, dtype=np.float32))[::-1]
            reranked = [top[i].docid for i in order] + [d.docid for d in c["hits"][qid][a2.rerank_top_k:]]
            local[(ci, qid)] = reranked

    # ---- gather + regroup by combo (rank 0 only) ----
    gathered = gather_object([local])
    if not acc.is_main_process:
        return
    merged = {}
    for g in gathered:
        merged.update(g)

    for ci, c in enumerate(combos):
        a2 = c["args"]
        os.makedirs(c["rdir"], exist_ok=True); os.makedirs(c["mdir"], exist_ok=True)
        run = {}
        with open(c["out_ranking"], "w") as f:
            for qid in c["qids"]:
                rr = merged.get((ci, qid))
                if rr is None:
                    continue
                run[qid] = {docid: 1.0/(r+1) for r, docid in enumerate(rr)}
                for r, docid in enumerate(rr):
                    f.write(f"{qid} Q0 {docid} {r+1} {1.0/(r+1)} {c['stem']}\n")
        if getattr(a2, "qrel_file_path", None):
            mlist = a2.metrics.split(",") if hasattr(a2, "metrics") else \
                "map,ndcg_cut.3,recall.10,recall.100,recip_rank".split(",")
            keyf = [m.replace(".", "_") for m in mlist]
            _, avg = evaluate(run, a2.qrel_file_path, c["out_ranking"], mlist, keyf)
            mtp = getattr(a2, "metrics_to_print", ["recip_rank", "ndcg_cut_3", "recall_10", "recall_100"])
            with open(c["out_metrics"], "w") as f:
                f.write("Print this line to your latex table:\n-------------------------------------\n")
                f.write("    " + print_formatted_latex_metrics(avg, mtp) + "\n")
                f.write("-------------------------------------\n\n")
                json.dump(avg, f, indent=4)
            print(f"[rerank_accel] {c['stem']}: "
                  f"{ {k: round(avg[k]*100,1) for k in ['recip_rank','ndcg_cut_3','recall_10','recall_100'] if k in avg} }")
        else:
            print(f"[rerank_accel] wrote {c['out_ranking']}")
    print(f"[rerank_accel] DONE: {len(combos)} settings reranked in one launch")


def single_prep(a):
    d = {**vars(a), "output_dir_path": ".", "collection": "ClueWeb_ikat",
         "generation_query_type": "none", "personalization_group": "all", "window_size": 4,
         "step": 1, "qe_type": "none", "fb_terms": 20, "original_query_weight": 0.5,
         "fusion_type": "none", "QRs_to_rank": [], "level_type": "none", "fuse_weights": [],
         "fusion_query_lists": [], "qid_personalized_level_dict": {}, "metrics_to_print":
         ["recip_rank", "ndcg_cut_3", "recall_10", "recall_100"]}
    a2 = SimpleNamespace(**d)
    rql, rrl, gql, fql, qids, plvl, wts, turns = get_query_list(a2)
    qid2query = {q: rq for q, rq in zip(qids, rrl)}
    hits = load_ranking_list_from_file(a.ranking_in)
    instr = (QWEN3_RERANK_CONV_INSTRUCTION if a.reranking_query_type == "qwen_3_rerank_instruct_full"
             else QWEN3_RERANK_DEFAULT_INSTRUCTION)
    return {"args": a2, "qid2query": qid2query, "hits": hits, "qids": [q for q in qids if q in hits],
            "instruction": instr, "out_ranking": a.ranking_out, "stem": a.run_name if hasattr(a, "run_name") else "rerank_accel",
            "rdir": os.path.dirname(a.ranking_out) or ".", "mdir": os.path.dirname(a.ranking_out) or ".",
            "out_metrics": a.ranking_out + ".metrics.json"}


if __name__ == "__main__":
    main()
