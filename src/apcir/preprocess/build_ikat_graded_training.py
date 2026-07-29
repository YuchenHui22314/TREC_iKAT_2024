"""Build GRADED-relevance training files for the personalized conversational dense retriever.

We fine-tune a query encoder (query-encoder-only; document embeddings are the FROZEN base-qwen3
ClueWeb22-B index) on the TREC iKAT 2023/2024/2025 graded qrels (relevance 0-4). This script,
which lives in TREC_iKAT_2024 (where the qrels / topics / query-builders / index live), produces
the files the continual_ir training session consumes.

Two phases:
  - MANIFEST (cheap, CPU, seconds; needs no index): per query FORM emit
    ``ikat_graded_{form}.jsonl`` -- one line per judged turn with the fully-built query string,
    the list of (docid, grade) candidates, and a by-CONVERSATION 9:1 train/val split.
  - EMBEDDINGS (expensive, one ~450G stream over the merged index, ~1h on the spinning HDD;
    SHARED across forms/inits): extract the frozen base-qwen3 embedding of every judged docid ->
    ``ikat_graded_doc_embeddings.pt`` = ``{docid: Tensor(1024) float32}``.

The query string is produced by the SAME builder the offline ``fuse_then_eval`` pipeline uses
(``Turn.query_type_2_query``), with the SAME context attachment as
``evaluation_util.get_query_list`` (fullconv_ctx, and for iKAT-25 applicable_new_ptkb), so the
training query == the eval query byte-for-byte.

Run from src/  (so ``apcir`` is importable):
  # cheap manifest-only smoke (verify format first):
  python -m apcir.preprocess.build_ikat_graded_training --manifest_only
  # full build (manifests + the shared doc-embedding .pt):
  python -m apcir.preprocess.build_ikat_graded_training
"""

import os
import gc
import json
import glob
import pickle
import random
import logging
import argparse
from collections import defaultdict
from types import SimpleNamespace

from apcir.functional.topics import load_turns_from_json

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-year file layout. `topics` is the string the query builders assert on
# (e.g. the iKAT-25-only forms require topics == "ikat_25_test").
# NOTE the iKAT-25 qrel file is ikat_2025_qrel.txt (four-digit), unlike 23/24.
# ---------------------------------------------------------------------------
YEAR_CFG = {
    "23": dict(topic="data/topics/ikat23/ikat_2023_test.json",
               qrel="data/qrels/ikat_23_qrel.txt",   topics="ikat_23_test"),
    "24": dict(topic="data/topics/ikat24/ikat_2024_test.json",
               qrel="data/qrels/ikat_24_qrel.txt",   topics="ikat_24_test"),
    "25": dict(topic="data/topics/ikat25/ikat_2025_test.json",
               qrel="data/qrels/ikat_2025_qrel.txt", topics="ikat_25_test"),
}

# Query FORM -> the topics.py query_type used to build the query string, per year.
#   ptkb     : full user profile + conversation        (qwen_conversation_ptkb)
#   rel_ptkb : relevant-only profile + conversation     (qwen_conversation_rel_ptkb);
#              iKAT-25 turns additionally fold in the carried-over new_ptkb facts
#              (qwen_conversation_rel_new_ptkb).
FORM_QTYPE = {
    "ptkb":     {"23": "qwen_conversation_ptkb",
                 "24": "qwen_conversation_ptkb",
                 "25": "qwen_conversation_ptkb"},
    "rel_ptkb": {"23": "qwen_conversation_rel_ptkb",
                 "24": "qwen_conversation_rel_ptkb",
                 "25": "qwen_conversation_rel_new_ptkb"},
}

REPO = "/data/rech/huiyuche/TREC_iKAT_2024"


# ---------------------------------------------------------------------------
# Context attachment -- faithful copy of evaluation_util.get_query_list (L92-145),
# minus prev_conv_ctx (none of our forms use the previous-conversation builder).
# ---------------------------------------------------------------------------
def attach_context(turns, topics_str, topic_path):
    conv_groups = defaultdict(list)
    for t in turns:
        conv_groups[t.conversation_id].append(t)
    # fullconv_ctx[t] = interleaved [u1, r1, ..., u_{k-1}, r_{k-1}] of everything BEFORE turn t
    for conv_turns in conv_groups.values():
        conv_turns.sort(key=lambda x: x.get_turn_order())
        hist = []
        for t in conv_turns:
            t.fullconv_ctx = list(hist)
            hist.append(t.current_utterance)
            hist.append(t.current_response)

    # iKAT-25 only: applicable_new_ptkb = organizer-oracle carried-over facts from
    # ptkb-update.json (sibling of the topics file) whose turn_dependence includes this turn.
    if topics_str == "ikat_25_test":
        upd = os.path.join(os.path.dirname(topic_path), "ptkb-update.json")
        new_map = {}
        if os.path.exists(upd):
            for e in json.load(open(upd)):
                new_map[str(e["number"])] = e.get("new_ptkb", [])
        else:
            logger.warning(f"ptkb-update.json not found at {upd}; applicable_new_ptkb will be empty")
        for cid, conv_turns in conv_groups.items():
            entries = new_map.get(str(cid), [])
            for t in conv_turns:
                tno = t.get_turn_order()
                t.applicable_new_ptkb = [n["statement"].strip() for n in entries
                                         if tno in (n.get("turn_dependence") or [])]


# ---------------------------------------------------------------------------
# qrel parsing.  TREC format: "qid Q0 docid grade".  Grade -1 (iKAT-24 only,
# 143 cases = not-judged) is dropped by default.
# ---------------------------------------------------------------------------
def parse_qrel(path, drop_minus1=True):
    qrel = defaultdict(list)
    grade_hist = defaultdict(int)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid, grade = parts[0], parts[1], parts[2], int(parts[3])
            grade_hist[grade] += 1
            if drop_minus1 and grade < 0:
                continue
            qrel[qid].append((docid, grade))
    return qrel, grade_hist


# ---------------------------------------------------------------------------
# One-pass docid -> base-qwen3 embedding extraction from the merged index.
# Drop-in of continual_ir/preprocess/data/extract_qwen_pos_neg_from_corpus.build_lookup_table.
# ---------------------------------------------------------------------------
def build_lookup_table(corpus_dir, needed_ids):
    import numpy as np  # local import; manifest-only mode needs no numpy/torch
    id_files = sorted(glob.glob(os.path.join(corpus_dir, "doc_embid_block.*.pb")))
    emb_files = sorted(glob.glob(os.path.join(corpus_dir, "doc_emb_block.*.pb")))
    assert len(id_files) == len(emb_files) and id_files, "id/emb block mismatch or none found"
    logger.info(f"Scanning {len(id_files)} blocks for {len(needed_ids)} doc ids ...")
    lookup = {}
    remaining = set(needed_ids)
    for id_file, emb_file in zip(id_files, emb_files):
        ids = pickle.load(open(id_file, "rb"))
        hits = remaining & set(ids)
        logger.info(f"  {os.path.basename(emb_file)}: {len(hits)} hits "
                    f"({len(remaining)} remaining)")
        if not hits:
            continue
        embs = pickle.load(open(emb_file, "rb"))  # np (N, 1024) float32
        for row, doc_id in enumerate(ids):
            if doc_id in hits:
                lookup[doc_id] = embs[row].copy()
                remaining.discard(doc_id)
        del embs
        gc.collect()
        if not remaining:
            logger.info("All needed doc ids found. Early stop.")
            break
    if remaining:
        logger.warning(f"{len(remaining)} doc ids NOT found in index "
                       f"(e.g. {list(remaining)[:3]})")
    return lookup


# ---------------------------------------------------------------------------
def collect_kept_turns(years, drop_minus1):
    """Return (kept, stats). kept = list of dicts {qid, year, conv_id, turn, candidates}
    for every judged turn that has at least one grade>0 doc."""
    kept = []
    stats = {}
    for y in years:
        cfg = YEAR_CFG[y]
        turns = load_turns_from_json(input_topic_path=os.path.join(REPO, cfg["topic"]))
        attach_context(turns, cfg["topics"], os.path.join(REPO, cfg["topic"]))
        qrel, grade_hist = parse_qrel(os.path.join(REPO, cfg["qrel"]), drop_minus1)

        n_kept, n_no_qrel, n_no_pos = 0, 0, 0
        for t in turns:
            qid = str(t.turn_id)
            if qid not in qrel:
                n_no_qrel += 1
                continue
            cands = qrel[qid]
            if not any(g > 0 for _, g in cands):
                n_no_pos += 1
                continue
            kept.append(dict(qid=qid, year=y, conv_id=str(t.conversation_id),
                             turn=t, candidates=cands))
            n_kept += 1
        stats[y] = dict(grade_hist=dict(grade_hist), n_turns_topics=len(turns),
                        n_qrel_qids=len(qrel), n_kept=n_kept,
                        n_no_qrel=n_no_qrel, n_no_pos=n_no_pos)
        logger.info(f"[ik{y}] kept {n_kept} turns "
                    f"(topics={len(turns)}, qrel_qids={len(qrel)}, "
                    f"no_pos={n_no_pos}); grades={dict(grade_hist)}")
    return kept, stats


def make_split(kept, val_frac, seed):
    """By-conversation 9:1 split, STRATIFIED by year (each year contributes ~val_frac of its turns
    to val, so EVERY year — esp. ik25, the newest test target — is represented; a global shuffle
    otherwise risks dropping a whole year out of val). Whole conversations held out (no leakage).
    conv_key = '<year>:<conversation_id>' to avoid cross-year id collisions."""
    ck = lambda k: f"{k['year']}:{k['conv_id']}"
    turns_in_conv = defaultdict(int)
    year_of_conv = {}
    for k in kept:
        turns_in_conv[ck(k)] += 1
        year_of_conv[ck(k)] = k["year"]
    convs_by_year = defaultdict(list)
    for c in sorted(turns_in_conv):            # deterministic base order
        convs_by_year[year_of_conv[c]].append(c)
    rng = random.Random(seed)
    val_convs = set()
    for y in sorted(convs_by_year):
        cs = list(convs_by_year[y]); rng.shuffle(cs)
        y_turns = sum(turns_in_conv[c] for c in cs)
        target = max(1, round(val_frac * y_turns))   # at least 1 conversation/year in val
        vc = 0
        for c in cs:
            if vc >= target:
                break
            val_convs.add(c); vc += turns_in_conv[c]
    split_map = {c: ("val" if c in val_convs else "train") for c in turns_in_conv}
    n_val = sum(turns_in_conv[c] for c in val_convs)
    logger.info(f"split (stratified by year): {len(kept)} turns over {len(turns_in_conv)} convs -> "
                f"val={n_val} turns / {len(val_convs)} convs, train={len(kept) - n_val}")
    return split_map, ck, val_convs, n_val


def build_query_text(turn, year, form):
    qtype = FORM_QTYPE[form][year]
    argshim = SimpleNamespace(retrieval_model="qwen3", topics=YEAR_CFG[year]["topics"])
    return qtype, turn.query_type_2_query(qtype, 0, 0.0, argshim)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    years = args.years.split()
    forms = args.forms.split()
    logger.info(f"years={years} forms={forms} manifest_only={args.manifest_only}")

    kept, stats = collect_kept_turns(years, drop_minus1=not args.keep_grade_minus1)
    split_map, ck, val_convs, n_val = make_split(kept, args.val_frac, args.seed)

    # ---- EMBEDDINGS phase (expensive; shared across forms/inits) ----
    missing = set()
    if not args.manifest_only:
        import torch
        union = sorted({d for k in kept for d, _ in k["candidates"]})
        logger.info(f"extracting {len(union)} unique judged-doc embeddings from {args.corpus_dir}")
        lookup = build_lookup_table(args.corpus_dir, set(union))
        missing = set(union) - set(lookup)
        # drop missing docids from candidate lists; drop turns that lose all positives
        before = len(kept)
        for k in kept:
            k["candidates"] = [(d, g) for d, g in k["candidates"] if d in lookup]
        kept = [k for k in kept if any(g > 0 for _, g in k["candidates"])]
        logger.info(f"missing-from-index docids: {len(missing)}; "
                    f"turns dropped after filtering: {before - len(kept)}")
        pt_path = os.path.join(args.out_dir, "ikat_graded_doc_embeddings.pt")
        torch.save({d: torch.from_numpy(lookup[d]) for d in lookup}, pt_path)
        logger.info(f"saved {len(lookup)} doc embeddings -> {pt_path}")

    # ---- MANIFEST phase (per form) ----
    for form in forms:
        out_path = os.path.join(args.out_dir, f"ikat_graded_{form}.jsonl")
        n = 0
        with open(out_path, "w") as fh:
            for k in kept:
                qtype, qtext = build_query_text(k["turn"], k["year"], form)
                # sample_id is YEAR-QUALIFIED ("{year}_{qid}") — bare qids collide across years
                # (e.g. 9-1-4 exists in both ik23 and ik25), which would corrupt per-example keying
                # in continual_ir and merge rows in the eval qrel. `qid` keeps the bare TREC id.
                rec = dict(sample_id=f"{k['year']}_{k['qid']}", qid=k["qid"],
                           conversation_id=k["conv_id"], year=int(k["year"]),
                           query_text=qtext,
                           candidates=[[d, g] for d, g in k["candidates"]],
                           split=split_map[ck(k)], template=qtype)
                fh.write(json.dumps(rec) + "\n")
                n += 1
        logger.info(f"[{form}] wrote {n} lines -> {out_path}")

    # ---- stats / data-analysis log ----
    n_train = len(kept) - sum(1 for k in kept if split_map[ck(k)] == "val")
    summary = {
        "years": years, "forms": forms, "manifest_only": args.manifest_only,
        "total_kept_turns": len(kept), "n_val_turns": sum(1 for k in kept if split_map[ck(k)] == "val"),
        "n_train_turns": n_train, "val_convs": sorted(val_convs),
        "missing_from_index": len(missing), "per_year": stats,
        "val_frac": args.val_frac, "seed": args.seed,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    try:
        with open(os.path.join(REPO, "logs", "data_analysis.log"), "a") as lg:
            from datetime import datetime  # timestamp only for the log line
            lg.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] build_ikat_graded_training "
                     f"{json.dumps(summary)}\n")
    except Exception as ex:
        logger.warning(f"could not append to data_analysis.log: {ex}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", default="23 24 25", help="space-separated subset of 23 24 25")
    p.add_argument("--forms", default="ptkb rel_ptkb", help="space-separated subset of ptkb rel_ptkb")
    p.add_argument("--out_dir", default=os.path.join(REPO, "data", "ikat_graded"))
    p.add_argument("--corpus_dir", default="/part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_grade_minus1", action="store_true",
                   help="keep iKAT-24 grade -1 entries (default: drop them)")
    p.add_argument("--manifest_only", action="store_true",
                   help="only build the JSONL manifests (skip the ~1h doc-embedding extraction)")
    main(p.parse_args())
