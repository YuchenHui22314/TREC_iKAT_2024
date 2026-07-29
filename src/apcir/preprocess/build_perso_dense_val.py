"""Emit the `perso_dense_val` offline-eval inputs from the already-built graded manifests.

`perso_dense_val` is a `topics` option for the offline fuse_then_eval / ClueWeb retrieval
pipeline whose query set = the HELD-OUT (val-split) turns of the personalized-dense-retriever
experiment. We reuse the pre-built per-form query strings from the manifests by baking them as
`reformulations` on each turn (names `perso_dense_val_ptkb` / `perso_dense_val_rel_ptkb`), so the
pipeline returns them via `Turn.query_type_2_query`'s reformulation fallback — no builders, no
year-coupled asserts, no conversation-context reconstruction needed.

Outputs (CPU-only, instant; reads the manifests + original flattened topics — NO index):
  - data/topics/perso_dense_val/perso_dense_val.json  (val turns, full Turn dicts + 2 reformulations)
  - data/qrels/perso_dense_val_qrel.txt               (val turns' graded qrel; form-independent)

Run from src/:  python -m apcir.preprocess.build_perso_dense_val
"""
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = "/data/rech/huiyuche/TREC_iKAT_2024"
GRADED_DIR = os.path.join(REPO, "data", "ikat_graded")
YEAR_TOPIC = {
    23: "data/topics/ikat23/ikat_2023_test.json",
    24: "data/topics/ikat24/ikat_2024_test.json",
    25: "data/topics/ikat25/ikat_2025_test.json",
}
OUT_TOPIC = os.path.join(REPO, "data", "topics", "perso_dense_val", "perso_dense_val.json")
OUT_QREL = os.path.join(REPO, "data", "qrels", "perso_dense_val_qrel.txt")


def load_manifest(form):
    path = os.path.join(GRADED_DIR, f"ikat_graded_{form}.jsonl")
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(path))}


def main():
    ptkb = load_manifest("ptkb")
    rel = load_manifest("rel_ptkb")
    # val turns (split is identical across forms; use ptkb as the reference)
    val_ids = [sid for sid, r in ptkb.items() if r["split"] == "val"]
    logger.info(f"val turns: {len(val_ids)}")

    # original full Turn dicts, keyed by (year, bare qid) — bare qids collide across years
    orig = {}
    for y, p in YEAR_TOPIC.items():
        for t in json.load(open(os.path.join(REPO, p))):
            orig[(y, str(t["turn_id"]))] = t

    out_turns, qrel_lines = [], []
    for sid in val_ids:
        row = ptkb[sid]
        td = dict(orig[(row["year"], row["qid"])])   # full Turn dict (year-aware lookup)
        td["turn_id"] = sid                  # YEAR-QUALIFIED unique id (== qrel qid == run qid)
        td["results"] = []                   # strip prior eval results (not needed; keeps file small)
        refs = list(td.get("reformulations", []))
        refs.append({"reformulation_name": "perso_dense_val_ptkb",
                     "reformulated_query": ptkb[sid]["query_text"], "ptkb_provenance": []})
        refs.append({"reformulation_name": "perso_dense_val_rel_ptkb",
                     "reformulated_query": rel[sid]["query_text"], "ptkb_provenance": []})
        td["reformulations"] = refs
        out_turns.append(td)
        for docid, grade in ptkb[sid]["candidates"]:   # candidates form-independent
            qrel_lines.append(f"{sid} 0 {docid} {grade}")

    os.makedirs(os.path.dirname(OUT_TOPIC), exist_ok=True)
    json.dump(out_turns, open(OUT_TOPIC, "w"))
    with open(OUT_QREL, "w") as f:
        f.write("\n".join(qrel_lines) + "\n")
    logger.info(f"wrote {len(out_turns)} turns -> {OUT_TOPIC}")
    logger.info(f"wrote {len(qrel_lines)} qrel lines -> {OUT_QREL}")
    # sanity
    by_year = {}
    for sid in val_ids:
        y = ptkb[sid]["year"]; by_year[y] = by_year.get(y, 0) + 1
    logger.info(f"val turns per year: {by_year}")


if __name__ == "__main__":
    main()
