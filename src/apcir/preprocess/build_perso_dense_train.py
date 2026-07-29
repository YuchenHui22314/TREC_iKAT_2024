"""仿 build_perso_dense_val.py 但 TRAIN split — 给 hard-neg miner 提供 train 280 turn 的 oracle
检索输入. hard-neg = oracle query 在全库 top docs 里的非 judged 文档. oracle_qwen_instruct query
type 用 turn.oracle_utterance(在 Turn dict 里, orig 保留), 不需要 reformulations.

只用 ORIG 280 train turn(排除 aug 视图 —— 它们的 oracle 与 base turn 相同, 不增 unique oracle).

Outputs:
  - data/topics/perso_dense_train/perso_dense_train.json
  - data/qrels/perso_dense_train_qrel.txt
Run from src/:  python -m apcir.preprocess.build_perso_dense_train
"""
import os, json, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = "/data/rech/huiyuche/TREC_iKAT_2024"
GRADED_DIR = os.path.join(REPO, "data", "ikat_graded")
YEAR_TOPIC = {23: "data/topics/ikat23/ikat_2023_test.json",
              24: "data/topics/ikat24/ikat_2024_test.json",
              25: "data/topics/ikat25/ikat_2025_test.json"}
OUT_TOPIC = os.path.join(REPO, "data", "topics", "perso_dense_train", "perso_dense_train.json")
OUT_QREL = os.path.join(REPO, "data", "qrels", "perso_dense_train_qrel.txt")


def load_manifest(form):
    path = os.path.join(GRADED_DIR, f"ikat_graded_{form}.jsonl")
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(path))}


def main():
    ptkb = load_manifest("ptkb"); rel = load_manifest("rel_ptkb")
    train_ids = [sid for sid, r in ptkb.items() if r["split"] == "train" and "_aug" not in sid]
    logger.info(f"orig train turns: {len(train_ids)}")
    orig = {}
    for y, p in YEAR_TOPIC.items():
        for t in json.load(open(os.path.join(REPO, p))):
            orig[(y, str(t["turn_id"]))] = t
    out_turns, qrel_lines = [], []
    for sid in train_ids:
        row = ptkb[sid]
        td = dict(orig[(row["year"], row["qid"])])
        td["turn_id"] = sid; td["results"] = []
        refs = list(td.get("reformulations", []))
        refs.append({"reformulation_name": "perso_dense_val_ptkb",
                     "reformulated_query": ptkb[sid]["query_text"], "ptkb_provenance": []})
        refs.append({"reformulation_name": "perso_dense_val_rel_ptkb",
                     "reformulated_query": rel[sid]["query_text"], "ptkb_provenance": []})
        td["reformulations"] = refs
        out_turns.append(td)
        for docid, grade in ptkb[sid]["candidates"]:
            qrel_lines.append(f"{sid} 0 {docid} {grade}")
    os.makedirs(os.path.dirname(OUT_TOPIC), exist_ok=True)
    json.dump(out_turns, open(OUT_TOPIC, "w"))
    with open(OUT_QREL, "w") as f:
        f.write("\n".join(qrel_lines) + "\n")
    logger.info(f"wrote {len(out_turns)} turns -> {OUT_TOPIC}; {len(qrel_lines)} qrel lines -> {OUT_QREL}")


if __name__ == "__main__":
    main()
