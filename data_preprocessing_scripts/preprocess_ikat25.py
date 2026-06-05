"""
Preprocess the TREC iKAT 2025 (offline / passage-ranking track) test topics and
NIST qrels into the same internal format used for iKAT 2023 / 2024.

Source of raw data: https://github.com/irlabamsterdam/ikat2025  (offline/)
  - offline/2025_test_topics.json   -> multi-turn conversations
  - offline/qrel/qrels-nist.trec    -> NIST passage relevance judgments (0-4)

Why a dedicated script (vs. reusing load_turns_from_ikat_topic_files):
  iKAT 2025 renamed several raw fields relative to 2023/2024:
      turns              -> responses
      utterance          -> user_utterance
      response_provenance-> citations
      ptkb_provenance    -> relevant_ptkbs  (now PTKB *strings*, not indices)
      ptkb (dict)        -> ptkb (list of strings)
  The processed schema we emit is byte-for-byte the 2023/2024 one
  (see Turn.to_dict / from_dict in src/apcir/functional/topics.py), so the
  existing evaluation pipeline (load_turns_from_json) can read it unchanged.

qid alignment (IMPORTANT):
  The 2025 NIST qrel uses qids like  "9-1_10"  =  "{number}_{turn_id}".
  But Turn.get_turn_order() does int(turn_id.split("-")[-1]), and 2023/2024
  processed turn_ids look like "9-1-3". To stay consistent with 2023/2024 AND
  keep get_turn_order working, we emit processed turn_id = "{number}-{turn_id}"
  ("9-1-10") and rewrite the qrel qids "9-1_10" -> "9-1-10" (replace "_"→"-").

Outputs:
  - data/topics/ikat25/ikat_2025_test.json   (processed topics, 23/24 schema)
  - data/qrels/ikat_2025_qrel.txt            (qrel with hyphen qids)
  - appends summary stats to logs/data_analysis.log

Run:
  cd /data/rech/huiyuche/TREC_iKAT_2024
  python data_preprocessing_scripts/preprocess_ikat25.py
"""

import json
import os
from datetime import datetime, timezone

# ----- paths (relative to repo root) ---------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TOPICS  = os.path.join(REPO_ROOT, "data/topics/ikat25/2025_test_topics.json")
RAW_QREL    = os.path.join(REPO_ROOT, "data/topics/ikat25/qrels-nist.trec")
OUT_TOPICS  = os.path.join(REPO_ROOT, "data/topics/ikat25/ikat_2025_test.json")
OUT_QREL    = os.path.join(REPO_ROOT, "data/qrels/ikat_2025_qrel.txt")
ANALYSIS_LOG = os.path.join(REPO_ROOT, "logs/data_analysis.log")


# Known verbatim typos in the released raw PTKB strings -> cleaned form.
# (e.g. persona 6-2 has a duplicated leading "I": "I I am a 40-year-old woman.")
PTKB_TYPO_FIXES = {
    "I I am a 40-year-old woman.": "I am a 40-year-old woman.",
}


def _fix_ptkb(s):
    return PTKB_TYPO_FIXES.get(s, s)


def build_turns(raw_data):
    """Flatten the raw 2025 conversations into a list of processed turn dicts
    that match the 2023/2024 schema."""
    list_of_turns = []
    for conv in raw_data:
        number = conv["number"]                 # e.g. "9-1"
        title  = conv["title"]
        ptkb_list = [_fix_ptkb(s) for s in conv.get("ptkb", [])]   # list[str], typo-fixed

        # 2023/2024 store ptkb as a 1-indexed dict {"1": str, ...}. Mirror that.
        ptkb_dict = {str(i + 1): s for i, s in enumerate(ptkb_list)}
        # map a PTKB string back to its 1-based index (for relevant_ptkbs)
        ptkb_str2idx = {s: i + 1 for i, s in enumerate(ptkb_list)}

        prior_user_utterances = []              # -> context_utterances
        for turn in conv["responses"]:          # 2025: "responses" (was "turns")
            turn_id = f"{number}-{turn['turn_id']}"        # "9-1-10"

            # relevant_ptkbs are PTKB *strings* in 2025 -> convert to indices
            relevant_ptkbs = [_fix_ptkb(s) for s in (turn.get("relevant_ptkbs", []) or [])]
            ptkb_provenance = [ptkb_str2idx[s] for s in relevant_ptkbs
                               if s in ptkb_str2idx]

            processed = {
                "turn_id": turn_id,
                "conversation_id": str(number),
                "title": title,
                "current_utterance": turn["user_utterance"],     # renamed
                "current_response": turn.get("response", ""),
                "oracle_utterance": turn["resolved_utterance"],  # human rewrite
                "response_provenance": turn.get("citations", []) or [],  # renamed
                "context_utterances": list(prior_user_utterances),
                "ptkb": ptkb_dict,
                "ptkb_provenance": ptkb_provenance,
                "reformulations": [],   # no LLM rewrites for 2025 (oracle is enough)
                "results": [],
            }
            list_of_turns.append(processed)
            prior_user_utterances.append(turn["user_utterance"])

    return list_of_turns


def convert_qrel(raw_qrel_path, out_qrel_path):
    """Rewrite qid "{number}_{turn}" -> "{number}-{turn}" so it matches the
    processed turn_id format. Returns the set of judged qids (hyphen form)."""
    judged_qids = set()
    n_lines = 0
    with open(raw_qrel_path, "r", encoding="utf-8") as fin, \
         open(out_qrel_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, q0, docid, rel = parts
            qid = qid.replace("_", "-")          # "9-1_10" -> "9-1-10"
            fout.write(f"{qid} {q0} {docid} {rel}\n")
            judged_qids.add(qid)
            n_lines += 1
    return judged_qids, n_lines


def main():
    with open(RAW_TOPICS, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    turns = build_turns(raw_data)
    os.makedirs(os.path.dirname(OUT_TOPICS), exist_ok=True)
    with open(OUT_TOPICS, "w", encoding="utf-8") as f:
        json.dump(turns, f, indent=4)

    os.makedirs(os.path.dirname(OUT_QREL), exist_ok=True)
    judged_qids, n_qrel_lines = convert_qrel(RAW_QREL, OUT_QREL)

    # ---- verification / stats --------------------------------------------
    processed_qids = {t["turn_id"] for t in turns}
    n_conv = len(raw_data)
    n_turns = len(turns)
    judged_in_topics = judged_qids & processed_qids
    judged_missing   = judged_qids - processed_qids

    lines = [
        f"[{datetime.now(timezone.utc).isoformat()}] iKAT-2025 preprocessing",
        f"  conversations (topics)          : {n_conv}",
        f"  total turns (all)               : {n_turns}",
        f"  judged qids in qrel             : {len(judged_qids)}",
        f"  judged qids present in topics   : {len(judged_in_topics)}",
        f"  judged qids MISSING from topics : {sorted(judged_missing)}",
        f"  qrel lines written              : {n_qrel_lines}",
        f"  topics out : {OUT_TOPICS}",
        f"  qrel out   : {OUT_QREL}",
    ]
    report = "\n".join(lines)
    print(report)

    os.makedirs(os.path.dirname(ANALYSIS_LOG), exist_ok=True)
    with open(ANALYSIS_LOG, "a", encoding="utf-8") as f:
        f.write(report + "\n")

    if judged_missing:
        print("\nWARNING: some judged qids are not found in the processed topics "
              "(check turn_id construction / raw data).")
    else:
        print("\nOK: every judged qid maps to a processed turn.")


if __name__ == "__main__":
    main()
