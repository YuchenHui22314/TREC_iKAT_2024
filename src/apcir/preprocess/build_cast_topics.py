"""Convert official TREC CAsT topic files into the pipeline's Turn JSON schema.

CAsT 2019/2020 share one collection (MS MARCO v1 passages + TREC CAR), so both years are
handled here. The output is the SAME schema `load_turns_from_json` already reads for iKAT,
so no loader changes are needed — only the topic name has to be registered in
evaluation.py / evaluation_util.py.

Field mapping (CAsT -> Turn):
  turn_id             = "{topic_number}_{turn_number}"   <- matches the qrel qid exactly
  conversation_id     = str(topic_number)
  current_utterance   = raw_utterance                    (the context-dependent question)
  oracle_utterance    = the MANUAL rewrite               (2019: separate .tsv; 2020: manual topics file)
  context_utterances  = all previous raw utterances of the same conversation
  reformulations      = [{"reformulation_name": "cast_automatic_rewrite", ...}] when the year
                        ships an automatic rewrite (2020 does, 2019 does not)

There is NO user profile in CAsT: `ptkb` is written as an empty dict so that any
profile-dependent query type fails loudly rather than silently producing an empty profile.

Usage
-----
    cd src
    python -m apcir.preprocess.build_cast_topics --year 19 \
        --raw_topics   <c19_topics.json> \
        --manual_tsv   <evaluation_topics_annotated_resolved_v1.0.tsv> \
        --out          ../data/topics/cast19/cast_19_test.json

    python -m apcir.preprocess.build_cast_topics --year 20 \
        --raw_topics   <2020_automatic_evaluation_topics_v1.0.json> \
        --manual_topics <2020_manual_evaluation_topics_v1.0.json> \
        --out          ../data/topics/cast20/cast_20_test.json
"""
import argparse
import json
import os


def load_manual_tsv(path):
    """2019: 'qid \t manually resolved utterance' per line."""
    m = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                m[parts[0].strip()] = parts[1].strip()
    return m


def load_manual_topics(path):
    """2020: manual rewrites live inside a parallel topics file."""
    m = {}
    for conv in json.load(open(path)):
        for t in conv["turn"]:
            m[f"{conv['number']}_{t['number']}"] = (t.get("manual_rewritten_utterance") or "").strip()
    return m


def build(raw_topics_path, manual_map, out_path):
    convs = json.load(open(raw_topics_path))
    out = []
    n_missing_oracle = 0
    for conv in convs:
        cid = str(conv["number"])
        history = []                      # previous raw utterances of THIS conversation
        for t in conv["turn"]:
            tid = f"{cid}_{t['number']}"
            raw = (t.get("raw_utterance") or "").strip()
            oracle = manual_map.get(tid, "")
            if not oracle:
                n_missing_oracle += 1
            reformulations = []
            auto = (t.get("automatic_rewritten_utterance") or "").strip()
            if auto:
                reformulations.append({
                    "reformulation_name": "cast_automatic_rewrite",
                    "reformulated_query": auto,
                    "ptkb_provenance": [],
                })
            out.append({
                "turn_id": tid,
                "conversation_id": cid,
                "title": conv.get("title", ""),
                "current_utterance": raw,
                "current_response": "",           # CAsT ships no gold response text for 19/20
                "response_provenance": [],        # required by Turn.from_dict
                "oracle_utterance": oracle,
                "context_utterances": list(history),
                "ptkb": {},                       # CAsT has no user profile
                "ptkb_provenance": [],
                "reformulations": reformulations,
                "results": [],
            })
            history.append(raw)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {len(out)} turns from {len(convs)} conversations -> {out_path}")
    if n_missing_oracle:
        print(f"WARNING: {n_missing_oracle} turns have no manual rewrite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", required=True, choices=["19", "20"])
    ap.add_argument("--raw_topics", required=True)
    ap.add_argument("--manual_tsv", default=None, help="2019 manual rewrites (.tsv)")
    ap.add_argument("--manual_topics", default=None, help="2020 manual rewrites (.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.year == "19":
        assert a.manual_tsv, "--manual_tsv is required for 2019"
        manual = load_manual_tsv(a.manual_tsv)
    else:
        assert a.manual_topics, "--manual_topics is required for 2020"
        manual = load_manual_topics(a.manual_topics)
    print(f"manual rewrites loaded: {len(manual)}")
    build(a.raw_topics, manual, a.out)


if __name__ == "__main__":
    main()
