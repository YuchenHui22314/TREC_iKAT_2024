"""Convert official TREC CAsT topic files into the pipeline's Turn JSON schema.

All four CAsT years are handled here. NOTE they do NOT share a collection:
  2019/2020 = MS MARCO v1 passages + TREC CAR
  2021      = MS MARCO v1 DOCUMENTS + Washington Post + KILT Wikipedia
  2022      = MS MARCO v2 + Washington Post + KILT, with docids carrying a passage suffix
              (e.g. MARCO_02_1687136851-3), i.e. a different collection from 2021.

The output is the SAME schema `load_turns_from_json` already reads for iKAT,
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

    # 2021 ships raw + manual + automatic in ONE file, so pass it as both
    python -m apcir.preprocess.build_cast_topics --year 21 \
        --raw_topics    <2021_manual_evaluation_topics_v1.0.json> \
        --manual_topics <2021_manual_evaluation_topics_v1.0.json> \
        --out           ../data/topics/cast21/cast_21_test.json

    python -m apcir.preprocess.build_cast_topics --year 22 \
        --raw_topics  <2022_evaluation_topics_flattened_duplicated_v1.0.json> \
        --auto_topics <2022_automatic_evaluation_topics_flattened_duplicated_v1.0.json> \
        --out         ../data/topics/cast22/cast_22_test.json
"""
import argparse
import ast
import json
import os


def load_manual_tsv(path):
    """2019: 'qid \t manually resolved utterance' per line."""
    m = {}
    with open(path) as f:
        for line in f:
            # split ONCE: a manual rewrite containing a tab would otherwise be truncated
            # at the first tab (only parts[1] was kept).
            parts = line.rstrip("\n").split("\t", 1)
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


def build_2022(eval_topics_path, auto_topics_path, out_path):
    """CAsT 2022 uses a different schema from 2019-2021.

    Topics are TREES: a turn number looks like "{branch}-{turn}" and the released
    `*_flattened_duplicated_*` file enumerates the 50 root-to-leaf PATHS, so turns on a
    shared prefix appear once per path (284 entries for 205 distinct turns). Because every
    node has exactly one parent, a given turn_id always carries the same ancestor chain,
    so de-duplicating by turn_id and keeping the first occurrence is safe.

    Field names also differ: `utterance` (not `raw_utterance`), and the system `response`
    text is shipped inline (2019-2021 only give a canonical result id).
    """
    convs = json.load(open(eval_topics_path))
    auto = {}
    for c in json.load(open(auto_topics_path)):
        for t in c["turn"]:
            a = (t.get("automatic_rewritten_utterance") or "").strip()
            if a:
                auto[f"{c['number']}_{t['number']}"] = a

    out, seen = [], set()
    n_missing_oracle = 0
    for conv in convs:
        cid = str(conv["number"])
        history = []
        for t in conv["turn"]:
            tid = f"{cid}_{t['number']}"
            raw = (t.get("utterance") or "").strip()
            if tid not in seen:
                seen.add(tid)
                oracle = (t.get("manual_rewritten_utterance") or "").strip()
                if not oracle:
                    n_missing_oracle += 1
                reformulations = []
                if tid in auto:
                    reformulations.append({
                        "reformulation_name": "cast_automatic_rewrite",
                        "reformulated_query": auto[tid],
                        "ptkb_provenance": [],
                    })
                prov = t.get("provenance") or []
                if isinstance(prov, str):          # released as a str(list)
                    try:
                        prov = ast.literal_eval(prov)
                    except (ValueError, SyntaxError):
                        prov = [prov]
                out.append({
                    "turn_id": tid,
                    "conversation_id": cid,
                    "title": conv.get("title", ""),
                    "current_utterance": raw,
                    "current_response": (t.get("response") or "").strip(),
                    "response_provenance": list(prov),
                    "oracle_utterance": oracle,
                    "context_utterances": list(history),
                    "ptkb": {},
                    "ptkb_provenance": [],
                    "reformulations": reformulations,
                    "results": [],
                })
            history.append(raw)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"wrote {len(out)} distinct turns from {len(convs)} paths -> {out_path}")
    if n_missing_oracle:
        print(f"WARNING: {n_missing_oracle} turns have no manual rewrite")


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
                # 2021 ships the canonical response TEXT inline (`passage`); 2020/2021 both
                # ship the id of the canonical result (`canonical_result_id`, or 2020's
                # `manual_canonical_result_id`). 2019 has neither. Previously both were
                # dropped, which made the conversion lossy for anything needing prior
                # responses (generation, full-conversation query building).
                "current_response": (t.get("passage") or "").strip(),
                "response_provenance": [pid for pid in [
                    t.get("canonical_result_id"),
                    t.get("manual_canonical_result_id"),
                    t.get("automatic_canonical_result_id"),
                ] if pid],
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
    ap.add_argument("--year", required=True, choices=["19", "20", "21", "22"])
    ap.add_argument("--raw_topics", required=True)
    ap.add_argument("--manual_tsv", default=None, help="2019 manual rewrites (.tsv)")
    ap.add_argument("--manual_topics", default=None, help="2020 manual rewrites (.json)")
    ap.add_argument("--auto_topics", default=None, help="2022 automatic rewrites (.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.year == "22":
        assert a.auto_topics, "--auto_topics is required for 2022"
        build_2022(a.raw_topics, a.auto_topics, a.out)
        return

    if a.year == "19":
        assert a.manual_tsv, "--manual_tsv is required for 2019"
        manual = load_manual_tsv(a.manual_tsv)
    else:                                   # 2020 and 2021 share the same schema
        assert a.manual_topics, f"--manual_topics is required for 20{a.year}"
        manual = load_manual_topics(a.manual_topics)
    print(f"manual rewrites loaded: {len(manual)}")
    build(a.raw_topics, manual, a.out)


if __name__ == "__main__":
    main()
