import argparse
import json
import os
from collections import Counter

from tqdm import tqdm


def mkdir_for_file(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_passage_text(text):
    # Keep one document per TSV line. The official passage text can contain
    # tabs/newlines, so normalize whitespace before writing.
    return " ".join(str(text).replace("\t", " ").split())


def iter_qrecc_passages(collection_dir):
    """
    Yield official QReCC paragraph passages.

    The Zenodo README says passages.zip contains the corpus after applying
    ml-qrecc/collection/paragraph_chunker.py. The extracted layout is expected
    to contain commoncrawl, wayback, and wayback-backfill directories, each with
    JSONL files whose records have "id" and "contents".
    """
    subdirs = ["commoncrawl", "wayback", "wayback-backfill"]
    for subdir in subdirs:
        cur_dir = os.path.join(collection_dir, subdir)
        if not os.path.isdir(cur_dir):
            print("Warning: skip missing passage directory: {}".format(cur_dir))
            continue

        filenames = sorted(os.listdir(cur_dir))
        for filename in filenames:
            path = os.path.join(cur_dir, filename)
            if not os.path.isfile(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    yield str(record["id"]), clean_passage_text(record.get("contents", ""))


def write_collection_tsv(collection_dir, output_tsv, max_docs=None):
    mkdir_for_file(output_tsv)

    num_docs = 0
    empty_docs = 0
    with open(output_tsv, "w", encoding="utf-8") as fw:
        for doc_id, text in tqdm(iter_qrecc_passages(collection_dir), desc="write qrecc collection tsv"):
            fw.write("{}\t{}\n".format(doc_id, text))
            num_docs += 1
            if len(text) == 0:
                empty_docs += 1
            if max_docs is not None and num_docs >= max_docs:
                break

    print("QReCC collection TSV written to {}".format(output_tsv))
    print("collection docs = {}, empty docs = {}".format(num_docs, empty_docs))
    return {"collection_docs": num_docs, "collection_empty_docs": empty_docs}


def make_qid(record):
    return "{}-{}".format(record["Conversation_no"], record["Turn_no"])


def convert_train(train_records, output_train_jsonl):
    """
    Write QReCC train turns in a conversation schema close to TopiOCQA.

    Official qrecc-training.json has rewrites and answers but no passage ids.
    This file is therefore useful for query rewriting / conversational query
    training, but not directly for supervised dense retrieval.
    """
    mkdir_for_file(output_train_jsonl)
    with open(output_train_jsonl, "w", encoding="utf-8") as fw:
        for record in tqdm(train_records, desc="write qrecc train jsonl"):
            out = {
                "sample_id": make_qid(record),
                "Conversation_no": record["Conversation_no"],
                "Turn_no": record["Turn_no"],
                "Conversation_source": record.get("Conversation_source", ""),
                "Context": record.get("Context", []),
                "Question": record["Question"],
                "Answer": record.get("Answer", ""),
                "Truth_rewrite": record.get("Rewrite", ""),
                "Answer_URL": record.get("Answer_URL", ""),
            }
            fw.write(json.dumps(out, ensure_ascii=False) + "\n")


def convert_valid_and_qrels(test_records, ground_truth_records, output_valid_jsonl, output_qrel):
    """
    Write QReCC test turns for eval_topiocqa-compatible evaluation.

    QIDs are Conversation_no-Turn_no, matching the TopiOCQA qrel convention in
    this repo. Qrels use official raw passage ids from Truth_passages.
    """
    if len(test_records) != len(ground_truth_records):
        raise ValueError(
            "test and ground-truth sizes differ: {} vs {}".format(
                len(test_records), len(ground_truth_records)
            )
        )

    mkdir_for_file(output_valid_jsonl)
    mkdir_for_file(output_qrel)

    qrel_rows = 0
    qids_with_qrels = set()
    passage_per_query = []

    with open(output_valid_jsonl, "w", encoding="utf-8") as valid_fw, \
            open(output_qrel, "w", encoding="utf-8") as qrel_fw:
        for record, gt_record in tqdm(
            zip(test_records, ground_truth_records),
            total=len(test_records),
            desc="write qrecc valid/qrel",
        ):
            qid = make_qid(record)
            out = {
                "sample_id": qid,
                "Conversation_no": record["Conversation_no"],
                "Turn_no": record["Turn_no"],
                "Conversation_source": record.get("Conversation_source", ""),
                "Context": record.get("Context", []),
                "Question": record["Question"],
                "Answer": gt_record.get("Truth_answer", record.get("Answer", "")),
                "Truth_rewrite": gt_record.get("Truth_rewrite", record.get("Rewrite", "")),
                "Answer_URL": record.get("Answer_URL", ""),
            }
            valid_fw.write(json.dumps(out, ensure_ascii=False) + "\n")

            truth_passages = gt_record.get("Truth_passages", [])
            passage_per_query.append(len(truth_passages))
            for passage_id in truth_passages:
                qrel_fw.write("{} 0 {} 1\n".format(qid, passage_id))
                qrel_rows += 1
                qids_with_qrels.add(qid)

    return {
        "valid_turns": len(test_records),
        "qrel_rows": qrel_rows,
        "qrel_queries": len(qids_with_qrels),
        "qrel_empty_queries": len(test_records) - len(qids_with_qrels),
        "qrel_unique_positive_doc_ids": len({
            passage_id
            for record in ground_truth_records
            for passage_id in record.get("Truth_passages", [])
        }),
        "avg_positive_docs_all_valid_queries": (
            float(sum(passage_per_query)) / max(1, len(passage_per_query))
        ),
        "avg_positive_docs_judged_queries": (
            float(sum(passage_per_query)) / max(1, len(qids_with_qrels))
        ),
        "max_positive_docs_per_query": max(passage_per_query) if passage_per_query else 0,
    }


def collect_split_stats(train_records, test_records):
    train_sources = Counter(record.get("Conversation_source", "") for record in train_records)
    test_sources = Counter(record.get("Conversation_source", "") for record in test_records)
    return {
        "train_turns": len(train_records),
        "train_conversations": len({record["Conversation_no"] for record in train_records}),
        "train_sources": dict(train_sources),
        "test_turns": len(test_records),
        "test_conversations": len({record["Conversation_no"] for record in test_records}),
        "test_sources": dict(test_sources),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrecc_train_json", type=str, required=True)
    parser.add_argument("--qrecc_test_json", type=str, required=True)
    parser.add_argument("--qrecc_ground_truth_json", type=str, required=True)

    parser.add_argument("--output_train_jsonl", type=str, required=True)
    parser.add_argument("--output_valid_jsonl", type=str, required=True)
    parser.add_argument("--output_qrel", type=str, required=True)
    parser.add_argument("--output_stats_json", type=str, required=True)

    parser.add_argument("--collection_dir", type=str, default=None,
                        help="Extracted QReCC passages.zip directory with commoncrawl/wayback subdirs.")
    parser.add_argument("--output_collection_tsv", type=str, default=None,
                        help="Dense indexing TSV output. Keeps official raw passage IDs.")
    parser.add_argument("--max_collection_docs", type=int, default=None,
                        help="Debug option: stop after this many corpus docs.")

    args = parser.parse_args()

    train_records = load_json(args.qrecc_train_json)
    test_records = load_json(args.qrecc_test_json)
    ground_truth_records = load_json(args.qrecc_ground_truth_json)

    stats = collect_split_stats(train_records, test_records)

    convert_train(train_records, args.output_train_jsonl)
    stats.update(
        convert_valid_and_qrels(
            test_records,
            ground_truth_records,
            args.output_valid_jsonl,
            args.output_qrel,
        )
    )

    if args.collection_dir is not None or args.output_collection_tsv is not None:
        if args.collection_dir is None or args.output_collection_tsv is None:
            raise ValueError("--collection_dir and --output_collection_tsv must be used together.")
        stats.update(
            write_collection_tsv(
                args.collection_dir,
                args.output_collection_tsv,
                max_docs=args.max_collection_docs,
            )
        )

    mkdir_for_file(args.output_stats_json)
    with open(args.output_stats_json, "w", encoding="utf-8") as fw:
        json.dump(stats, fw, indent=2, ensure_ascii=False)

    print("QReCC preprocessing done.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
