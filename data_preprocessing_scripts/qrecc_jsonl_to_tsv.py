import argparse
import glob
import json
import os
import time

from tqdm import tqdm


def mkdir_for_file(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def clean_text(text):
    return " ".join(str(text).replace("\t", " ").split())


def iter_jsonl_files(input_jsonl_dir, max_files=None):
    paths = sorted(glob.glob(os.path.join(input_jsonl_dir, "*.jsonl")))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl_dir", type=str, required=True)
    parser.add_argument("--output_tsv", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    mkdir_for_file(args.output_tsv)
    tmp_path = args.output_tsv + ".tmp"
    done_path = args.output_tsv + ".done"
    lock_path = args.output_tsv + ".lock"

    if os.path.exists(done_path):
        print("Skip TSV conversion because done marker exists: {}".format(done_path), flush=True)
        return

    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(lock_fd, "w", encoding="utf-8") as fw:
                fw.write("pid={}\n".format(os.getpid()))
            break
        except FileExistsError:
            print("TSV lock exists, waiting for another process: {}".format(lock_path), flush=True)
            for _ in range(60):
                if os.path.exists(done_path):
                    print("Detected done marker while waiting: {}".format(done_path), flush=True)
                    return
                time.sleep(5)

    try:
        jsonl_files = iter_jsonl_files(args.input_jsonl_dir, max_files=args.max_files)
        print("input_jsonl_dir = {}".format(args.input_jsonl_dir), flush=True)
        print("num jsonl files = {}".format(len(jsonl_files)), flush=True)
        print("output_tsv = {}".format(args.output_tsv), flush=True)

        num_docs = 0
        empty_docs = 0
        with open(tmp_path, "w", encoding="utf-8") as fw:
            for path in tqdm(jsonl_files, desc="qrecc jsonl -> tsv"):
                with open(path, "r", encoding="utf-8") as fr:
                    for line in fr:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        doc_id = str(record["id"])
                        text = clean_text(record.get("contents", ""))
                        fw.write("{}\t{}\n".format(doc_id, text))
                        num_docs += 1
                        if len(text) == 0:
                            empty_docs += 1
                        if args.max_docs is not None and num_docs >= args.max_docs:
                            break
                if args.max_docs is not None and num_docs >= args.max_docs:
                    break

        os.replace(tmp_path, args.output_tsv)
        with open(done_path, "w", encoding="utf-8") as fw:
            fw.write("docs={}\nempty_docs={}\n".format(num_docs, empty_docs))

        print("QReCC TSV written to {}".format(args.output_tsv), flush=True)
        print("docs = {}, empty_docs = {}".format(num_docs, empty_docs), flush=True)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


if __name__ == "__main__":
    main()
