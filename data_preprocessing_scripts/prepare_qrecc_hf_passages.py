import argparse
import json
import os
import subprocess

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from tqdm import tqdm


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def clean_text(text):
    return " ".join(str(text).replace("\t", " ").split())


def list_parquet_files(repo_id, repo_type):
    api = HfApi()
    filenames = api.list_repo_files(repo_id, repo_type=repo_type)
    parquet_files = [name for name in filenames if name.endswith(".parquet")]
    return sorted(parquet_files)


def download_file(repo_id, repo_type, filename, parquet_dir):
    mkdir(parquet_dir)
    local_name = filename.replace("/", "__")
    output_path = os.path.join(parquet_dir, local_name)
    url = hf_hub_url(repo_id, filename, repo_type=repo_type)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("Reuse parquet shard: {}".format(output_path), flush=True)
        return output_path

    print("Download {} -> {}".format(url, output_path), flush=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "10",
        "-C",
        "-",
        "-o",
        output_path,
        url,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def convert_parquet_to_jsonl(parquet_path, jsonl_path, batch_size, max_docs=None, tsv_fw=None):
    done_path = jsonl_path + ".done"
    if os.path.exists(done_path):
        print("Skip converted shard: {}".format(jsonl_path), flush=True)
        return 0, True

    tmp_path = jsonl_path + ".tmp"
    num_docs = 0
    empty_docs = 0
    mkdir(os.path.dirname(jsonl_path))

    parquet_file = pq.ParquetFile(parquet_path)
    with open(tmp_path, "w", encoding="utf-8") as jsonl_fw:
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["id", "contents"]):
            ids = batch.column("id").to_pylist()
            contents = batch.column("contents").to_pylist()
            for doc_id, text in zip(ids, contents):
                text = clean_text(text)
                record = {"id": str(doc_id), "contents": text}
                jsonl_fw.write(json.dumps(record, ensure_ascii=False) + "\n")
                if tsv_fw is not None:
                    tsv_fw.write("{}\t{}\n".format(record["id"], text))
                num_docs += 1
                if len(text) == 0:
                    empty_docs += 1
                if max_docs is not None and num_docs >= max_docs:
                    break
            if max_docs is not None and num_docs >= max_docs:
                break

    os.replace(tmp_path, jsonl_path)
    with open(done_path, "w", encoding="utf-8") as fw:
        fw.write("docs={}\nempty_docs={}\n".format(num_docs, empty_docs))

    print(
        "Converted {} docs (empty={}) -> {}".format(num_docs, empty_docs, jsonl_path),
        flush=True,
    )
    return num_docs, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="slupart/qrecc-passages")
    parser.add_argument("--repo_type", type=str, default="dataset")
    parser.add_argument("--parquet_dir", type=str, required=True)
    parser.add_argument("--output_jsonl_dir", type=str, required=True)
    parser.add_argument("--output_tsv", type=str, default=None,
                        help="Optional dense-indexing TSV. For resume safety, use only for one-shot runs.")
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None,
                        help="Debug option: cap docs converted per shard.")
    args = parser.parse_args()

    mkdir(args.parquet_dir)
    mkdir(args.output_jsonl_dir)
    if args.output_tsv is not None:
        mkdir(os.path.dirname(args.output_tsv))

    parquet_files = list_parquet_files(args.repo_id, args.repo_type)
    if args.max_files is not None:
        parquet_files = parquet_files[:args.max_files]

    print("repo_id = {}".format(args.repo_id), flush=True)
    print("num parquet shards to process = {}".format(len(parquet_files)), flush=True)
    print("output_jsonl_dir = {}".format(args.output_jsonl_dir), flush=True)

    total_docs = 0
    tsv_fw = open(args.output_tsv, "w", encoding="utf-8") if args.output_tsv is not None else None
    try:
        for filename in tqdm(parquet_files, desc="qrecc hf parquet shards"):
            parquet_path = download_file(args.repo_id, args.repo_type, filename, args.parquet_dir)
            shard_name = os.path.splitext(os.path.basename(filename))[0] + ".jsonl"
            jsonl_path = os.path.join(args.output_jsonl_dir, shard_name)
            num_docs, skipped = convert_parquet_to_jsonl(
                parquet_path,
                jsonl_path,
                batch_size=args.batch_size,
                max_docs=args.max_docs,
                tsv_fw=tsv_fw,
            )
            total_docs += num_docs
    finally:
        if tsv_fw is not None:
            tsv_fw.close()

    print("QReCC HF passage preparation done. Newly converted docs = {}".format(total_docs), flush=True)


if __name__ == "__main__":
    main()
