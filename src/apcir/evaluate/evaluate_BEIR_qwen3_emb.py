import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from apcir.functional.encoders import BEIRQwenEncoder


BASE_PATH = "/data/rech/huiyuche/beir"
EMBEDDING_BASE_PATH = "/data/rech/huiyuche/beir/embeddings/qwen3_emb_0.6B"
HF_CACHE_DIR = "/data/rech/huiyuche/huggingface"
RESULT_BASE_DIR = "/data/rech/huiyuche/TREC_iKAT_2024/results/beir"
QWEN_REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
LIVE_MONITOR_PATH = Path("/data/rech/huiyuche/TREC_iKAT_2024/logs/beir_qwen_emb_0.6b_log.txt")

DEFAULT_DATASETS = [
    "msmarco",
    "scifact",
    "trec-covid",
    "nfcorpus",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "quora",
    "scidocs",
    "nq",
    "hotpotqa",
    "dbpedia-entity",
    "fever",
    "climate-fever",
    "cqadupstack",
]


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate BEIR datasets with Qwen3-Embedding-0.6B")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--worker_datasets", nargs="+", default=None)
    parser.add_argument("--worker_gpu_id", type=int, default=None)
    parser.add_argument("--worker_index", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--corpus_chunk_size", type=int, default=50000)
    parser.add_argument("--max_length_query", type=int, default=512)
    parser.add_argument("--max_length_doc", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--encode_only",
        action="store_true",
        help="Only encode and save embeddings. Skip faiss retrieval, ranking, and metric computation.",
    )
    parser.add_argument("--cache_dir", type=str, default=HF_CACHE_DIR)
    parser.add_argument("--embedding_base_path", type=str, default=EMBEDDING_BASE_PATH)
    parser.add_argument(
        "--query_instruction",
        type=str,
        default="Given a web search query, retrieve relevant passages that answer the query",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional, e.g. flash_attention_2 if your environment supports it.",
    )
    parser.add_argument(
        "--download_if_missing",
        action="store_true",
        help="Download Qwen3-Embedding-0.6B into cache_dir if the local snapshot is missing.",
    )
    return parser.parse_args()


def resolve_qwen_model_path(cache_dir: str, download_if_missing: bool = False) -> str:
    repo_dir = Path(cache_dir) / "models--Qwen--Qwen3-Embedding-0.6B"
    refs_main = repo_dir / "refs" / "main"
    snapshots_dir = repo_dir / "snapshots"

    if refs_main.exists():
        snapshot_name = refs_main.read_text().strip()
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return str(snapshot_path)

    if snapshots_dir.exists():
        snapshots = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if snapshots:
            return str(snapshots[-1])

    if download_if_missing:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=QWEN_REPO_ID, cache_dir=cache_dir)

    raise FileNotFoundError(
        f"Could not find a local snapshot for {QWEN_REPO_ID} under {cache_dir}. "
        "Download it first, or rerun with --download_if_missing."
    )


def split_datasets(datasets: List[str], num_workers: int) -> List[List[str]]:
    shards = [[] for _ in range(num_workers)]
    for index, dataset in enumerate(datasets):
        shards[index % num_workers].append(dataset)
    return [shard for shard in shards if shard]


def ensure_output_dirs(args) -> None:
    Path(args.embedding_base_path).mkdir(parents=True, exist_ok=True)
    Path(RESULT_BASE_DIR, "metrics").mkdir(parents=True, exist_ok=True)
    Path(RESULT_BASE_DIR, "ranking").mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if dataset == "cqadupstack":
            cqadupstack_dir = Path(BASE_PATH) / "cqadupstack" / "cqadupstack"
            if cqadupstack_dir.exists():
                for sub_dir in sorted([path for path in cqadupstack_dir.iterdir() if path.is_dir()]):
                    Path(args.embedding_base_path, "cqadupstack", sub_dir.name).mkdir(parents=True, exist_ok=True)
        else:
            Path(args.embedding_base_path, dataset).mkdir(parents=True, exist_ok=True)


def append_live_log(message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    LIVE_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp}] {message}\n"
    with open(LIVE_MONITOR_PATH, "a") as handle:
        handle.write(line)
        handle.flush()
    return str(LIVE_MONITOR_PATH)


def build_retriever(args, gpu_id: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    model_path = resolve_qwen_model_path(
        cache_dir=args.cache_dir,
        download_if_missing=args.download_if_missing,
    )

    encoder = BEIRQwenEncoder(
        model_path=model_path,
        cache_dir=args.cache_dir,
        device=device,
        max_length_query=args.max_length_query,
        max_length_doc=args.max_length_doc,
        query_instruction=args.query_instruction,
        attn_implementation=args.attn_implementation,
    )
    model = DRES(
        encoder,
        batch_size=args.batch_size,
        corpus_chunk_size=args.corpus_chunk_size,
    )
    return EvaluateRetrieval(model, score_function="dot")


def load_dataset(dataset_name: str):
    dataset_path = Path(BASE_PATH) / dataset_name / dataset_name
    split = "dev" if dataset_name == "msmarco" else "test"
    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)
    return corpus, queries, qrels, split


def save_ranking(dataset_name: str, ranking: Dict, gpu_id: int) -> str:
    ranking_path = Path(RESULT_BASE_DIR) / "ranking" / f"beir_qwen3_emb_{dataset_name}_gpu{gpu_id}.pkl"
    with open(ranking_path, "wb") as handle:
        pickle.dump(ranking, handle)
    return str(ranking_path)


def flatten_metrics(
    ndcg: Dict[str, float],
    _map: Dict[str, float],
    recall: Dict[str, float],
    precision: Dict[str, float],
    mrr: Dict[str, float],
) -> Dict[str, float]:
    return {
        **ndcg,
        **_map,
        **recall,
        **precision,
        **mrr,
    }


def evaluate_standard_dataset(dataset_name: str, retriever, args, gpu_id: int) -> Dict:
    corpus, queries, qrels, split = load_dataset(dataset_name)
    embedding_dir = Path(args.embedding_base_path) / dataset_name
    embedding_dir.mkdir(parents=True, exist_ok=True)
    append_live_log(
        f"START dataset={dataset_name} gpu={gpu_id} split={split} embedding_dir={embedding_dir}"
    )

    if args.encode_only:
        retriever.retriever.encode(
            corpus=corpus,
            queries=queries,
            encode_output_path=str(embedding_dir),
            overwrite=args.overwrite,
            query_filename="queries.pkl",
        )
        num_corpus = len(corpus)
        num_shards = (num_corpus + args.corpus_chunk_size - 1) // args.corpus_chunk_size
        summary = {
            "dataset": dataset_name,
            "split": split,
            "gpu_id": gpu_id,
            "num_queries": len(queries),
            "num_corpus": num_corpus,
            "num_shards": num_shards,
            "embedding_dir": str(embedding_dir),
            "ranking_path": None,
            "metrics": {},
            "encode_only": True,
        }
        append_live_log(
            f"ENCODE_DONE dataset={dataset_name} gpu={gpu_id} embedding_dir={embedding_dir} "
            f"num_queries={len(queries)} num_corpus={num_corpus} num_shards={num_shards}"
        )
        return summary

    results = retriever.encode_and_retrieve(
        corpus=corpus,
        queries=queries,
        encode_output_path=str(embedding_dir),
        overwrite=args.overwrite,
        query_filename="queries.pkl",
    )

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    ranking_path = save_ranking(dataset_name, results, gpu_id)
    summary = {
        "dataset": dataset_name,
        "split": split,
        "gpu_id": gpu_id,
        "num_queries": len(queries),
        "embedding_dir": str(embedding_dir),
        "ranking_path": ranking_path,
        "metrics": flatten_metrics(ndcg, _map, recall, precision, mrr),
        "encode_only": False,
    }
    append_live_log(
        f"DONE dataset={dataset_name} gpu={gpu_id} ranking_path={ranking_path} "
        f"NDCG@10={summary['metrics'].get('NDCG@10')} Recall@100={summary['metrics'].get('Recall@100')} "
        f"MRR@10={summary['metrics'].get('MRR@10')}"
    )
    return summary


def evaluate_cqadupstack(retriever, args, gpu_id: int) -> Dict:
    base_path = Path(BASE_PATH) / "cqadupstack" / "cqadupstack"
    split = "test"
    append_live_log(f"START dataset=cqadupstack gpu={gpu_id} split={split}")

    query_num_metric_dict = {}
    ranking_by_subdataset = {}

    sub_datasets = sorted([path for path in base_path.iterdir() if path.is_dir()])
    for sub_data_path in sub_datasets:
        embedding_dir = Path(args.embedding_base_path) / "cqadupstack" / sub_data_path.name
        embedding_dir.mkdir(parents=True, exist_ok=True)

        corpus, queries, qrels = GenericDataLoader(str(sub_data_path)).load(split=split)
        append_live_log(
            f"START dataset=cqadupstack/{sub_data_path.name} gpu={gpu_id} embedding_dir={embedding_dir}"
        )
        if args.encode_only:
            retriever.retriever.encode(
                corpus=corpus,
                queries=queries,
                encode_output_path=str(embedding_dir),
                overwrite=args.overwrite,
                query_filename="queries.pkl",
            )
            num_corpus = len(corpus)
            num_shards = (num_corpus + args.corpus_chunk_size - 1) // args.corpus_chunk_size
            query_num_metric_dict[sub_data_path.name] = {
                "num_queries": len(queries),
                "num_corpus": num_corpus,
                "num_shards": num_shards,
                "metrics": {},
                "embedding_dir": str(embedding_dir),
                "encode_only": True,
            }
            append_live_log(
                f"ENCODE_DONE dataset=cqadupstack/{sub_data_path.name} gpu={gpu_id} "
                f"num_queries={len(queries)} num_corpus={num_corpus} num_shards={num_shards}"
            )
            continue

        results = retriever.encode_and_retrieve(
            corpus=corpus,
            queries=queries,
            encode_output_path=str(embedding_dir),
            overwrite=args.overwrite,
            query_filename="queries.pkl",
        )

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

        query_num_metric_dict[sub_data_path.name] = {
            "num_queries": len(queries),
            "metrics": flatten_metrics(ndcg, _map, recall, precision, mrr),
            "embedding_dir": str(embedding_dir),
            "encode_only": False,
        }
        ranking_by_subdataset[sub_data_path.name] = results
        append_live_log(
            f"DONE dataset=cqadupstack/{sub_data_path.name} gpu={gpu_id} "
            f"NDCG@10={query_num_metric_dict[sub_data_path.name]['metrics'].get('NDCG@10')} "
            f"Recall@100={query_num_metric_dict[sub_data_path.name]['metrics'].get('Recall@100')}"
        )

    if args.encode_only:
        total_query_number = sum(info["num_queries"] for info in query_num_metric_dict.values())
        total_corpus_number = sum(info["num_corpus"] for info in query_num_metric_dict.values())
        total_shards = sum(info["num_shards"] for info in query_num_metric_dict.values())
        summary = {
            "dataset": "cqadupstack",
            "split": split,
            "gpu_id": gpu_id,
            "num_queries": total_query_number,
            "num_corpus": total_corpus_number,
            "num_shards": total_shards,
            "ranking_path": None,
            "metrics": {},
            "subdatasets": query_num_metric_dict,
            "encode_only": True,
        }
        append_live_log(
            f"ENCODE_DONE dataset=cqadupstack gpu={gpu_id} num_queries={total_query_number} "
            f"num_corpus={total_corpus_number} num_shards={total_shards}"
        )
        return summary

    total_query_number = sum(info["num_queries"] for info in query_num_metric_dict.values())
    all_metric_keys = sorted(
        {
            metric_key
            for info in query_num_metric_dict.values()
            for metric_key in info["metrics"].keys()
        }
    )

    weighted_metrics = {}
    for metric_key in all_metric_keys:
        weighted_sum = sum(
            info["metrics"].get(metric_key, 0.0) * info["num_queries"]
            for info in query_num_metric_dict.values()
        )
        weighted_metrics[metric_key] = round(weighted_sum / total_query_number, 5)

    ranking_path = save_ranking("cqadupstack", ranking_by_subdataset, gpu_id)
    summary = {
        "dataset": "cqadupstack",
        "split": split,
        "gpu_id": gpu_id,
        "num_queries": total_query_number,
        "ranking_path": ranking_path,
        "metrics": weighted_metrics,
        "subdatasets": query_num_metric_dict,
        "encode_only": False,
    }
    append_live_log(
        f"DONE dataset=cqadupstack gpu={gpu_id} ranking_path={ranking_path} "
        f"NDCG@10={weighted_metrics.get('NDCG@10')} Recall@100={weighted_metrics.get('Recall@100')}"
    )
    return summary


def evaluate_dataset(dataset_name: str, retriever, args, gpu_id: int) -> Dict:
    if dataset_name == "cqadupstack":
        return evaluate_cqadupstack(retriever=retriever, args=args, gpu_id=gpu_id)
    return evaluate_standard_dataset(dataset_name=dataset_name, retriever=retriever, args=args, gpu_id=gpu_id)


def worker_main(worker_index: int, gpu_id: int, datasets: List[str], args) -> Dict:
    try:
        append_live_log(
            f"WORKER_START worker={worker_index} gpu={gpu_id} pid={os.getpid()} datasets={datasets}"
        )
        retriever = build_retriever(args=args, gpu_id=0)
        summaries = []
        for dataset_name in datasets:
            print(f"[worker {worker_index}] evaluating {dataset_name} on GPU {gpu_id}")
            summaries.append(evaluate_dataset(dataset_name, retriever, args, gpu_id))
        append_live_log(
            f"WORKER_DONE worker={worker_index} gpu={gpu_id} completed_datasets={datasets}"
        )
        payload = {
            "worker_index": worker_index,
            "gpu_id": gpu_id,
            "datasets": datasets,
            "summaries": summaries,
        }
        summary_path = Path(RESULT_BASE_DIR) / "metrics" / f"beir_qwen3_emb_worker_{worker_index}.json"
        with open(summary_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        return payload
    except Exception as exc:
        append_live_log(
            f"WORKER_ERROR worker={worker_index} gpu={gpu_id} datasets={datasets} error={repr(exc)}"
        )
        raise


def run_single_worker(args) -> List[Dict]:
    if args.worker_gpu_id is None or args.worker_index is None:
        raise ValueError("worker_gpu_id and worker_index must be provided in single-worker mode.")

    payload = worker_main(
        worker_index=args.worker_index,
        gpu_id=args.worker_gpu_id,
        datasets=args.worker_datasets or [],
        args=args,
    )
    return [payload]


def run_parallel_via_subprocess(args) -> List[Dict]:
    datasets = list(dict.fromkeys(args.datasets))
    if not datasets:
        raise ValueError("No datasets were provided.")

    ensure_output_dirs(args)
    worker_datasets = split_datasets(datasets, len(args.gpu_ids))

    processes = []
    for worker_index, datasets_for_worker in enumerate(worker_datasets):
        gpu_id = args.gpu_ids[worker_index]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        env.setdefault("USE_TF", "0")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "apcir.evaluate.evaluate_BEIR_qwen3_emb",
            "--worker_gpu_id",
            str(gpu_id),
            "--worker_index",
            str(worker_index),
            "--batch_size",
            str(args.batch_size),
            "--corpus_chunk_size",
            str(args.corpus_chunk_size),
            "--max_length_query",
            str(args.max_length_query),
            "--max_length_doc",
            str(args.max_length_doc),
            "--cache_dir",
            args.cache_dir,
            "--embedding_base_path",
            args.embedding_base_path,
            "--query_instruction",
            args.query_instruction,
        ]

        if args.attn_implementation is not None:
            cmd.extend(["--attn_implementation", args.attn_implementation])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.download_if_missing:
            cmd.append("--download_if_missing")
        if args.encode_only:
            cmd.append("--encode_only")

        cmd.append("--worker_datasets")
        cmd.extend(datasets_for_worker)

        process = subprocess.Popen(
            cmd,
            cwd="/data/rech/huiyuche/TREC_iKAT_2024/src",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_index, gpu_id, datasets_for_worker, process))

    worker_payloads = []
    for worker_index, gpu_id, datasets_for_worker, process in processes:
        stdout, _ = process.communicate()
        if stdout:
            append_live_log(
                f"SUBPROCESS_STDOUT worker={worker_index} gpu={gpu_id} output={stdout[-1500:]}"
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"Worker subprocess {worker_index} on GPU {gpu_id} failed with exit code {process.returncode}."
            )

        summary_path = Path(RESULT_BASE_DIR) / "metrics" / f"beir_qwen3_emb_worker_{worker_index}.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing worker summary file: {summary_path}")
        with open(summary_path, "r") as handle:
            worker_payloads.append(json.load(handle))

    return worker_payloads


def write_metrics_report(args, worker_payloads: List[Dict]) -> str:
    metrics_path = Path(RESULT_BASE_DIR) / "metrics" / "beir_qwen3_emb_metrics.txt"
    with open(metrics_path, "w") as handle:
        handle.write(f"model: {QWEN_REPO_ID}\n")
        handle.write(f"embedding_base_path: {args.embedding_base_path}\n")
        handle.write(f"gpu_ids: {args.gpu_ids}\n")
        handle.write(f"batch_size: {args.batch_size}\n")
        handle.write(f"encode_only: {args.encode_only}\n")
        handle.write(f"max_length_query: {args.max_length_query}\n")
        handle.write(f"max_length_doc: {args.max_length_doc}\n")
        handle.write("\n")

        for payload in sorted(worker_payloads, key=lambda item: item["worker_index"]):
            handle.write(
                f"worker_index={payload['worker_index']} gpu_id={payload['gpu_id']} datasets={payload['datasets']}\n"
            )
            for summary in payload["summaries"]:
                handle.write(f"Results for dataset: {summary['dataset']}\n")
                handle.write(f"split: {summary['split']}\n")
                handle.write(f"num_queries: {summary['num_queries']}\n")
                if "num_corpus" in summary:
                    handle.write(f"num_corpus: {summary['num_corpus']}\n")
                if "num_shards" in summary:
                    handle.write(f"num_shards: {summary['num_shards']}\n")
                handle.write(f"ranking_path: {summary.get('ranking_path')}\n")
                if "embedding_dir" in summary:
                    handle.write(f"embedding_dir: {summary['embedding_dir']}\n")
                if summary.get("encode_only"):
                    handle.write("mode: encode_only\n")
                for metric_name, metric_value in summary["metrics"].items():
                    handle.write(f"{metric_name}: {metric_value}\n")
                handle.write("\n")
    return str(metrics_path)


def write_json_report(worker_payloads: List[Dict]) -> str:
    json_path = Path(RESULT_BASE_DIR) / "metrics" / "beir_qwen3_emb_summary.json"
    with open(json_path, "w") as handle:
        json.dump(worker_payloads, handle, indent=2)
    return str(json_path)


def run_parallel(args) -> List[Dict]:
    datasets = list(dict.fromkeys(args.datasets))
    if not datasets:
        raise ValueError("No datasets were provided.")

    ensure_output_dirs(args)

    if not torch.cuda.is_available() or len(args.gpu_ids) == 1:
        gpu_id = args.gpu_ids[0] if args.gpu_ids else 0
        payload = worker_main(worker_index=0, gpu_id=gpu_id, datasets=datasets, args=args)
        return [payload]
    return run_parallel_via_subprocess(args)


def main():
    args = get_args()
    LIVE_MONITOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_top_level = args.worker_gpu_id is None
    if is_top_level:
        with open(LIVE_MONITOR_PATH, "w") as handle:
            handle.write("")
        append_live_log(
            f"RUN_START datasets={args.datasets} gpu_ids={args.gpu_ids} batch_size={args.batch_size} "
            f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}"
        )

    if args.worker_gpu_id is not None:
        worker_payloads = run_single_worker(args)
    else:
        worker_payloads = run_parallel(args)

    metrics_path = write_metrics_report(args, worker_payloads)
    json_path = write_json_report(worker_payloads)
    if is_top_level:
        append_live_log(f"RUN_DONE metrics_path={metrics_path} json_path={json_path}")
    print(f"Saved metrics report to {metrics_path}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Live monitor path: {LIVE_MONITOR_PATH}")


if __name__ == "__main__":
    main()
