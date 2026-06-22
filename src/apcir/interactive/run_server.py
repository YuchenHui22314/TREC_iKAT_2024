"""CLI: launch the interactive search server (uvicorn, 1 worker).

    python -m apcir.interactive.run_server \
        --retrievers BM25 ance \
        --retrieval_query_types raw full_conversation_dense \
        --fusion_type RRF --host 127.0.0.1 --port 8000

Defaults target octal40 /part/01 (fast SSD) ANCE + the official BM25 sparse index.
Run under tmux (long-running, big resident state). Use the trec_ikat py3.12 env.
"""

from __future__ import annotations

import argparse

import uvicorn

from .pipeline import PipelineConfig, RetrieverSpec
from .search_server import create_app


def build_parser() -> argparse.ArgumentParser:
    c = PipelineConfig()
    p = argparse.ArgumentParser(description="apcir interactive search server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    # retrievers (parallel lists)
    p.add_argument("--retrievers", nargs="+", default=["BM25", "ance"])
    p.add_argument("--retrieval_query_types", nargs="+",
                   default=["raw", "full_conversation_dense"])
    # online QR per retriever leg (parallel to --retrievers; "" or "none" = no QR).
    # e.g. --qr rar none  -> BM25 uses the rar rewrite, ANCE uses full_conversation_dense.
    p.add_argument("--qr", nargs="+", default=None,
                   help="QR name per retriever leg (rar / rar_personalized_cot1 / "
                        "rar_non_personalized_cot1 / MQ4CS_persq / GtR / ptkb_sum / none)")
    p.add_argument("--dense_encoder_paths", nargs="+", default=None,
                   help="per-leg DENSE query-encoder ckpt, parallel to --retrievers; token '-' = "
                        "use the global --dense_query_encoder_path. Lets two dense legs (e.g. "
                        "conv-qwen3 + pers-conv-qwen3) use different encoders on the same index.")
    # fusion
    p.add_argument("--fusion_type", default="RRF",
                   choices=["RRF", "linear_combination", "round_robin", "concat"])
    p.add_argument("--fusion_normalization", default="min-max")
    p.add_argument("--fuse_weights", nargs="*", type=float, default=None)
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--retrieval_top_k", type=int, default=c.retrieval_top_k)
    # reranking (between fusion and generation; co-hosted on the gen-LLM GPU)
    p.add_argument("--reranker", default=c.reranker, choices=["none", "qwen3_reranker", "rankllama"])
    p.add_argument("--rerank_top_k", type=int, default=c.rerank_top_k)
    p.add_argument("--rerank_batch_size", type=int, default=c.rerank_batch_size)
    p.add_argument("--rerank_quant", default=c.rerank_quant, choices=["none", "8b", "4b"])
    p.add_argument("--rerank_remote_url", default=None,
                   help="e.g. http://octal31:8200 — score via a remote rerank_server.py "
                        "(zero local VRAM) instead of co-hosting the model")
    p.add_argument("--qwen3_reranker_path", default=c.qwen3_reranker_path)
    p.add_argument("--reranking_query_type", default=c.reranking_query_type,
                   help="qwen_3_rerank_instruct_full (conversational instruction + "
                        "profile-first query) OR an online-QR reformulation name like "
                        "MQ4CS_persq_rw (native instruction + that rewrite)")
    # generation
    p.add_argument("--generation", default=c.generation, choices=["rag", "extractive"])
    p.add_argument("--response_max_tokens", type=int, default=c.response_max_tokens)
    p.add_argument("--citations_max", type=int, default=c.citations_max)
    p.add_argument("--generation_top_k", type=int, default=c.generation_top_k)
    # shared LLM (QR + RAG generation)
    p.add_argument("--llm_backend", default=c.llm_backend, choices=["local_vllm", "openai"])
    p.add_argument("--llm_model", default=None,
                   help="LLM model id. If unset, defaults by backend: local_vllm->qwen3-32b, "
                        "openai->gpt-5-mini.")
    p.add_argument("--llm_base_url", default=c.llm_base_url)
    p.add_argument("--llm_gpu_id", type=int, default=c.llm_gpu_id)
    p.add_argument("--llm_max_tokens", type=int, default=c.llm_max_tokens)
    p.add_argument("--gtr_phi", type=int, default=c.gtr_phi)
    p.add_argument("--demo_file", default=c.demo_file)
    p.add_argument("--personalized_demo_file", default=c.personalized_demo_file)
    p.add_argument("--non_personalized_demo_file", default=c.non_personalized_demo_file)
    p.add_argument("--vllm_bin", default=c.vllm_bin)
    p.add_argument("--vllm_hf_model", default=c.vllm_hf_model)
    p.add_argument("--vllm_port", type=int, default=c.vllm_port)
    p.add_argument("--vllm_max_model_len", type=int, default=c.vllm_max_model_len)
    # dense index
    p.add_argument("--dense_index_dir_path", default=c.dense_index_dir_path)
    p.add_argument("--dense_query_encoder_path", default=c.dense_query_encoder_path)
    p.add_argument("--embed_dim", type=int, default=c.embed_dim)
    p.add_argument("--dense_dtype", choices=["float32", "float16"], default=c.dense_dtype,
                   help="RAM dtype for the dense index; float16 halves it (qwen3 491G -> ~245G)")
    p.add_argument("--passage_block_num", type=int, default=c.passage_block_num)
    p.add_argument("--faiss_n_gpu", type=int, default=c.faiss_n_gpu)
    p.add_argument("--query_gpu_id", type=int, default=c.query_gpu_id)
    p.add_argument("--query_encoder_batch_size", type=int, default=c.query_encoder_batch_size)
    # sparse index (also doc-fetch)
    p.add_argument("--sparse_index_dir_path", default=c.sparse_index_dir_path)
    p.add_argument("--bm25_k1", type=float, default=c.bm25_k1)
    p.add_argument("--bm25_b", type=float, default=c.bm25_b)
    p.add_argument("--topics", default=c.topics)
    return p


def config_from_args(args) -> PipelineConfig:
    if len(args.retrievers) != len(args.retrieval_query_types):
        raise SystemExit(
            f"--retrievers ({len(args.retrievers)}) and --retrieval_query_types "
            f"({len(args.retrieval_query_types)}) must have the same length")
    qr_list = args.qr if args.qr is not None else [""] * len(args.retrievers)
    if len(qr_list) != len(args.retrievers):
        raise SystemExit(
            f"--qr ({len(qr_list)}) must have the same length as --retrievers "
            f"({len(args.retrievers)})")
    qr_list = ["" if q in ("none", "None") else q for q in qr_list]
    enc_list = (args.dense_encoder_paths if args.dense_encoder_paths is not None
                else ["-"] * len(args.retrievers))
    if len(enc_list) != len(args.retrievers):
        raise SystemExit(
            f"--dense_encoder_paths ({len(enc_list)}) must have the same length as "
            f"--retrievers ({len(args.retrievers)})")
    enc_list = [None if e in ("-", "default", "none", "None", "") else e for e in enc_list]
    retrievers = [RetrieverSpec(n, qt, qr, encoder_path=enc)
                  for n, qt, qr, enc in zip(args.retrievers, args.retrieval_query_types,
                                            qr_list, enc_list)]
    # default LLM model by backend (local vLLM -> qwen3-32b ; OpenAI -> gpt-5-mini)
    llm_model = args.llm_model or ("gpt-5-mini" if args.llm_backend == "openai" else "qwen3-32b")
    # co-hosting VRAM: when the qwen3 reranker shares the LLM GPU with a local vLLM we
    # boot ourselves, shrink vLLM's reservation 0.90 -> 0.72 (~33G) so the reranker's
    # bf16 ~9G fits on the 46G card. Reusing an EXTERNAL vLLM (--llm_base_url) skips
    # this — that server's util was fixed at its own launch.
    vllm_util = 0.90
    if (args.reranker == "qwen3_reranker" and args.llm_backend == "local_vllm"
            and not args.llm_base_url and not args.rerank_remote_url):
        vllm_util = 0.72
        print(f"[run_server] reranker co-hosted with local vLLM -> gpu_mem_util {vllm_util}")
    return PipelineConfig(
        retrievers=retrievers,
        fusion_type=args.fusion_type,
        fusion_normalization=args.fusion_normalization,
        fuse_weights=args.fuse_weights,
        rrf_k=args.rrf_k,
        retrieval_top_k=args.retrieval_top_k,
        generation=args.generation,
        response_max_tokens=args.response_max_tokens,
        citations_max=args.citations_max,
        generation_top_k=args.generation_top_k,
        llm_backend=args.llm_backend,
        llm_model=llm_model,
        llm_base_url=args.llm_base_url,
        llm_gpu_id=args.llm_gpu_id,
        llm_max_tokens=args.llm_max_tokens,
        gtr_phi=args.gtr_phi,
        demo_file=args.demo_file,
        personalized_demo_file=args.personalized_demo_file,
        non_personalized_demo_file=args.non_personalized_demo_file,
        vllm_bin=args.vllm_bin,
        vllm_hf_model=args.vllm_hf_model,
        vllm_port=args.vllm_port,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_mem_util=vllm_util,
        reranker=args.reranker,
        rerank_top_k=args.rerank_top_k,
        rerank_batch_size=args.rerank_batch_size,
        rerank_quant=args.rerank_quant,
        rerank_remote_url=args.rerank_remote_url,
        qwen3_reranker_path=args.qwen3_reranker_path,
        reranking_query_type=args.reranking_query_type,
        dense_index_dir_path=args.dense_index_dir_path,
        dense_query_encoder_path=args.dense_query_encoder_path,
        embed_dim=args.embed_dim,
        dense_dtype=args.dense_dtype,
        passage_block_num=args.passage_block_num,
        faiss_n_gpu=args.faiss_n_gpu,
        query_gpu_id=args.query_gpu_id,
        query_encoder_batch_size=args.query_encoder_batch_size,
        sparse_index_dir_path=args.sparse_index_dir_path,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        topics=args.topics,
    )


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    app = create_app(config)
    # single worker: huge resident state + serialized GPU search
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
