"""Standalone Qwen3-Reranker HTTP server — run it on ANY machine with a free GPU
(e.g. octal31's A5000s) and point eval / the interactive server at it from octal40.

The code tree + envs live on shared NFS, so the SAME command works on every octal:

    # on octal31 (or wherever the free GPU is):
    cd /data/rech/huiyuche/TREC_iKAT_2024/src && \
    PATH=/data/rech/huiyuche/envs/trec_ikat/bin:$PATH \
    python -m apcir.search.rerank_server --gpu_id 0 --port 8200

    # then on octal40:
    #   eval:   --rerank_remote_url http://octal31:8200
    #   server: --rerank_remote_url http://octal31:8200

API (mirrors QwenReranker.score 1:1):
    POST /rerank {instruction, query, docs: [str], batch_size} -> {scores: [float]}
    GET  /health -> {model, device, ready}

LAN cost is negligible: ~100KB per 50-doc request at 0.37ms RTT / ~96MB/s (measured
octal40<->octal31). VRAM: bf16 ~9G (fits an A5000 24G with room), or --quant 8b (~5G).
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class RerankRequest(BaseModel):
    instruction: str
    query: str
    docs: list
    batch_size: int = 8


def create_app(reranker_type: str, model_path: str, cache_dir: str, quant: str,
               gpu_id: int, max_length: int) -> FastAPI:
    from apcir.search.rerank import build_local_reranker
    import torch

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"[rerank_server] loading {reranker_type} on {device} (quant={quant})...")
    reranker = build_local_reranker(reranker_type, cache_dir=cache_dir, device=device,
                                    quant=quant, qwen3_path=model_path)
    print("[rerank_server] ready.")

    app = FastAPI(title="apcir rerank server")

    @app.get("/health")
    def health():
        return {"reranker_type": reranker_type, "model": model_path, "device": device, "ready": True}

    @app.post("/rerank")
    def rerank(req: RerankRequest):
        scores = reranker.score(req.instruction, req.query, req.docs, req.batch_size)
        return {"scores": scores}

    return app


def main() -> int:
    p = argparse.ArgumentParser(description="standalone HF rerank server (any reranker)")
    p.add_argument("--reranker_type", default="qwen3_reranker",
                   help="qwen3_reranker | monot5_base|_10k | monot5_large|_10k | monot5_3b|_10k | rankllama")
    p.add_argument("--model_path", default="Qwen/Qwen3-Reranker-4B",
                   help="qwen3 HF path (other types resolve their own HF name)")
    p.add_argument("--cache_dir", default="/data/rech/huiyuche/huggingface")
    p.add_argument("--quant", default="none", choices=["none", "8b", "4b"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--host", default="0.0.0.0")   # reachable from the other octals
    p.add_argument("--port", type=int, default=8200)
    args = p.parse_args()
    app = create_app(args.reranker_type, args.model_path, args.cache_dir, args.quant,
                     args.gpu_id, args.max_length)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
