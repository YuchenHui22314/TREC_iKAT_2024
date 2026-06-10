"""FastAPI search server: loads the index ONCE at startup, answers POST /search.

The expensive resident state (RAM dense index + GPU faiss + BM25 lucene + doc-fetch)
is loaded once in the lifespan startup, so the Sim.API driver can be restarted/iterated
without paying the 2-3 min index load again. Run with a SINGLE uvicorn worker (huge
resident state, GPU search serialized).

Endpoints:
  POST /search  {utterance, history, ptkb, topic_id, user_id, turn_index}
                -> {response, citations, hits, ptkb_provenance, qid}
  GET  /health  -> index-loaded flags + config

`create_app(config)` builds the app for a given PipelineConfig; `run_server.py` is the CLI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .pipeline import InteractivePipeline, PipelineConfig
from .ptkb_store import PTKBStore


class SearchRequest(BaseModel):
    utterance: str
    history: List[str] = []                 # interleaved [u1, r1, u2, r2, ...] content list
    ptkb: List[str] = []                    # base PTKB statements (may be empty)
    topic_id: str = "0"
    user_id: str = "0"
    turn_index: int = 0


class SearchResponse(BaseModel):
    response: str
    citations: Dict[str, float]
    hits: List[List[Any]]                   # [[docid, score], ...]
    ptkb_provenance: List[str] = []
    qid: str = ""


def create_app(config: PipelineConfig) -> FastAPI:
    pipeline = InteractivePipeline(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("[search_server] loading resident index state (this can take 2-3 min)...")
        pipeline.load()
        print("[search_server] ready:", pipeline.health())
        yield
        print("[search_server] shutting down.")

    app = FastAPI(title="apcir interactive search server", lifespan=lifespan)

    @app.get("/health")
    def health():
        return pipeline.health()

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        # stateless w.r.t. PTKB across HTTP calls: the driver owns the per-conversation
        # PTKBStore; here we build a throwaway store from the base ptkb passed in so the
        # pipeline can pick relevant statements for this turn.
        store = PTKBStore(conversation_id=f"{req.topic_id}-{req.user_id}")
        store.update({"ptkb": req.ptkb})
        result = pipeline.process_turn(
            utterance=req.utterance,
            history=req.history,
            ptkb_store=store,
            topic_id=req.topic_id,
            user_id=req.user_id,
            turn_index=req.turn_index,
        )
        return SearchResponse(
            response=result.response,
            citations=result.citations,
            hits=[[d, s] for d, s in result.hits],
            ptkb_provenance=result.ptkb_provenance,
            qid=result.qid,
        )

    return app
