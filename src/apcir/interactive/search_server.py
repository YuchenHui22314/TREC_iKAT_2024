"""FastAPI search server.

Two modes share one app builder `create_app(config, eager_load=, pipeline=)`:
  - eager_load=True (default, the Sim.API driver): load ONE index at startup, answer /search.
  - eager_load=False (the RALI Searcher): start EMPTY; load/evict capacity units on demand via
    /activate; /search runs against whatever is resident, with per-request RunSpec overrides.

Endpoints:
  GET  /health                       -> index-loaded flags + config
  GET  /models                       -> unit catalog + live residency + free RAM/VRAM
  POST /activate {units:[...]}        -> capacity-guarded background load/evict; -> {task_id, plan}
  GET  /activate/status/{task_id}     -> {state: running|done|error, progress, resident, error}
  POST /search {utterance, ...,       -> {response, citations, hits, ptkb_provenance, qid}
                retrievers?, fusion_type?, reranker?, generation?, ...}

Run with a SINGLE uvicorn worker (huge resident state, GPU search serialized).
"""

from __future__ import annotations

import threading
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from .pipeline import InteractivePipeline, PipelineConfig, RetrieverSpec, RunSpec
from .ptkb_store import PTKBStore


class RetrieverLeg(BaseModel):
    name: str = "qwen3"
    query_type: str = "raw"
    qr: str = ""
    encoder_path: Optional[str] = None
    unit: Optional[str] = None              # resident capacity unit to search


class ActivateRequest(BaseModel):
    units: List[str]                        # capacity units to make resident (the active set)


class SearchRequest(BaseModel):
    utterance: str
    history: List[str] = []                 # interleaved [u1, r1, u2, r2, ...] content list
    ptkb: List[str] = []                    # base PTKB statements (may be empty)
    topic_id: str = "0"
    user_id: str = "0"
    turn_index: int = 0
    # per-request RunSpec overrides (all optional; None -> server config default)
    retrievers: Optional[List[RetrieverLeg]] = None
    fusion_type: Optional[str] = None
    reranker: Optional[str] = None
    generation: Optional[str] = None
    generation_top_k: Optional[int] = None
    retrieval_top_k: Optional[int] = None

    @field_validator("retrievers")
    @classmethod
    def _retrievers_non_empty(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("retrievers must be non-empty when provided")
        return v


class SearchResponse(BaseModel):
    response: str
    citations: Dict[str, float]
    hits: List[List[Any]]                   # [[docid, score], ...]
    ptkb_provenance: List[str] = []
    qid: str = ""


def _build_run_spec(req: SearchRequest) -> Optional[RunSpec]:
    """Turn the optional RunSpec fields of a SearchRequest into a RunSpec (or None if none set)."""
    fields: Dict[str, Any] = {}
    if req.retrievers is not None:
        fields["retrievers"] = [
            RetrieverSpec(name=l.name, query_type=l.query_type, qr=l.qr,
                          encoder_path=l.encoder_path, unit=l.unit)
            for l in req.retrievers
        ]
    for f in ("fusion_type", "reranker", "generation", "generation_top_k", "retrieval_top_k"):
        v = getattr(req, f)
        if v is not None:
            fields[f] = v
    return RunSpec(**fields) if fields else None


def create_app(config: PipelineConfig, eager_load: bool = True,
               pipeline: Optional[InteractivePipeline] = None) -> FastAPI:
    pipeline = pipeline or InteractivePipeline(config)
    tasks: Dict[str, Dict[str, Any]] = {}   # task_id -> {state, progress, resident, error}
    tasks_lock = threading.Lock()
    activate_lock = threading.Lock()        # admits ONE activation at a time + guards `inflight`
    inflight = {"v": False}
    MAX_DONE_TASKS = 50

    def _prune_locked():                    # call under tasks_lock: cap finished tasks
        done = [tid for tid, t in tasks.items() if t["state"] != "running"]
        for tid in done[:max(0, len(done) - MAX_DONE_TASKS)]:
            tasks.pop(tid, None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if eager_load:
            print("[search_server] startup: booting LLM (if any) + loading index...")
            pipeline.load()
            print("[search_server] ready:", pipeline.health())
        else:
            print("[search_server] ready (empty; load units via POST /activate).")
        try:
            yield
        finally:
            pipeline.shutdown()

    app = FastAPI(title="apcir interactive search server", lifespan=lifespan)

    @app.get("/health")
    def health():
        return pipeline.health()

    @app.get("/models")
    def models():
        return pipeline.models_status()

    @app.post("/activate")
    def activate(req: ActivateRequest):
        unknown = [u for u in req.units if u not in pipeline.registry.names()]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown units: {unknown}")
        with activate_lock:                              # admit one activation at a time
            if inflight["v"]:
                raise HTTPException(status_code=409,
                                    detail="an activation is already in progress; wait for it")
            # synchronous fast-fail on capacity (set_active re-plans + is the real gate)
            plan = pipeline.capacity.plan(req.units, list(pipeline.resident()))
            if not plan.fits:
                raise HTTPException(status_code=409, detail=plan.reason)
            inflight["v"] = True
            task_id = uuid.uuid4().hex[:12]
            with tasks_lock:
                _prune_locked()
                tasks[task_id] = {"state": "running", "progress": [],
                                  "resident": sorted(pipeline.resident()), "error": None}

        def _work():
            def cb(msg, frac):
                with tasks_lock:
                    tasks[task_id]["progress"].append({"msg": msg, "frac": frac})
            try:
                pipeline.set_active(req.units, progress_cb=cb)
                with tasks_lock:
                    tasks[task_id]["state"] = "done"
            except Exception as e:                       # CapacityError / load failure
                with tasks_lock:
                    tasks[task_id]["state"] = "error"
                    tasks[task_id]["error"] = str(e)
            finally:
                with tasks_lock:
                    tasks[task_id]["resident"] = sorted(pipeline.resident())
                with activate_lock:
                    inflight["v"] = False

        threading.Thread(target=_work, daemon=True).start()
        return {"task_id": task_id, "plan": {"to_load": plan.to_load, "to_unload": plan.to_unload}}

    @app.get("/activate/status/{task_id}")
    def activate_status(task_id: str):
        with tasks_lock:
            t = tasks.get(task_id)
            if t is None:
                raise HTTPException(status_code=404, detail="unknown task_id")
            return dict(t, progress=list(t["progress"]))     # consistent snapshot under the lock

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        run_spec = _build_run_spec(req)
        ok, reason = pipeline.can_serve(run_spec)
        if not ok:
            raise HTTPException(status_code=409, detail=reason)
        store = PTKBStore(conversation_id=f"{req.topic_id}-{req.user_id}")
        store.update({"ptkb": req.ptkb})
        result = pipeline.process_turn(
            utterance=req.utterance, history=req.history, ptkb_store=store,
            topic_id=req.topic_id, user_id=req.user_id, turn_index=req.turn_index,
            run_spec=run_spec,
        )
        return SearchResponse(
            response=result.response, citations=result.citations,
            hits=[[d, s] for d, s in result.hits],
            ptkb_provenance=result.ptkb_provenance, qid=result.qid,
        )

    return app
