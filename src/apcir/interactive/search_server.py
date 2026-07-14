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
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

from .pipeline import InteractivePipeline, PipelineConfig, RetrieverSpec, RunSpec
from .ptkb_store import PTKBStore
from .store import Store


class RetrieverLeg(BaseModel):
    name: str = "qwen3"
    query_type: str = "raw"
    qr: str = ""
    encoder_path: Optional[str] = None
    unit: Optional[str] = None              # resident capacity unit to search
    encoder_label: Optional[str] = None     # short display label of the chosen encoder (leg-label seg)


class ActivateRequest(BaseModel):
    units: List[str]                        # capacity units to make resident (the active set)
    modes: Dict[str, str] = {}              # optional per-dense-unit load mode:
                                            # ram_fp16 (default) | gpu_resident | pq_refine


class SearchRequest(BaseModel):
    utterance: str
    history: List[str] = []                 # interleaved [u1, r1, u2, r2, ...] content list
    ptkb: List[str] = []                    # base PTKB statements (may be empty)
    topic_id: str = "0"
    user_id: str = "0"
    turn_index: int = 0
    session_id: Optional[int] = None        # if set + authed: persist this turn to the session
    extract_ptkb: bool = False              # if set + authed: extract + store new PTKB facts this turn
    # per-request RunSpec overrides (all optional; None -> server config default)
    retrievers: Optional[List[RetrieverLeg]] = None
    fusion_type: Optional[str] = None
    reranker: Optional[str] = None
    generation: Optional[str] = None
    generation_top_k: Optional[int] = None
    retrieval_top_k: Optional[int] = None
    cite_passages: Optional[bool] = None

    @field_validator("retrievers")
    @classmethod
    def _retrievers_non_empty(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("retrievers must be non-empty when provided")
        return v


class SearchResponse(BaseModel):
    response: str
    citations: Dict[str, float]
    hits: List[List[Any]]                   # [[docid, score], ...] — the fused/reranked ranking
    ptkb_provenance: List[str] = []
    qid: str = ""
    per_retriever: List[Dict[str, Any]] = []      # [{retriever, hits:[[docid,score]]}] per leg
    shared_docs: Dict[str, List[str]] = {}        # docid -> retrievers it appears in (>=2)
    reformulations: Dict[str, List[str]] = {}     # leg label -> reformulated query string(s)
    citation_spans: List[Dict[str, Any]] = []     # [{n, docid, start, end}] inline-[n] -> passage
    persisted: bool = False                       # was this turn saved to a session (authed + owned)?
    extracted_ptkb: List[str] = []                # NEW user-profile facts learned THIS turn (if any)


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _username_sane(cls, v):
        v = v.strip()
        if not v or len(v) > 40:
            raise ValueError("username must be 1-40 non-blank characters")
        return v

    @field_validator("password")
    @classmethod
    def _password_nonempty(cls, v):
        if not v:
            raise ValueError("password must not be empty")
        return v


class SessionCreate(BaseModel):
    title: str = ""


class SessionRename(BaseModel):
    title: str


class PtkbCreate(BaseModel):
    statement: str


class PtkbUpdate(BaseModel):
    statement: str


def _build_run_spec(req: SearchRequest) -> Optional[RunSpec]:
    """Turn the optional RunSpec fields of a SearchRequest into a RunSpec (or None if none set)."""
    fields: Dict[str, Any] = {}
    if req.retrievers is not None:
        fields["retrievers"] = [
            RetrieverSpec(name=l.name, query_type=l.query_type, qr=l.qr,
                          encoder_path=l.encoder_path, unit=l.unit, encoder_label=l.encoder_label)
            for l in req.retrievers
        ]
    for f in ("fusion_type", "reranker", "generation", "generation_top_k", "retrieval_top_k",
              "cite_passages"):
        v = getattr(req, f)
        if v is not None:
            fields[f] = v
    return RunSpec(**fields) if fields else None


def _persist_turn(store, pipeline, uid, req, result) -> Tuple[bool, List[str]]:
    """If authed + a valid OWNED session_id, persist this turn (SERVER-allocated index) and, when
    req.extract_ptkb, extract + store NEW per-user PTKB facts. Best-effort + isolated: a save or
    extract error never breaks the /search response, and an extract failure never loses the saved
    turn. Returns (saved, new_facts) — new_facts are the per-user PTKB statements learned THIS turn
    (for the UI's per-turn 'learned …' line); empty when not saved / nothing new / extraction off."""
    if uid is None or req.session_id is None:
        return False, []
    if store.get_session(req.session_id, uid) is None:
        return False, []                         # session isn't this user's
    new_facts: List[str] = []
    if req.extract_ptkb:                          # extract FIRST so the facts go INTO the saved payload
        try:                                      # (an extract error must NOT lose the turn -> caught)
            existing = store.list_ptkb(uid)
            seen = {p["statement"].strip().lower() for p in existing}      # normalized dedup set
            current = [p["statement"] for p in existing]
            for fact in pipeline.extract_ptkb(current, req.utterance, result.response or ""):
                norm = fact.strip().lower()
                if norm and norm not in seen:                              # dedup across turns + batch
                    seen.add(norm)
                    store.add_ptkb(uid, fact, source="extracted")
                    new_facts.append(fact)
        except Exception as e:  # noqa: BLE001
            print(f"[persist_turn] ptkb extract failed: {e}", flush=True)
    payload = {"citations": result.citations, "per_retriever": result.per_retriever,
               "shared_docs": result.shared_docs, "reformulations": result.reformulations,
               "citation_spans": result.citation_spans,
               "hits": [[d, s] for d, s in result.hits],  # full fused ranking -> faithful reload
               "extracted_ptkb": new_facts}                # -> reloaded session shows the 'learned' line
    try:
        store.add_turn(req.session_id, None, req.utterance, result.response or "", payload)
    except Exception as e:  # noqa: BLE001
        print(f"[persist_turn] save failed: {e}", flush=True)
        return False, []
    return True, new_facts


def create_app(config: PipelineConfig, eager_load: bool = True,
               pipeline: Optional[InteractivePipeline] = None,
               store: Optional[Store] = None) -> FastAPI:
    pipeline = pipeline or InteractivePipeline(config)
    store = store or Store(":memory:")
    tasks: Dict[str, Dict[str, Any]] = {}   # task_id -> {state, progress, resident, error}
    tasks_lock = threading.Lock()
    activate_lock = threading.Lock()        # admits ONE activation at a time + guards `inflight`
    inflight = {"v": False}
    MAX_DONE_TASKS = 50

    def _prune_locked():                    # call under tasks_lock: cap finished tasks
        done = [tid for tid, t in tasks.items() if t["state"] != "running"]
        for tid in done[:max(0, len(done) - MAX_DONE_TASKS)]:
            tasks.pop(tid, None)

    def _optional_user(authorization: Optional[str] = Header(None)) -> Optional[int]:
        """Resolve a bearer token to a user id, or None if absent/invalid (no 401) — lets /search
        run anonymously yet persist + extract when a logged-in user supplies a token."""
        if authorization and authorization.startswith("Bearer "):
            return store.user_for_token(authorization[7:])
        return None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if eager_load:
            print("[search_server] startup: booting LLM (if any) + loading index...")
            pipeline.load()
            print("[search_server] ready:", pipeline.health())
        else:
            pipeline.setup_remote_llm_if_needed()  # build the resource-free OpenAI LLM up front (rag/QR)
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
            plan = pipeline.capacity.plan(req.units, list(pipeline.resident()), modes=req.modes)
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
                pipeline.set_active(req.units, progress_cb=cb, modes=req.modes)
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
    def search(req: SearchRequest, uid: Optional[int] = Depends(_optional_user)):
        run_spec = _build_run_spec(req)
        ok, reason = pipeline.can_serve(run_spec)
        if not ok:
            raise HTTPException(status_code=409, detail=reason)
        ptkb_store = PTKBStore(conversation_id=f"{req.topic_id}-{req.user_id}")
        ptkb_store.update({"ptkb": req.ptkb})
        result = pipeline.process_turn(
            utterance=req.utterance, history=req.history, ptkb_store=ptkb_store,
            topic_id=req.topic_id, user_id=req.user_id, turn_index=req.turn_index,
            run_spec=run_spec,
        )
        try:                                               # never fail an OK search on persistence
            persisted, extracted = _persist_turn(store, pipeline, uid, req, result)
        except Exception as e:  # noqa: BLE001
            print(f"[search] persist failed: {e}", flush=True)
            persisted, extracted = False, []
        return SearchResponse(
            response=result.response, citations=result.citations,
            hits=[[d, s] for d, s in result.hits],
            ptkb_provenance=result.ptkb_provenance, qid=result.qid,
            per_retriever=result.per_retriever, shared_docs=result.shared_docs,
            reformulations=result.reformulations, citation_spans=result.citation_spans,
            persisted=persisted, extracted_ptkb=extracted,
        )

    @app.get("/doc")
    def get_doc(docid: str):
        """Fetch a passage's full text for the UI's 'view passage' modal. Needs a resident
        doc-fetch (sparse/BM25) index; 409 otherwise. docid is a QUERY param (robust for
        URL-shaped qrecc docids with slashes/colons): GET /doc?docid=..."""
        if pipeline._docfetch is None:
            raise HTTPException(status_code=409,
                                detail="no doc-fetch index resident; activate a sparse/BM25 unit")
        return {"docid": docid, "text": pipeline._passage_text(docid)}

    # --- auth + session management ----------------------------------------- #
    def _current_user(authorization: Optional[str] = Header(None)) -> int:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        uid = store.user_for_token(authorization[7:])
        if uid is None:
            raise HTTPException(status_code=401, detail="invalid or expired token")
        return uid

    @app.post("/auth/login")
    def login(req: LoginRequest):
        u = store.verify_user(req.username, req.password)
        if not u:
            raise HTTPException(status_code=401, detail="invalid username or password")
        return {"token": store.create_token(u["id"]), "user": u}

    @app.post("/auth/register")
    def register(req: RegisterRequest):
        """Open self-signup for the demo (the service sits behind an SSH tunnel; accounts are
        non-admin and passwords are hashed). Auto-logs the new user in."""
        try:
            uid = store.create_user(req.username, req.password, is_admin=False)
        except ValueError:
            raise HTTPException(status_code=409, detail="username already exists")
        u = store.get_user(uid)
        return {"token": store.create_token(uid), "user": u}

    @app.get("/auth/users")
    def users_directory():
        """Usernames + admin flags for the login screen's account picker (demo affordance)."""
        return store.list_users()

    @app.post("/auth/logout")
    def logout(authorization: Optional[str] = Header(None)):
        if authorization and authorization.startswith("Bearer "):
            store.delete_token(authorization[7:])
        return {"ok": True}

    @app.get("/auth/me")
    def me(uid: int = Depends(_current_user)):
        return store.get_user(uid)

    @app.get("/sessions")
    def list_sessions(uid: int = Depends(_current_user)):
        return store.list_sessions(uid)

    @app.post("/sessions")
    def create_session(req: SessionCreate, uid: int = Depends(_current_user)):
        sid = store.create_session(uid, req.title)
        return {"id": sid, "title": req.title}

    @app.patch("/sessions/{session_id}")
    def rename_session(session_id: int, req: SessionRename, uid: int = Depends(_current_user)):
        if not store.rename_session(session_id, uid, req.title):
            raise HTTPException(status_code=404, detail="session not found")
        return {"ok": True}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: int, uid: int = Depends(_current_user)):
        if not store.delete_session(session_id, uid):
            raise HTTPException(status_code=404, detail="session not found")
        return {"ok": True}

    @app.get("/sessions/{session_id}/turns")
    def session_turns(session_id: int, uid: int = Depends(_current_user)):
        if store.get_session(session_id, uid) is None:
            raise HTTPException(status_code=404, detail="session not found")
        return store.list_turns(session_id)

    # --- per-user PTKB (view / edit / delete / manual-add) ----------------- #
    @app.get("/ptkb")
    def list_ptkb(uid: int = Depends(_current_user)):
        return store.list_ptkb(uid)

    @app.post("/ptkb")
    def add_ptkb(req: PtkbCreate, uid: int = Depends(_current_user)):
        return {"id": store.add_ptkb(uid, req.statement, source="manual")}

    @app.put("/ptkb/{ptkb_id}")
    def update_ptkb(ptkb_id: int, req: PtkbUpdate, uid: int = Depends(_current_user)):
        if not store.update_ptkb(ptkb_id, uid, req.statement):
            raise HTTPException(status_code=404, detail="ptkb statement not found")
        return {"ok": True}

    @app.delete("/ptkb/{ptkb_id}")
    def delete_ptkb(ptkb_id: int, uid: int = Depends(_current_user)):
        if not store.delete_ptkb(ptkb_id, uid):
            raise HTTPException(status_code=404, detail="ptkb statement not found")
        return {"ok": True}

    return app
