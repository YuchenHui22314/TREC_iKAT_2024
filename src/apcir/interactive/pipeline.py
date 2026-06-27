"""InteractivePipeline: the per-turn retrieve -> fuse -> answer logic, server side.

Holds the resident search state (RAM dense index + GPU faiss + BM25 lucene + doc-fetch
searcher) and, per simulated turn:
  1. build a `Turn` from the API payload (utterance + interleaved history + base PTKB);
  2. for each configured retriever, run online QR (`Turn.query_type_2_query`) then retrieve
     (BM25 via lucene batch_search; ANCE via encode -> RAM PHASE-B search -> ranking);
  3. normalize + fuse (RRF by default; personalization-free linear/round_robin/concat);
  4. build an extractive response (top passages, hard-capped) + top-10 citations
     ({docid: score}) + optional meta.ptkb_provenance.

v1 reuses the offline functions verbatim for faithfulness:
  - apcir.search.dense_search.get_test_query_embedding / build_faiss_index / get_dense_ranking_list
  - apcir.interactive.ram_index.search_query_against_ram
  - apcir.search.fuse.RRF / normalize_scores / per_query_linear_combination / round_robin_fusion / concat
  - apcir.functional.topics.Turn / load_document_by_id
NOTE: qwen3 query encoders are CACHED by (path, device) in dense_search._get_qwen_encoder
(loaded once, reused across turns — supports multiple per-leg encoders); ANCE encoders still
load per-turn (acceptable, not on the current path).
"""

from __future__ import annotations

import gc
import json
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from pyserini.search.lucene import LuceneSearcher

from apcir.functional.topics import Turn, load_document_by_id
from apcir.search.dense_search import (
    build_faiss_index, get_test_query_embedding, get_dense_ranking_list,
)
from apcir.search import fuse as fuse_mod
from .ram_index import RamBlockSource, search_query_against_ram
from .ptkb_store import PTKBStore
from .llm_client import SharedLLMClient
from .rewriter import OnlineRewriter, RewriterConfig, context_turns_from_history
from .generation import rag_response
from .capacity import IndexRegistry, CapacityManager, CapacityPlan, CapacityError


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RetrieverSpec:
    """One configured retriever leg of the fusion."""
    name: str                 # "BM25" | "ance" | "qwen3" | "conv-qwen3" | "splade_v3" ...
    query_type: str           # used when qr=="": "raw" | "full_conversation_dense" | "qwen_conversation"[_ptkb]
    qr: str = ""              # online QR name (rar / rar_personalized_cot1 / MQ4CS_persq / GtR / ptkb_sum);
                              # "" or "none" = no QR, use query_type. A QR may emit MULTIPLE queries (GtR) -> fused.
    encoder_path: Optional[str] = None  # per-leg DENSE query-encoder ckpt; None -> global
                              # dense_query_encoder_path. Lets two qwen3 legs (e.g. conv-qwen3 +
                              # pers-conv-qwen3) use DIFFERENT encoders while sharing one doc index.


@dataclass
class PipelineConfig:
    retrievers: List[RetrieverSpec] = field(default_factory=lambda: [
        RetrieverSpec("BM25", "raw"),
        RetrieverSpec("ance", "full_conversation_dense"),
    ])
    # fusion
    fusion_type: str = "RRF"               # RRF | linear_combination | round_robin | concat
    fusion_normalization: str = "min-max"  # for linear_combination; RRF ignores scores
    fuse_weights: Optional[List[float]] = None
    rrf_k: int = 60
    retrieval_top_k: int = 1000
    # reranking (between fusion and generation; co-hosted on the gen-LLM GPU)
    reranker: str = "none"                 # "none" | "qwen3_reranker"
    rerank_top_k: int = 50
    rerank_batch_size: int = 8             # (instruction+profile+conv+doc) pairs are long
    rerank_quant: str = "none"             # none -> bf16 (~9G) | 8b | 4b (bitsandbytes)
    rerank_remote_url: Optional[str] = None  # e.g. http://octal31:8200 -> RemoteReranker (0 local VRAM)
    qwen3_reranker_path: str = "Qwen/Qwen3-Reranker-4B"
    reranking_query_type: str = "qwen_3_rerank_instruct_full"
    #   qwen_3_rerank_instruct_full -> conversational instruction + profile-first query
    #   (built from the live Turn); any *_rw reformulation name (e.g. MQ4CS_persq_rw)
    #   -> NATIVE default instruction + that online-QR rewrite as the query.
    # generation
    generation: str = "rag"                # "rag" (LLM, shared) | "extractive" (no-LLM fallback)
    response_max_tokens: int = 512
    citations_max: int = 10
    generation_top_k: int = 3
    # shared LLM (QR + RAG generation) — OpenAI-compatible client
    llm_backend: str = "local_vllm"        # "local_vllm" | "openai"
    llm_model: str = "qwen3-32b"           # served-model-name (local) or e.g. gpt-4o-mini (openai)
    llm_base_url: Optional[str] = None     # local default http://127.0.0.1:8100/v1 ; openai default None
    llm_gpu_id: int = 3                    # vLLM server pinned here (CUDA_VISIBLE_DEVICES=3); faiss uses 0..n-1
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0
    # online-QR promptor demos + GtR fan-out
    demo_file: str = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/"
                      "original_demonstration.json")                       # rar (no ptkb)
    personalized_demo_file: str = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/"
                                   "demonstration_using_ikat23.json")      # personalized_cot (has ptkb)
    non_personalized_demo_file: str = ("/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat23/"
                                       "non_personalized_demonstration_using_ikat23.json")
    gtr_phi: int = 2
    # local vLLM server (used only when llm_backend=="local_vllm"); see vllm_server.py
    vllm_bin: str = "/data/rech/huiyuche/envs/vllm_qwen3/bin/vllm"
    vllm_hf_model: str = "Qwen/Qwen3-32B-AWQ"
    vllm_port: int = 8100
    vllm_max_model_len: int = 16384
    vllm_gpu_mem_util: float = 0.90
    # dense (ANCE) index
    dense_index_dir_path: str = "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_ance_merged_2"
    dense_query_encoder_path: str = (
        "/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/"
        "snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234")
    embed_dim: int = 768
    dense_dtype: str = "float32"            # "float16" halves index RAM (qwen3 491G -> ~245G)
    passage_block_num: int = 12
    faiss_n_gpu: int = 3                    # GPUs 0,1,2 (GPU 3 reserved for the vLLM server)
    use_gpu_for_faiss: bool = True
    tempmem: int = -1
    query_gpu_id: int = 0
    query_encoder_batch_size: int = 200
    # sparse (BM25) index — also used for doc-fetch (passage text lives in the lucene index).
    # NOTE: the /part/01 ".._official_sparse_index" dir is EMPTY (wiped); the populated 159G
    # index on fast disk is fengran_sparse_index_2 (116,838,987 passages, stores raw contents,
    # docid form "clueweb22-en0037-11-03275:0" == the docid:passage citation key).
    sparse_index_dir_path: str = (
        "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2")
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    # SPLADE_v3 learned-sparse leg (retriever name "splade_v3"); its inverted index loads to RAM.
    # splade_value_dtype: "int16" (~176G, empirically lossless vs fp32, numba-safe) | "float32" (~235G).
    splade_query_encoder_path: str = (
        "/data/rech/huiyuche/huggingface/models--naver--splade-v3/snapshots/"
        "8291b13eb8f4e24cc745c542825f14eb87296879")
    splade_index_dir_path: str = "/part/01/Tmp/yuchen/indexes/splade_v3_clueweb22B"
    splade_dim_voc: int = 30522
    splade_value_dtype: str = "int16"
    # topics tag (drives the iKAT branch in query building); interactive synthetic tag
    topics: str = "ikat_26_sim"
    seed: int = 42


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class TurnResult:
    response: str
    citations: Dict[str, float]                 # {docid: score}, top-N
    hits: List[Tuple[str, float]]               # full fused ranking (docid, score)
    ptkb_provenance: List[str] = field(default_factory=list)
    qid: str = ""


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class InteractivePipeline:
    def __init__(self, config: PipelineConfig, registry: Optional[IndexRegistry] = None,
                 capacity: Optional[CapacityManager] = None):
        self.config = config
        self._ram: Optional[RamBlockSource] = None
        self._faiss = None
        self._bm25: Optional[LuceneSearcher] = None
        self._docfetch: Optional[LuceneSearcher] = None
        self._llm: Optional[SharedLLMClient] = None
        self._rewriter: Optional[OnlineRewriter] = None
        self._vllm = None                  # VLLMServer (local_vllm backend only)
        self._reranker = None              # QwenReranker (co-hosted on the LLM GPU)
        self._splade = None                # SparseRetrieval (SPLADE_v3 inverted index in RAM)
        self._needs_dense = any(r.name in ("ance", "conv-ance", "qwen3", "conv-qwen3")
                                for r in config.retrievers)
        self._needs_sparse = any(r.name == "BM25" for r in config.retrievers)
        self._needs_splade = any(r.name == "splade_v3" for r in config.retrievers)
        self._needs_llm = (config.generation == "rag"
                           or any(r.qr and r.qr != "none" for r in config.retrievers))
        # --- dynamic residency (RALI Searcher): which capacity units are loaded right now ---
        self._dense: Dict[str, RamBlockSource] = {}   # unit name -> resident dense index
        self._resident: set = set()                    # all resident unit names (any kind)
        if registry is None:
            from os.path import join, dirname
            registry = IndexRegistry.from_yaml(join(dirname(__file__), "capacity_config.yaml"))
        self.registry = registry
        self.capacity = capacity or CapacityManager(registry)
        self._residency_lock = threading.RLock()    # serialize residency changes (vs each other + search)

    # --- startup: load resident state once --------------------------------- #
    def load(self):
        c = self.config
        # LLM first: if local vLLM fails to boot, fail fast BEFORE the 336G index load.
        if self._needs_llm:
            self._setup_llm()
        if self._needs_dense:
            self._ram = RamBlockSource(c.dense_index_dir_path, c.passage_block_num, c.embed_dim,
                                       store_dtype=c.dense_dtype)
            self._faiss = build_faiss_index(self._make_args())   # GPUs 0..faiss_n_gpu-1 (NOT GPU 3)
        if self._needs_splade:
            from apcir.splade_index import SparseRetrieval
            print(f"[pipeline] loading SPLADE_v3 index ({c.splade_value_dtype}) from "
                  f"{c.splade_index_dir_path} (after the dense index, to bound peak RAM)...")
            self._splade = SparseRetrieval(c.splade_index_dir_path, "None", c.splade_dim_voc,
                                           c.retrieval_top_k, value_dtype=c.splade_value_dtype)
        if self._needs_sparse:
            self._bm25 = LuceneSearcher(c.sparse_index_dir_path)
            self._bm25.set_bm25(c.bm25_k1, c.bm25_b)
        # doc-fetch: passage text is stored in the lucene (sparse) index
        self._docfetch = LuceneSearcher(c.sparse_index_dir_path)
        if c.reranker != "none":
            if c.rerank_remote_url:
                # model hosted on another machine (rerank_server.py) — the remote knows the
                # reranker TYPE (qwen3_reranker / rankllama / monot5 ...), so ANY reranker is
                # usable remotely with zero local VRAM; LAN round-trip is negligible.
                from apcir.search.rerank import RemoteReranker
                print(f"[pipeline] using remote reranker ({c.reranker}) at {c.rerank_remote_url}")
                self._reranker = RemoteReranker(c.rerank_remote_url)
            elif c.reranker == "qwen3_reranker":
                # co-hosted on the gen-LLM GPU (cuda:llm_gpu_id). VRAM: openai backend ->
                # GPU 3 is free (bf16 ~9G fits trivially); local_vllm backend -> run_server
                # lowers vllm_gpu_mem_util to ~0.72 so ~13G stays free. rerank_quant 8b/4b
                # is available but its RANKING quality on iKAT is NOT yet validated (see
                # QwenReranker docstring) — prefer bf16 or the remote reranker.
                from apcir.search.rerank import QwenReranker
                import torch as _torch
                dev = (f"cuda:{c.llm_gpu_id}" if _torch.cuda.is_available() else "cpu")
                print(f"[pipeline] loading qwen3 reranker on {dev} (quant={c.rerank_quant})...")
                self._reranker = QwenReranker(
                    model_path=c.qwen3_reranker_path, quant=c.rerank_quant, device=dev)
            else:
                raise SystemExit(
                    f"reranker '{c.reranker}' has no LOCAL loader in the interactive pipeline; "
                    f"host it with `python -m apcir.search.rerank_server --reranker_type "
                    f"{c.reranker} ...` and pass --rerank_remote_url.")

    def _setup_llm(self):
        """Build the shared LLM client (+ rewriter). For local_vllm, boot the vLLM
        OpenAI server on GPU 3 in a SEPARATE process (CUDA_VISIBLE_DEVICES=3) so it can
        never collide with FAISS on GPUs 0-2."""
        c = self.config
        base_url = c.llm_base_url
        if c.llm_backend == "local_vllm" and base_url:
            # an EXTERNAL vLLM server was given (e.g. a persistent one already on GPU 3):
            # reuse it, do NOT boot a second engine (it would fight for the same GPU/port).
            print(f"[pipeline] reusing external vLLM server at {base_url}")
        elif c.llm_backend == "local_vllm":
            from .vllm_server import VLLMServer, VLLMServerConfig
            self._vllm = VLLMServer(VLLMServerConfig(
                vllm_bin=c.vllm_bin, hf_model=c.vllm_hf_model, served_model_name=c.llm_model,
                gpu_id=c.llm_gpu_id, port=c.vllm_port, max_model_len=c.vllm_max_model_len,
                gpu_memory_utilization=c.vllm_gpu_mem_util))
            self._vllm.start()
            self._vllm.wait_until_ready()
            base_url = self._vllm.base_url()
        self._llm = SharedLLMClient(
            backend=c.llm_backend, model=c.llm_model, base_url=base_url,
            max_tokens=c.llm_max_tokens, temperature=c.llm_temperature)
        self._rewriter = OnlineRewriter(self._llm, RewriterConfig(
            demo_file=c.demo_file, personalized_demo_file=c.personalized_demo_file,
            non_personalized_demo_file=c.non_personalized_demo_file, gtr_phi=c.gtr_phi))
        print(f"[pipeline] LLM ready: {self._llm}")

    def shutdown(self):
        if self._vllm is not None:
            self._vllm.stop()

    def health(self) -> Dict[str, Any]:
        return {
            "dense_loaded": self._ram is not None,
            "dense_vectors": getattr(self._ram, "total_vecs", None),
            "sparse_loaded": self._bm25 is not None,
            "splade_loaded": self._splade is not None,
            "docfetch_loaded": self._docfetch is not None,
            "retrievers": [(r.name, r.query_type, r.qr) for r in self.config.retrievers],
            "fusion_type": self.config.fusion_type,
            "generation": self.config.generation,
            "llm_backend": self.config.llm_backend if self._needs_llm else None,
            "llm_ready": (self._llm.health() if self._llm is not None else None),
            "vllm_pid": (self._vllm.pid() if self._vllm is not None else None),
            "reranker": (self.config.reranker if self._reranker is not None else "none"),
            "reranking_query_type": (self.config.reranking_query_type
                                     if self._reranker is not None else None),
        }

    # --- dynamic residency: load/unload capacity units on demand ----------- #
    @staticmethod
    def _progress(cb, msg, frac):
        """Best-effort progress callback: a UI exception must never break load/unload/rollback."""
        if cb is None:
            return
        try:
            cb(msg, frac)
        except Exception:
            pass

    def resident(self) -> set:
        """Names of all currently-resident capacity units."""
        return set(self._resident)

    _LOADABLE_KINDS = {"dense"}   # kinds with a load_/unload_ impl (extended as more are wired)

    def set_active(self, active_set, progress_cb=None) -> CapacityPlan:
        """Make exactly `active_set` resident (active-set semantics): evict units not in it, load
        units missing from it. Refuse (CapacityError) if the set won't fit by load-peak vs live
        free RAM/VRAM. Transactional + locked: holds an exclusive residency lock, validates every
        loader BEFORE mutating (so an unimplemented kind fails clean without first evicting the
        current set), and on a mid-load failure rolls back the partial loads so `_resident` always
        matches the real objects. `progress_cb(msg, frac)` wraps each load/unload.
        NOTE: units EVICTED earlier in this call are NOT restored on a later load failure — the
        service may be left with fewer units resident (but `_resident`/`_dense` stay consistent)."""
        with self._residency_lock:
            plan = self.capacity.plan(list(active_set), list(self._resident))
            if not plan.fits:
                raise CapacityError(plan.reason)
            for unit in plan.to_load:                       # validate BEFORE touching residency
                kind = self.registry.get(unit).kind
                if kind not in self._LOADABLE_KINDS:
                    raise NotImplementedError(
                        f"no loader for kind {kind!r} (unit {unit!r}); supported: "
                        f"{sorted(self._LOADABLE_KINDS)}")
            for unit in plan.to_unload:
                self._unload_unit(unit, progress_cb)
            loaded_now = []
            try:
                for unit in plan.to_load:
                    self._load_unit(unit, progress_cb)
                    loaded_now.append(unit)
            except Exception:
                for unit in loaded_now:                     # roll back partial loads -> consistent
                    self._unload_unit(unit, progress_cb)
                raise
            return plan

    def _load_unit(self, unit: str, progress_cb=None):
        kind = self.registry.get(unit).kind
        if kind == "dense":
            self.load_dense(unit, progress_cb)
        else:
            raise NotImplementedError(
                f"load for kind {kind!r} not yet wired (set_active is dense-only in 2A.2a)")

    def _unload_unit(self, unit: str, progress_cb=None):
        kind = self.registry.get(unit).kind
        if kind == "dense":
            self.unload_dense(unit, progress_cb)
        else:
            raise NotImplementedError(f"unload for kind {kind!r} not yet wired")

    def load_dense(self, unit: str, progress_cb=None):
        """Construct a RAM-resident dense index for `unit` (RamBlockSource preloads all blocks
        into RAM at construction, so this IS the load). Acquires the residency lock (reentrant)."""
        with self._residency_lock:
            fp = self.registry.get(unit)
            self._progress(progress_cb, f"loading {unit}", 0.0)
            ram = RamBlockSource(fp.index_dir, fp.block_num, fp.embed_dim,
                                 store_dtype=fp.dtype or "float16")
            self._dense[unit] = ram
            self._resident.add(unit)
            self._progress(progress_cb, f"loaded {unit} ({ram.total_vecs:,} vecs)", 1.0)

    def unload_dense(self, unit: str, progress_cb=None):
        """Drop the resident dense index for `unit` and reclaim its RAM/VRAM. Acquires the lock."""
        with self._residency_lock:
            self._progress(progress_cb, f"unloading {unit}", 0.0)
            self._dense.pop(unit, None)
            self._resident.discard(unit)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._progress(progress_cb, f"unloaded {unit}", 1.0)

    def search_dense_unit(self, unit: str, query_embeddings, topN: int):
        """Search a RESIDENT dense unit with raw query embeddings. Returns (D, I): scores + docids,
        shape (n_query, topN). fp16 units use the GPU fp16 path (no faiss). The RamBlockSource ref
        is snapshotted under the residency lock so a concurrent set_active unload can't pull it out
        mid-search (the in-flight search keeps it alive; a concurrent activate may thus transiently
        hold both the old unit and a newly-loaded one in RAM)."""
        with self._residency_lock:
            if unit not in self._dense:
                raise KeyError(f"unit {unit!r} not resident; activate it first "
                               f"(resident dense: {sorted(self._dense)})")
            ram = self._dense[unit]
        if str(getattr(ram, "store_dtype", "")) != "float16":
            raise ValueError(
                f"search_dense_unit currently supports only fp16 units; {unit!r} is {ram.store_dtype} "
                f"(fp32 needs a faiss index, not wired into dynamic residency yet)")
        gpus = list(range(self.config.faiss_n_gpu))
        return search_query_against_ram(query_embeddings, ram, None, topN, gpus=gpus)

    # --- per-turn ---------------------------------------------------------- #
    def process_turn(
        self,
        utterance: str,
        history: List[str],
        ptkb_store: Optional[PTKBStore] = None,
        topic_id: str = "0",
        user_id: str = "0",
        turn_index: int = 0,
    ) -> TurnResult:
        c = self.config
        qid = f"{topic_id}-{user_id}-{turn_index}"
        turn = self._build_turn(utterance, history, qid, topic_id, user_id, ptkb_store)
        context_turns = context_turns_from_history(history)

        # 1) retrieve. The shared-corpus dense group + parallel-sparse speedup applies ONLY for the
        #    order-insensitive RRF fusion (our submissions): non-QR dense legs on the single index do
        #    ONE shared corpus pass (self._dense_group_search), and GPU-FREE sparse legs (BM25/splade,
        #    incl. their slow online-QR LLM call) run in a worker thread CONCURRENTLY with that GPU
        #    search — disjoint resources (HTTP/Lucene vs CUDA, GIL released in both). Any GPU-using
        #    "rest" leg (a dense leg WITH a QR) runs SERIALLY in the main thread (it would collide
        #    with the dense group on the GPU). For order-sensitive fusion (round_robin /
        #    linear_combination / concat) or no dense group, fall back to the original sequential
        #    per-leg loop (preserves config order + positional fuse weights).
        _SPARSE = ("BM25", "splade_v3")
        def _is_dense_grp(s: RetrieverSpec) -> bool:
            return s.name in self._DENSE_NAMES and (not s.qr or s.qr == "none")
        dense_group = [s for s in c.retrievers if _is_dense_grp(s)]

        if c.fusion_type == "RRF" and dense_group:
            sparse_rest = [s for s in c.retrievers if not _is_dense_grp(s) and s.name in _SPARSE]
            gpu_rest = [s for s in c.retrievers if not _is_dense_grp(s) and s.name not in _SPARSE]

            def _run(specs) -> List[Dict[str, List[Any]]]:
                hl: List[Dict[str, List[Any]]] = []
                for spec in specs:
                    hl.extend(self._retrieve_one(spec, turn, qid, context_turns))
                return hl

            with ThreadPoolExecutor(max_workers=1) as ex:
                sparse_fut = ex.submit(_run, sparse_rest) if sparse_rest else None
                dense_hits = self._dense_group_search(dense_group, turn, qid)   # GPU, main thread
                gpu_rest_hits = _run(gpu_rest)                                   # GPU, after dense
                sparse_hits = sparse_fut.result() if sparse_fut else []
            hits_list = sparse_hits + dense_hits + gpu_rest_hits   # RRF is order-insensitive
        else:
            hits_list = []
            for spec in c.retrievers:
                for hits in self._retrieve_one(spec, turn, qid, context_turns):
                    if c.fusion_type == "linear_combination":
                        hits = fuse_mod.normalize_scores(hits, c.fusion_normalization)
                    hits_list.append(hits)

        # 2) fuse
        fused = self._fuse(hits_list, qid)
        ranked = fused[qid]

        # 2.5) rerank the fused top-k (tail [k:] keeps its original order behind it)
        if self._reranker is not None and len(ranked) > 1:
            ranked = self._rerank(ranked, turn)

        # 3) generate + citations
        response = None
        if c.generation == "rag" and self._llm is not None:
            response = rag_response(
                self._llm, self._docfetch, ranked, context_turns, turn.ptkb,
                utterance, c.generation_top_k, _truncate_tokens, c.response_max_tokens)
        if response is None:                       # extractive fallback (also the no-LLM path)
            response = self._extractive_response(ranked)
        citations = {d.docid: float(d.score) for d in ranked[:c.citations_max]}
        hits_out = [(d.docid, float(d.score)) for d in ranked]

        # 4) ptkb provenance (optional best-effort)
        prov = ptkb_store.relevant_for(turn) if ptkb_store is not None else []

        return TurnResult(response=response, citations=citations, hits=hits_out,
                          ptkb_provenance=prov, qid=qid)

    # --- internals --------------------------------------------------------- #
    def _make_args(self, **overrides) -> SimpleNamespace:
        c = self.config
        a = SimpleNamespace(
            seed=c.seed, topics=c.topics,
            retrieval_top_k=c.retrieval_top_k,
            # dense
            use_pyserini_dense_search=False,
            dense_query_encoder_path=c.dense_query_encoder_path,
            dense_index_dir_path=c.dense_index_dir_path,
            embed_dim=c.embed_dim, passage_block_num=c.passage_block_num,
            faiss_n_gpu=c.faiss_n_gpu, use_gpu_for_faiss=c.use_gpu_for_faiss,
            tempmem=c.tempmem, query_gpu_id=c.query_gpu_id,
            query_encoder_batch_size=c.query_encoder_batch_size,
            # sparse
            sparse_index_dir_path=c.sparse_index_dir_path,
            bm25_k1=c.bm25_k1, bm25_b=c.bm25_b, qe_type="none",
        )
        for k, v in overrides.items():
            setattr(a, k, v)
        return a

    def _build_turn(self, utterance, history, qid, topic_id, user_id, ptkb_store) -> Turn:
        turn = Turn()
        turn.turn_id = qid
        turn.conversation_id = f"{topic_id}-{user_id}"
        turn.current_utterance = utterance
        turn.current_response = None
        turn.oracle_utterance = None
        # history is the interleaved [u1, r1, u2, r2, ...] content list (mirrors
        # evaluation_util.get_query_list fullconv_ctx). Everything BEFORE this turn.
        turn.fullconv_ctx = list(history)
        turn.context_utterances = list(history)
        if ptkb_store is not None and ptkb_store.base:
            turn.ptkb = {i: s for i, s in enumerate(ptkb_store.base, 1)}
        return turn

    def _retrieve_one(self, spec: RetrieverSpec, turn: Turn, qid: str,
                      context_turns: List[Turn]) -> List[Dict[str, List[Any]]]:
        """Return a LIST of hits dicts (one per query). Non-QR leg -> 1 query; a QR leg ->
        the rewriter's query list (>=1; GtR returns phi)."""
        c = self.config
        if spec.qr and spec.qr != "none":
            queries = self._rewriter.rewrite(turn, spec.qr, context_turns, turn.ptkb or {})
            is_conv = False                      # QR rewrites are plain query strings
        else:
            a0 = self._make_args(retrieval_model=spec.name, retrieval_query_type=spec.query_type)
            queries = [turn.query_type_2_query(spec.query_type, 0, 0.0, a0)]
            is_conv = (spec.query_type == "full_conversation_dense")

        out: List[Dict[str, List[Any]]] = []
        for q in queries:
            if spec.name == "BM25":
                res = self._bm25.batch_search([q], [qid], k=c.retrieval_top_k, threads=40)
                out.append({qid: list(res.get(qid, []))})
            elif spec.name in ("ance", "conv-ance", "qwen3", "conv-qwen3"):
                # a QR rewrite is a plain string -> plain encode path (NOT full_conversation_dense,
                # which expects a JSON turn-list). Keep the conv type only for non-QR conv legs.
                a = self._make_args(retrieval_model=spec.name,
                                    retrieval_query_type=(spec.query_type if is_conv else "raw"),
                                    dense_query_encoder_path=(spec.encoder_path
                                                              or c.dense_query_encoder_path))
                a.retrieval_query_list = [q]
                a.qid_list_string = [qid]
                emb, emb2id = get_test_query_embedding(a)
                D, I = search_query_against_ram(emb, self._ram, self._faiss, c.retrieval_top_k,
                                                gpus=list(range(c.faiss_n_gpu)))
                out.append(get_dense_ranking_list(emb2id, D, I, c.retrieval_top_k))
            elif spec.name == "splade_v3":
                # learned-sparse leg: encode q (the rar rewrite) into a SPLADE vocab vector, then
                # score the inverted index (CPU/numba, GIL released -> runs concurrent with the dense
                # GPU leg). retrieve() returns RRF-ready {qid: [PyScoredDoc]}.
                from apcir.search.splade_search import splade_encode_query
                sp_dev = f"cuda:{c.query_gpu_id}" if c.query_gpu_id >= 0 else "cpu"
                q_rep = splade_encode_query(q, c.splade_query_encoder_path, sp_dev)
                _, sp_hits = self._splade.retrieve({qid: q_rep})
                out.append({qid: sp_hits[str(qid)]})
            else:
                raise NotImplementedError(f"retriever {spec.name} not wired in interactive pipeline")
        return out

    # dense retriever leg names (encode a query into the frozen doc-embedding space).
    _DENSE_NAMES = ("ance", "conv-ance", "qwen3", "conv-qwen3")

    def _dense_group_search(self, dense_specs: List[RetrieverSpec], turn: Turn,
                            qid: str) -> List[Dict[str, List[Any]]]:
        """Shared-corpus dense search for a GROUP of NON-QR dense legs that share ONE index:
        encode each leg's query with its OWN (cached) encoder -> stack to (K,dim) -> ONE pass over
        the corpus (`search_query_against_ram` scores all K rows per block) -> split into K per-leg
        hits dicts. The corpus read dominates, so K legs ~= 1 leg. All legs here target self._ram
        (the single loaded index); a multi-index future would group by index and call this once per
        group. NOTE: legs in a group MUST share the index/dim (e.g. conv-qwen3 + pers-conv-qwen3,
        both 1024-d on the qwen index); a different index (e.g. ANCE 768-d) cannot share — it needs
        its own RamBlockSource + its own group."""
        c = self.config
        if not dense_specs:
            return []
        vecs = []
        for spec in dense_specs:
            enc = spec.encoder_path or c.dense_query_encoder_path
            is_conv = (spec.query_type == "full_conversation_dense")
            a0 = self._make_args(retrieval_model=spec.name, retrieval_query_type=spec.query_type,
                                 dense_query_encoder_path=enc)
            q = turn.query_type_2_query(spec.query_type, 0, 0.0, a0)
            a = self._make_args(retrieval_model=spec.name,
                                retrieval_query_type=(spec.query_type if is_conv else "raw"),
                                dense_query_encoder_path=enc)
            a.retrieval_query_list = [q]
            a.qid_list_string = [qid]
            emb, _ = get_test_query_embedding(a)              # (1, dim)
            vecs.append(np.asarray(emb, dtype=np.float32))
        dim0 = vecs[0].shape[1]
        assert all(v.shape[1] == dim0 for v in vecs), (
            "dense-group legs must share the embedding dim / index space — cannot mix e.g. "
            "ANCE 768-d + qwen 1024-d in one shared corpus search")
        Q_all = np.concatenate(vecs, axis=0)                 # (K, dim) — one row per leg
        D, I = search_query_against_ram(Q_all, self._ram, self._faiss, c.retrieval_top_k,
                                        gpus=list(range(c.faiss_n_gpu)))
        # one hits dict per leg (this group is only used in the order-insensitive RRF path).
        return [get_dense_ranking_list([qid], D[k:k + 1], I[k:k + 1], c.retrieval_top_k)
                for k in range(len(dense_specs))]

    def _rerank(self, ranked: List[Any], turn) -> List[Any]:
        """Step 2.5: Qwen3-Reranker over the fused top rerank_top_k; the tail [k:] keeps
        its original order behind the reranked head. Instruction routing mirrors
        rerank.py: the conversational field -> custom instruction + profile-first query
        built live from the Turn; any other reranking_query_type is treated as a
        reformulation name (e.g. an online-QR '_rw') -> NATIVE instruction + that rewrite."""
        from apcir.search.rerank import (
            QWEN3_RERANK_CONV_INSTRUCTION, QWEN3_RERANK_DEFAULT_INSTRUCTION)
        c = self.config
        rqt = c.reranking_query_type
        if rqt == "qwen_3_rerank_instruct_full":
            instruction = QWEN3_RERANK_CONV_INSTRUCTION
            query = turn.query_type_2_query(rqt, 0, 0.0, self._make_args(
                retrieval_model="none", retrieval_query_type=rqt))
        else:
            instruction = QWEN3_RERANK_DEFAULT_INSTRUCTION
            ref = turn.find_reformulation(rqt)
            query = ref.reformulated_query if ref is not None else turn.current_utterance

        head = ranked[:c.rerank_top_k]
        docs = [self._passage_text(d.docid) for d in head]
        scores = self._reranker.score(instruction, query, docs, c.rerank_batch_size)
        order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        reranked_head = [head[i] for i in order]
        # rewrite scores as 1/rank so head/tail stay consistently ordered (same
        # convention as rerank.py) without inventing comparable raw scores
        out = reranked_head + ranked[c.rerank_top_k:]
        for rank, d in enumerate(out):
            d.score = 1.0 / (rank + 1)
        return out

    def _fuse(self, hits_list: List[Dict[str, List[Any]]], qid: str) -> Dict[str, List[Any]]:
        c = self.config
        if len(hits_list) == 1:
            return hits_list[0]
        ft = c.fusion_type
        if ft == "RRF":
            return fuse_mod.RRF(hits_list, c.retrieval_top_k, k=c.rrf_k)
        if ft == "round_robin":
            return fuse_mod.round_robin_fusion(hits_list, c.retrieval_top_k, c.seed)
        if ft == "concat":
            return fuse_mod.concat(hits_list)
        if ft == "linear_combination":
            weights = c.fuse_weights or [1.0 / len(hits_list)] * len(hits_list)
            return fuse_mod.per_query_linear_combination(hits_list, {qid: weights}, c.retrieval_top_k)
        raise ValueError(f"unknown fusion_type {ft!r}")

    def _passage_text(self, docid: str) -> str:
        try:
            return load_document_by_id(docid, self._docfetch)["contents"]
        except Exception:
            return ""

    def _extractive_response(self, ranked: List[Any]) -> str:
        c = self.config
        parts, seen = [], set()
        for d in ranked[:c.generation_top_k]:
            if d.docid in seen:
                continue
            seen.add(d.docid)
            txt = self._passage_text(d.docid).strip()
            if txt:
                parts.append(txt)
        return _truncate_tokens(" ".join(parts), c.response_max_tokens)


# --------------------------------------------------------------------------- #
# Token-budget truncation (response must be <=max spaCy v3.3 tokens)
# --------------------------------------------------------------------------- #
_SPACY = None
_SPACY_TRIED = False


def _get_spacy():
    global _SPACY, _SPACY_TRIED
    if not _SPACY_TRIED:
        _SPACY_TRIED = True
        try:
            import spacy
            from spacy.lang.en import English
            _SPACY = English()  # blank tokenizer (matches iKAT's spaCy token COUNT)
        except Exception:
            _SPACY = None
    return _SPACY


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` to <= max_tokens. Uses the spaCy tokenizer (the iKAT counter)
    if available; else a conservative whitespace cap (spaCy splits punctuation, so it
    counts >= whitespace tokens -> apply a safety factor)."""
    if not text:
        return text
    nlp = _get_spacy()
    if nlp is not None:
        toks = list(nlp(text))
        if len(toks) <= max_tokens:
            return text
        return text[: toks[max_tokens].idx].rstrip()
    # fallback: whitespace, with safety factor (spaCy count >= whitespace count)
    words = text.split()
    cap = int(max_tokens * 0.6)
    if len(words) <= cap:
        return text
    return " ".join(words[:cap])
