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
NOTE: the ANCE query encoder is rebuilt inside get_test_query_embedding every turn
(a few hundred ms); v1 accepts this — refactor to a persistent encoder if turn latency
matters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from pyserini.search.lucene import LuceneSearcher

from apcir.functional.topics import Turn, load_document_by_id
from apcir.search.dense_search import (
    build_faiss_index, get_test_query_embedding, get_dense_ranking_list,
)
from apcir.search import fuse as fuse_mod
from .ram_index import RamBlockSource, search_query_against_ram
from .ptkb_store import PTKBStore


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RetrieverSpec:
    """One configured retriever leg of the fusion."""
    name: str                 # "BM25" | "ance" | "qwen3" | "splade_v3" ...
    query_type: str           # online-safe QR: "raw" | "full_conversation_dense" | "qwen_conversation"[_ptkb]


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
    # generation
    generation: str = "extractive"
    response_max_tokens: int = 512
    citations_max: int = 10
    generation_top_k: int = 3
    # dense (ANCE) index
    dense_index_dir_path: str = "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_ance_merged_2"
    dense_query_encoder_path: str = (
        "/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/"
        "snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234")
    embed_dim: int = 768
    passage_block_num: int = 12
    faiss_n_gpu: int = 4
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
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ram: Optional[RamBlockSource] = None
        self._faiss = None
        self._bm25: Optional[LuceneSearcher] = None
        self._docfetch: Optional[LuceneSearcher] = None
        self._needs_dense = any(r.name in ("ance", "conv-ance", "qwen3", "conv-qwen3")
                                for r in config.retrievers)
        self._needs_sparse = any(r.name == "BM25" for r in config.retrievers)

    # --- startup: load resident state once --------------------------------- #
    def load(self):
        c = self.config
        if self._needs_dense:
            self._ram = RamBlockSource(c.dense_index_dir_path, c.passage_block_num, c.embed_dim)
            self._faiss = build_faiss_index(self._make_args())
        if self._needs_sparse:
            self._bm25 = LuceneSearcher(c.sparse_index_dir_path)
            self._bm25.set_bm25(c.bm25_k1, c.bm25_b)
        # doc-fetch: passage text is stored in the lucene (sparse) index
        self._docfetch = LuceneSearcher(c.sparse_index_dir_path)

    def health(self) -> Dict[str, Any]:
        return {
            "dense_loaded": self._ram is not None,
            "dense_vectors": getattr(self._ram, "total_vecs", None),
            "sparse_loaded": self._bm25 is not None,
            "docfetch_loaded": self._docfetch is not None,
            "retrievers": [(r.name, r.query_type) for r in self.config.retrievers],
            "fusion_type": self.config.fusion_type,
        }

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

        # 1) retrieve per leg
        hits_list: List[Dict[str, List[Any]]] = []
        for spec in c.retrievers:
            hits = self._retrieve_one(spec, turn, qid)
            if c.fusion_type == "linear_combination":
                hits = fuse_mod.normalize_scores(hits, c.fusion_normalization)
            hits_list.append(hits)

        # 2) fuse
        fused = self._fuse(hits_list, qid)
        ranked = fused[qid]

        # 3) generate + citations
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

    def _retrieve_one(self, spec: RetrieverSpec, turn: Turn, qid: str) -> Dict[str, List[Any]]:
        c = self.config
        a = self._make_args(retrieval_model=spec.name, retrieval_query_type=spec.query_type)
        query = turn.query_type_2_query(spec.query_type, 0, 0.0, a)

        if spec.name == "BM25":
            res = self._bm25.batch_search([query], [qid], k=c.retrieval_top_k, threads=40)
            return {qid: list(res.get(qid, []))}

        if spec.name in ("ance", "conv-ance", "qwen3", "conv-qwen3"):
            a.retrieval_query_list = [query]
            a.qid_list_string = [qid]
            emb, emb2id = get_test_query_embedding(a)
            D, I = search_query_against_ram(emb, self._ram, self._faiss, c.retrieval_top_k)
            return get_dense_ranking_list(emb2id, D, I, c.retrieval_top_k)

        raise NotImplementedError(f"retriever {spec.name} not wired in the interactive pipeline")

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
