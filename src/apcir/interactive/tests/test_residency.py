"""Integration tests for dynamic residency (set_active load/unload) against a REAL index.

Uses the tiny `qrecc_ance_mini` unit (1 block, 1000 vecs) so the load is instant and needs no
GPU. Exercises the real RamBlockSource construct->resident->drop mechanism + the capacity guard.
Run from src/ in the trec_ikat env:
  python -m pytest apcir/interactive/tests/test_residency.py -v
"""
import os

from apcir.interactive.capacity import IndexRegistry, CapacityManager, CapacityError
from apcir.interactive.pipeline import InteractivePipeline, PipelineConfig

CONFIG_YAML = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")


def _pipe(free_ram_gb):
    reg = IndexRegistry.from_yaml(CONFIG_YAML)
    cap = CapacityManager(reg, free_ram_fn=lambda: free_ram_gb,
                          free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
    return InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)


def _custom_pipe(footprints):
    reg = IndexRegistry(footprints)
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0,
                          free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
    return InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)


def test_encoder_for_prefers_unit_footprint_over_global():
    """A unit-routed dense leg with no explicit encoder_path must use the unit's OWN query_encoder
    (from its footprint), NOT the single global default — else an index built with encoder A is
    queried with the global encoder B and returns garbage scores."""
    from apcir.interactive.capacity import IndexFootprint
    from apcir.interactive.pipeline import RetrieverSpec
    p = _custom_pipe({
        "ance_unit": IndexFootprint("ance_unit", "dense", 0.01, 0.02, index_dir="/x",
                                    query_encoder="/enc/ance"),
        "qwen_unit": IndexFootprint("qwen_unit", "dense", 0.01, 0.02, index_dir="/y",
                                    query_encoder="/enc/qwen"),
        "bare_unit": IndexFootprint("bare_unit", "dense", 0.01, 0.02, index_dir="/z"),
    })
    cfg = p.config
    # unit-routed, no explicit encoder_path -> the unit's footprint encoder
    assert p._encoder_for(RetrieverSpec("ance", "raw", unit="ance_unit"), cfg) == "/enc/ance"
    assert p._encoder_for(RetrieverSpec("qwen3", "raw", unit="qwen_unit"), cfg) == "/enc/qwen"
    # an explicit per-leg encoder_path still wins over the footprint
    assert p._encoder_for(
        RetrieverSpec("ance", "raw", unit="ance_unit", encoder_path="/explicit"), cfg) == "/explicit"
    # a unit with NO query_encoder, or a legacy leg with no unit -> the global default
    assert p._encoder_for(RetrieverSpec("ance", "raw", unit="bare_unit"), cfg) \
        == cfg.dense_query_encoder_path
    assert p._encoder_for(RetrieverSpec("BM25", "raw"), cfg) == cfg.dense_query_encoder_path


def test_models_status_surfaces_query_encoder():
    """GET /models must report each dense unit's query_encoder so an operator/client can see which
    encoder a unit needs."""
    from apcir.interactive.capacity import IndexFootprint
    p = _custom_pipe({
        "u": IndexFootprint("u", "dense", 0.01, 0.02, index_dir="/x", corpus="qrecc",
                            query_encoder="/enc/q"),
    })
    unit = next(x for x in p.models_status()["units"] if x["name"] == "u")
    assert unit["query_encoder"] == "/enc/q"
    assert unit["corpus"] == "qrecc"


def test_should_build_remote_llm_gating():
    """The resource-free OpenAI LLM is built eagerly (so rag/QR work in dynamic mode WITHOUT a
    resident 'llm' capacity unit); local_vllm is NOT (needs an explicit GPU boot); extractive+no-QR
    needs no LLM; a built LLM is never rebuilt."""
    def pipe(cfg):
        reg = IndexRegistry.from_yaml(CONFIG_YAML)
        cap = CapacityManager(reg, free_ram_fn=lambda: 200.0,
                              free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
        return InteractivePipeline(cfg, registry=reg, capacity=cap)
    assert pipe(PipelineConfig(llm_backend="openai", generation="rag"))._should_build_remote_llm() is True
    assert pipe(PipelineConfig(llm_backend="local_vllm", generation="rag"))._should_build_remote_llm() is False
    assert pipe(PipelineConfig(llm_backend="openai", generation="extractive"))._should_build_remote_llm() is False
    p = pipe(PipelineConfig(llm_backend="openai", generation="rag"))
    p._llm = object()                       # already built -> never rebuild
    assert p._should_build_remote_llm() is False


def test_set_active_loads_and_unloads_mini_dense():
    p = _pipe(200.0)
    plan = p.set_active(["qrecc_ance_mini"])
    assert plan.fits
    assert plan.to_load == ["qrecc_ance_mini"]
    assert p.resident() == {"qrecc_ance_mini"}
    ram = p._dense["qrecc_ance_mini"]
    assert ram.total_vecs == 1000
    assert str(ram.store_dtype) == "float16"
    # switch to empty active set -> evict
    plan2 = p.set_active([])
    assert plan2.to_unload == ["qrecc_ance_mini"]
    assert p.resident() == set()
    assert "qrecc_ance_mini" not in p._dense


def test_set_active_refuses_clueweb_qwen_on_small_host():
    p = _pipe(237.0)
    try:
        p.set_active(["clueweb_qwen"])
        assert False, "expected CapacityError (qwen peak > 237G free)"
    except CapacityError as e:
        assert "clueweb_qwen" in str(e)
    assert p.resident() == set()


def test_set_active_dispatches_splade_reranker_llm():
    # verify the kind->loader dispatch (mock the heavy loads so no real models are needed)
    from apcir.interactive.capacity import IndexFootprint
    reg = IndexRegistry({
        "sp": IndexFootprint("sp", "splade", 1.0, 1.0, index_dir="/x"),
        "rr": IndexFootprint("rr", "reranker", 1.0, 1.0, vram_gb=2.0),
        "lm": IndexFootprint("lm", "llm", 1.0, 1.0, vram_gb=2.0),
    })
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    calls = []
    p.load_splade = lambda u, cb=None: (calls.append(("splade", u)), p._resident.add(u))
    p.load_reranker = lambda u, cb=None: (calls.append(("reranker", u)), p._resident.add(u))
    p.load_llm = lambda u, cb=None: (calls.append(("llm", u)), p._resident.add(u))
    p.set_active(["sp", "rr", "lm"])
    assert ("splade", "sp") in calls and ("reranker", "rr") in calls and ("llm", "lm") in calls
    assert p.resident() == {"sp", "rr", "lm"}


def test_can_serve_generation_needs_docfetch():
    from apcir.interactive.pipeline import RunSpec, RetrieverSpec
    p = _pipe(200.0)
    p.set_active(["qrecc_ance_mini"])           # dense only -> no doc-fetch loaded
    rs = RunSpec(retrievers=[RetrieverSpec("qwen3", "raw", unit="qrecc_ance_mini")],
                 generation="extractive", reranker="none")
    ok, reason = p.can_serve(rs)
    assert not ok and "doc-fetch" in reason.lower()   # extractive/RAG needs passage text


def test_set_active_rejects_two_singleton_units():
    from apcir.interactive.capacity import IndexFootprint
    reg = IndexRegistry({
        "bm25a": IndexFootprint("bm25a", "sparse", 0.5, 0.5, index_dir="/x", corpus="c"),
        "bm25b": IndexFootprint("bm25b", "sparse", 0.5, 0.5, index_dir="/y", corpus="c"),
    })
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    try:
        p.set_active(["bm25a", "bm25b"])         # two sparse singletons -> reject
        assert False, "expected ValueError"
    except ValueError as e:
        assert "sparse" in str(e)


def test_set_active_rejects_mixed_corpora():
    from apcir.interactive.capacity import IndexFootprint
    reg = IndexRegistry({
        "d_q": IndexFootprint("d_q", "dense", 1.0, 1.0, index_dir="/x", corpus="qrecc",
                              embed_dim=768, block_num=1),
        "d_c": IndexFootprint("d_c", "dense", 1.0, 1.0, index_dir="/y", corpus="clueweb",
                              embed_dim=1024, block_num=1),
    })
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    try:
        p.set_active(["d_q", "d_c"])             # mixed corpora -> reject
        assert False, "expected ValueError"
    except ValueError as e:
        assert "corpus" in str(e).lower()


def test_can_serve_qr_needs_llm():
    from apcir.interactive.pipeline import RunSpec, RetrieverSpec
    p = _pipe(200.0)
    p.set_active(["qrecc_ance_mini", "qrecc_bm25"])    # dense + doc-fetch, but NO LLM
    rs = RunSpec(retrievers=[RetrieverSpec("qwen3", "raw", unit="qrecc_ance_mini", qr="rar")],
                 generation="none", reranker="none")
    ok, reason = p.can_serve(rs)
    assert not ok and "qr" in reason.lower()           # online QR needs the LLM


def test_set_active_loads_sparse_bm25_and_docfetch():
    p = _pipe(200.0)
    p.set_active(["qrecc_bm25"])
    assert p.resident() == {"qrecc_bm25"}
    assert p._bm25 is not None and p._docfetch is not None   # BM25 search + passage doc-fetch
    p.set_active([])
    assert p._bm25 is None and p._docfetch is None
    assert p.resident() == set()


def test_set_active_validates_loader_before_evicting():
    # an UNSUPPORTED kind must raise BEFORE evicting the current resident set.
    from apcir.interactive.capacity import IndexFootprint
    mini = "/part/01/Tmp/yuchenhui/indexes/qrecc_ance_mini_merged"
    reg = IndexRegistry({
        "mini": IndexFootprint("mini", "dense", 0.01, 0.02, index_dir=mini, dtype="float16",
                               embed_dim=768, block_num=1),
        "bogus": IndexFootprint("bogus", "weird_kind", 1.0, 1.0),
    })
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    p.set_active(["mini"])
    try:
        p.set_active(["bogus"])              # unsupported kind -> raise BEFORE evicting mini
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    assert p.resident() == {"mini"}          # mini was NOT evicted


def test_set_active_swallows_progress_callback_errors():
    # codex #5: a raising progress callback must not abort the load or corrupt residency
    p = _pipe(200.0)

    def bad_cb(msg, frac):
        raise RuntimeError("ui blew up")

    p.set_active(["qrecc_ance_mini"], progress_cb=bad_cb)
    assert p.resident() == {"qrecc_ance_mini"}     # loaded despite the callback raising


def test_set_active_rolls_back_partial_load_on_failure():
    from apcir.interactive.capacity import IndexFootprint
    mini = "/part/01/Tmp/yuchenhui/indexes/qrecc_ance_mini_merged"
    reg = IndexRegistry({
        "good": IndexFootprint("good", "dense", 0.01, 0.02, index_dir=mini,
                               dtype="float16", embed_dim=768, block_num=1),
        "bad": IndexFootprint("bad", "dense", 0.01, 0.02, index_dir="/nonexistent/dir",
                              dtype="float16", embed_dim=768, block_num=1),
    })
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    try:
        p.set_active(["good", "bad"])        # good loads, bad's dir is missing -> load raises
        assert False, "expected load failure"
    except Exception:
        pass
    assert p.resident() == set()             # rolled back: good unloaded, bad never resident
    assert p._dense == {}


def test_effective_config_applies_runspec_overrides():
    from apcir.interactive.pipeline import RunSpec, RetrieverSpec
    p = _pipe(200.0)
    rs = RunSpec(retrievers=[RetrieverSpec("ance", "raw", unit="qrecc_ance_mini")],
                 fusion_type="concat", reranker="none", generation="extractive")
    c = p._effective_config(rs)
    assert c.retrievers[0].unit == "qrecc_ance_mini"
    assert c.fusion_type == "concat"
    assert c.generation == "extractive"
    assert c.retrieval_top_k == p.config.retrieval_top_k   # untouched field -> config default
    assert p.config.fusion_type == "RRF"                   # original config NOT mutated


def test_effective_config_none_returns_config():
    p = _pipe(200.0)
    assert p._effective_config(None) is p.config


def test_extract_ptkb_keeps_first_person_facts():
    p = _pipe(200.0)

    class FakeLLM:
        def generate(self, prompt):
            return ("I am a vegetarian.\nI live in Montreal.\nThe weather is nice today.\nNONE", {})

    p._llm = FakeLLM()
    facts = p.extract_ptkb([], "vegan places near me?", "Here are some options...")
    assert facts == ["I am a vegetarian.", "I live in Montreal."]   # non-first-person/NONE dropped


def test_extract_ptkb_no_llm_returns_empty():
    p = _pipe(200.0)                          # no LLM loaded
    assert p.extract_ptkb([], "hi", "hello") == []


def test_parse_citations():
    from apcir.interactive.generation import parse_citations
    resp = "The sky is blue [1] and grass is green [2][3]. ignore array[1] here."
    docids = ["docA", "docB"]                 # only 2 passages; [3] out of range -> dropped
    cites = parse_citations(resp, docids)
    assert [c["docid"] for c in cites] == ["docA", "docB"]   # array[1] is NOT a citation
    assert cites[0]["n"] == 1
    assert resp[cites[0]["start"]:cites[0]["end"]] == "[1]"
    assert cites[0]["start"] == resp.index(" [1]") + 1       # the real [1], not array[1]
    assert all(c["n"] <= 2 for c in cites)                   # [3] dropped (no docid for it)


def test_leg_label():
    from apcir.interactive.pipeline import RetrieverSpec, InteractivePipeline as P
    assert P._leg_label(RetrieverSpec("BM25", "raw")) == "BM25"
    assert P._leg_label(RetrieverSpec("qwen3", "raw", unit="qrecc_ance")) == "qwen3@qrecc_ance"
    assert P._leg_label(RetrieverSpec("qwen3", "raw", qr="rar")) == "qwen3:rar"


def test_enrich_per_retriever_and_shared_docs():
    from apcir.interactive.pipeline import InteractivePipeline as P

    class D:
        def __init__(self, docid, score):
            self.docid, self.score = docid, score

    legs = [("bm25", {"q": [D("a", 1.0), D("b", 0.5)]}),
            ("qwen", {"q": [D("a", 0.9), D("c", 0.8)]})]
    per, shared = P._enrich(legs, "q", top_k=20)
    assert [e["retriever"] for e in per] == ["bm25", "qwen"]
    assert per[0]["hits"][0] == ["a", 1.0]                 # docid+score, JSON-friendly list
    assert shared == {"a": ["bm25", "qwen"]}               # "a" shared across both; b/c not


def test_can_serve_false_until_unit_resident():
    from apcir.interactive.pipeline import RunSpec, RetrieverSpec
    p = _pipe(200.0)
    # generation="none" -> retrieval-only, so this tests just retriever residency (not doc-fetch/LLM)
    rs = RunSpec(retrievers=[RetrieverSpec("qwen3", "raw", unit="qrecc_ance_mini")], generation="none")
    ok, reason = p.can_serve(rs)
    assert not ok and "qrecc_ance_mini" in reason
    p.set_active(["qrecc_ance_mini"])
    ok2, _ = p.can_serve(rs)
    assert ok2


def test_models_status_lists_units_and_live_state():
    p = _pipe(200.0)
    st = p.models_status()
    names = {u["name"] for u in st["units"]}
    assert {"qrecc_ance_mini", "clueweb_qwen", "qrecc_ance", "reranker_qwen3"} <= names
    assert st["resident"] == []
    assert st["free_ram_gb"] == 200.0
    assert st["free_vram_gb"] == [24.0, 24.0, 24.0, 24.0]
    mini = next(u for u in st["units"] if u["name"] == "qrecc_ance_mini")
    assert mini["available"] is True and mini["kind"] == "dense"


def test_process_turn_rejects_unavailable_reranker():
    # codex#3 #4: RunSpec asking for a reranker that isn't resident must fail fast, not silently skip
    from apcir.interactive.pipeline import RunSpec
    p = _pipe(200.0)                                  # no reranker loaded
    try:
        p.process_turn("q", [], run_spec=RunSpec(reranker="qwen3_reranker"))
        assert False, "expected RuntimeError (reranker not resident)"
    except RuntimeError as e:
        assert "reranker" in str(e).lower()


def test_fuse_honors_cfg_override():
    # the helpers must read the EFFECTIVE config, not self.config. A bogus fusion_type in cfg makes
    # _fuse raise; the default (self.config="RRF") would not -> proves the cfg param is used.
    from dataclasses import replace

    class D:
        def __init__(self, docid, score):
            self.docid, self.score = docid, score

    p = _pipe(200.0)
    hits = [{"q": [D("a", 1.0)]}, {"q": [D("b", 0.9)]}]   # 2 legs -> reaches the fusion dispatch
    assert "q" in p._fuse(hits, "q")                       # default RRF fuses fine
    try:
        p._fuse(hits, "q", cfg=replace(p.config, fusion_type="BOGUS_FUSION"))
        assert False, "expected ValueError for the bogus fusion_type from cfg"
    except ValueError as e:
        assert "BOGUS_FUSION" in str(e)
