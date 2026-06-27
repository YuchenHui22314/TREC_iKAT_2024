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


def test_set_active_validates_loader_before_evicting():
    # reranker_qwen3 has no loader yet; requesting it (which would evict the resident mini) must
    # raise BEFORE the eviction, leaving the current resident set intact.
    p = _pipe(200.0)
    p.set_active(["qrecc_ance_mini"])
    try:
        p.set_active(["reranker_qwen3"])     # plan: evict mini, load reranker (unsupported kind)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
    assert p.resident() == {"qrecc_ance_mini"}    # mini was NOT evicted


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


def test_can_serve_false_until_unit_resident():
    from apcir.interactive.pipeline import RunSpec, RetrieverSpec
    p = _pipe(200.0)
    rs = RunSpec(retrievers=[RetrieverSpec("qwen3", "raw", unit="qrecc_ance_mini")])
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
