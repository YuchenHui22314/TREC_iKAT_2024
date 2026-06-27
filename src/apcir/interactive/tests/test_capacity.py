import os

from apcir.interactive.capacity import (
    IndexFootprint, IndexRegistry, CapacityManager, CapacityPlan,
)


def _reg():
    return IndexRegistry({
        "dense_qwen": IndexFootprint("dense_qwen", "dense", 245.0, 330.0,
                                     index_dir="/x/qwen", dtype="float16"),
        "bm25": IndexFootprint("bm25", "sparse", 8.0, 8.0, index_dir="/x/bm25"),
        "reranker_qwen3": IndexFootprint("reranker_qwen3", "reranker", 1.0, 1.0, vram_gb=9.0),
    })


def _mgr(free_ram_gb, reg=None):
    return CapacityManager(reg or _reg(), free_ram_fn=lambda: free_ram_gb,
                           free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])


# --------------------------------------------------------------------------- #
# Task 1 — IndexRegistry
# --------------------------------------------------------------------------- #
def test_registry_get_returns_footprint():
    reg = _reg()
    fp = reg.get("dense_qwen")
    assert fp.kind == "dense"
    assert fp.resident_ram_gb == 245.0
    assert fp.load_peak_ram_gb == 330.0


def test_registry_get_unknown_raises_with_known_names():
    reg = _reg()
    try:
        reg.get("nope")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "dense_qwen" in str(e)


def test_registry_names_lists_all():
    assert set(_reg().names()) == {"dense_qwen", "bm25", "reranker_qwen3"}


# --------------------------------------------------------------------------- #
# Task 2 — CapacityManager.plan (active-set + RAM load-peak)
# --------------------------------------------------------------------------- #
def test_plan_fits_on_empty_octal40():
    plan = _mgr(480.0).plan(active_set=["dense_qwen", "bm25"], resident=[])
    assert plan.fits
    assert set(plan.to_load) == {"dense_qwen", "bm25"}
    assert plan.to_unload == []
    assert plan.projected_resident_ram_gb == 253.0


def test_plan_refuses_qwen_on_octal31_by_load_peak():
    plan = _mgr(237.0).plan(active_set=["dense_qwen"], resident=[])
    assert not plan.fits
    assert "dense_qwen" in plan.reason and "330" in plan.reason


def test_plan_evicts_then_loads_active_set():
    plan = _mgr(200.0).plan(active_set=["dense_qwen"], resident=["bm25"])
    assert plan.to_unload == ["bm25"]
    assert plan.to_load == ["dense_qwen"]
    assert not plan.fits


def test_plan_evict_makes_it_fit():
    reg = IndexRegistry({
        "dense_qwen": IndexFootprint("dense_qwen", "dense", 245.0, 330.0),
        "dense_ance": IndexFootprint("dense_ance", "dense", 184.0, 220.0),
    })
    plan = _mgr(100.0, reg).plan(active_set=["dense_ance"], resident=["dense_qwen"])
    assert plan.to_unload == ["dense_qwen"] and plan.to_load == ["dense_ance"]
    assert plan.fits


def test_plan_noop_when_already_resident():
    plan = _mgr(50.0).plan(active_set=["bm25"], resident=["bm25"])
    assert plan.to_load == [] and plan.to_unload == [] and plan.fits


# --------------------------------------------------------------------------- #
# Task 3 — VRAM placement for GPU-resident components
# --------------------------------------------------------------------------- #
def test_plan_reranker_fits_on_a_gpu():
    mgr = CapacityManager(_reg(), free_ram_fn=lambda: 100.0,
                          free_vram_fn=lambda: [2.0, 12.0])
    plan = mgr.plan(active_set=["reranker_qwen3"], resident=[])
    assert plan.fits


def test_plan_reranker_refused_when_no_gpu_room():
    mgr = CapacityManager(_reg(), free_ram_fn=lambda: 100.0,
                          free_vram_fn=lambda: [2.0, 6.0])
    plan = mgr.plan(active_set=["reranker_qwen3"], resident=[])
    assert not plan.fits
    assert "VRAM" in plan.reason


# --------------------------------------------------------------------------- #
# Task 4 — capacity_config.yaml
# --------------------------------------------------------------------------- #
def test_capacity_config_yaml_loads_and_has_expected_units():
    path = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")
    reg = IndexRegistry.from_yaml(path)
    for unit in ("clueweb_qwen", "qrecc_ance", "qrecc_qwen", "qrecc_ance_mini", "reranker_qwen3"):
        fp = reg.get(unit)
        assert fp.load_peak_ram_gb >= fp.resident_ram_gb >= 0
    # clueweb qwen's load peak must exceed octal31's ~237G free -> guard refuses it here
    assert reg.get("clueweb_qwen").load_peak_ram_gb > 237
    # qrecc ance fits comfortably under octal31's free RAM
    assert reg.get("qrecc_ance").load_peak_ram_gb < 237
    assert reg.get("reranker_qwen3").vram_gb > 0


def test_plan_on_octal31_config_refuses_clueweb_qwen_but_fits_qrecc_ance():
    path = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")
    reg = IndexRegistry.from_yaml(path)
    mgr = CapacityManager(reg, free_ram_fn=lambda: 237.0,
                          free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
    assert not mgr.plan(active_set=["clueweb_qwen"], resident=[]).fits
    assert mgr.plan(active_set=["qrecc_ance"], resident=[]).fits
    # switching from qrecc_ance to qrecc_qwen evicts the first, loads the second
    p = mgr.plan(active_set=["qrecc_qwen"], resident=["qrecc_ance"])
    assert p.to_unload == ["qrecc_ance"] and p.to_load == ["qrecc_qwen"] and p.fits
