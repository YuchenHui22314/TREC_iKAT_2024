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


def test_footprint_resolves_first_existing_dir_then_falls_back():
    """resolved_index_dir() picks the first existing of index_dir then index_dir_alts (fast SSD over
    NFS); is_available is True iff any candidate exists; missing-everywhere keeps the primary path."""
    import tempfile
    with tempfile.TemporaryDirectory() as real:
        # primary missing -> NFS fallback wins
        fp = IndexFootprint("u", "dense", 1.0, 1.0, index_dir="/no/such/primary",
                            index_dir_alts=[real])
        assert fp.resolved_index_dir() == real
        assert fp.is_available is True
        # primary exists -> use it (don't fall back)
        fp2 = IndexFootprint("u", "dense", 1.0, 1.0, index_dir=real, index_dir_alts=["/no/such"])
        assert fp2.resolved_index_dir() == real
        # none exist -> keep the primary (for reporting) + unavailable
        fp3 = IndexFootprint("u", "dense", 1.0, 1.0, index_dir="/no/a", index_dir_alts=["/no/b"])
        assert fp3.resolved_index_dir() == "/no/a"
        assert fp3.is_available is False
        # no dir needed (reranker) -> always available
        assert IndexFootprint("r", "reranker", 1.0, 1.0).is_available is True


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


def test_plan_dedups_active_and_resident():
    # duplicate names must not double-count projected RAM or schedule duplicate loads
    plan = _mgr(480.0).plan(active_set=["dense_qwen", "dense_qwen", "bm25"], resident=[])
    assert plan.to_load == ["dense_qwen", "bm25"]
    assert plan.projected_resident_ram_gb == 253.0


def test_plan_vram_credits_eviction():
    # one GPU, 6G free; evict a 9G reranker (frees its VRAM) then load a 9G reranker -> fits
    reg = IndexRegistry({
        "r_old": IndexFootprint("r_old", "reranker", 1.0, 1.0, vram_gb=9.0),
        "r_new": IndexFootprint("r_new", "reranker", 1.0, 1.0, vram_gb=9.0),
    })
    mgr = CapacityManager(reg, free_ram_fn=lambda: 100.0, free_vram_fn=lambda: [6.0])
    plan = mgr.plan(active_set=["r_new"], resident=["r_old"])
    assert plan.to_unload == ["r_old"] and plan.to_load == ["r_new"]
    assert plan.fits  # 6 free + 9 evicted = 15 >= 9 + 1 safety


def test_plan_vram_credit_not_false_accepting_multigpu():
    # codex #4: free [8,1,1] with two 9G units on the two near-full GPUs; evict both, load 20G.
    # Optimistic "credit all evicted vram to the most-free GPU" would model [26,1,1] and ACCEPT,
    # but the real post-evict state is [8,10,10] where NO single GPU holds 20G. Must NOT accept.
    reg = IndexRegistry({
        "r1": IndexFootprint("r1", "reranker", 1.0, 1.0, vram_gb=9.0),
        "r2": IndexFootprint("r2", "reranker", 1.0, 1.0, vram_gb=9.0),
        "big": IndexFootprint("big", "reranker", 1.0, 1.0, vram_gb=20.0),
    })
    mgr = CapacityManager(reg, free_ram_fn=lambda: 100.0, free_vram_fn=lambda: [8.0, 1.0, 1.0])
    plan = mgr.plan(active_set=["big"], resident=["r1", "r2"])
    assert not plan.fits


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


def test_registry_parses_query_encoders_list():
    """capacity_config.yaml may declare per-unit query_encoders (list of {label,path,leg_name,
    default_query_type}); from_yaml must accept it (dataclass field exists)."""
    import tempfile, os, yaml as _yaml
    cfg = {"indexes": {"u": {
        "kind": "dense", "resident_ram_gb": 1.0, "load_peak_ram_gb": 1.0,
        "query_encoder": "/enc/base",
        "query_encoders": [{"label": "ANCE", "path": "/enc/base", "leg_name": "ance",
                            "default_query_type": "raw"}],
    }}}
    with tempfile.TemporaryDirectory() as d:
        pth = os.path.join(d, "c.yaml")
        with open(pth, "w") as f:
            _yaml.safe_dump(cfg, f)
        reg = IndexRegistry.from_yaml(pth)
    assert reg.get("u").query_encoders[0]["label"] == "ANCE"


# --------------------------------------------------------------------------- #
# Load modes (ram_fp16 / gpu_resident / pq_refine) + per-GPU placement
# --------------------------------------------------------------------------- #
def _dense_fp(name="u", ram=40.0, peak=52.0, docs=26_000_000, pq=None):
    from apcir.interactive.capacity import IndexFootprint
    return IndexFootprint(name, "dense", ram, peak, index_dir="/nonexistent",
                          dtype="float16", embed_dim=768, block_num=26,
                          num_docs=docs, pq_index_path=pq)


def test_mode_requirements_ram_fp16_matches_legacy():
    fp = _dense_fp()
    r = fp.mode_requirements("ram_fp16")
    assert r["ram_gb"] == 40.0 and r["load_peak_ram_gb"] == 52.0
    assert r["vram_total_gb"] == 0 and r["vram_single_gb"] == 0


def test_mode_requirements_gpu_resident_moves_store_to_vram():
    r = _dense_fp().mode_requirements("gpu_resident")
    assert r["vram_total_gb"] == 40.0          # fp16 store lives on the GPUs, sharded
    assert r["ram_gb"] < 40.0                  # nothing resident in RAM (stream-through load)


def test_mode_requirements_pq_refine_needs_ram_store_plus_one_gpu():
    r = _dense_fp().mode_requirements("pq_refine")
    assert r["ram_gb"] == 40.0                 # fp16 rescore store stays in RAM
    # PQ64 codes: 26M x 64B ~ 1.7G, with overhead < 3G, on ONE gpu
    assert 1.0 < r["vram_single_gb"] < 3.0
    assert r["vram_total_gb"] == 0


def test_plan_pq_refine_refused_without_gpu_room():
    from apcir.interactive.capacity import IndexRegistry, CapacityManager
    fp = _dense_fp()
    reg = IndexRegistry({"u": fp})
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0,
                          free_vram_fn=lambda: [1.0, 1.0])       # no GPU fits ~2G + safety
    plan = cap.plan(["u"], [], modes={"u": "pq_refine"})
    assert not plan.fits and "VRAM" in plan.reason


def test_plan_gpu_resident_shards_across_gpus():
    from apcir.interactive.capacity import IndexRegistry, CapacityManager
    reg = IndexRegistry({"u": _dense_fp(ram=40.0, peak=52.0)})
    cap = CapacityManager(reg, free_ram_fn=lambda: 60.0,
                          free_vram_fn=lambda: [12.0, 12.0, 12.0, 12.0])  # 40G fits only sharded x4
    assert cap.plan(["u"], [], modes={"u": "gpu_resident"}).fits
    cap2 = CapacityManager(reg, free_ram_fn=lambda: 60.0,
                           free_vram_fn=lambda: [12.0, 12.0])            # 2x12 < 40 -> refuse
    plan = cap2.plan(["u"], [], modes={"u": "gpu_resident"})
    assert not plan.fits and "VRAM" in plan.reason


def test_allocator_places_and_credits_eviction():
    """Multi-GPU eviction credit via per-unit placement tracking (the documented TODO)."""
    from apcir.interactive.capacity import IndexRegistry, CapacityManager
    a = _dense_fp("a", ram=10, peak=12, docs=200_000_000)   # PQ64 ~ 15G -> one GPU
    b = _dense_fp("b", ram=10, peak=12, docs=200_000_000)
    reg = IndexRegistry({"a": a, "b": b})
    free = [17.0, 10.0]
    cap = CapacityManager(reg, free_ram_fn=lambda: 100.0, free_vram_fn=lambda: list(free))
    got = cap.allocate_gpus("a", "pq_refine")
    assert len(got) == 1 and got[0][0] == 0               # placed on the roomier GPU 0
    free[0] -= got[0][1]                                  # simulate the load consuming VRAM
    # b (~15G) fits neither GPU now (gpu0 ~1G, gpu1 10G) -> refused when a stays...
    assert not cap.plan(["a", "b"], ["a"], modes={"a": "pq_refine", "b": "pq_refine"}).fits
    # ...but replacing a with b fits: the plan credits a's TRACKED placement on gpu0.
    assert cap.plan(["b"], ["a"], modes={"b": "pq_refine"}).fits
