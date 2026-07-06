"""Search a RESIDENT dense unit end-to-end (fp16 GPU path) — proves set_active determines which
index is searched. Needs free GPU(s) (octal31 A5000s)."""
import os
import numpy as np

from apcir.interactive.capacity import IndexRegistry, CapacityManager
from apcir.interactive.pipeline import InteractivePipeline, PipelineConfig

CONFIG_YAML = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")


def _pipe(free_ram_gb=200.0):
    reg = IndexRegistry.from_yaml(CONFIG_YAML)
    cap = CapacityManager(reg, free_ram_fn=lambda: free_ram_gb,
                          free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])
    return InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)


def test_search_dense_unit_retrieves_self_from_resident_index():
    p = _pipe()
    p.set_active(["qrecc_ance_mini"])
    _bid, emb, ids = next(p._dense["qrecc_ance_mini"].iter_blocks())
    qi = [0, 137, 642]
    Q = np.ascontiguousarray(emb[qi], dtype=np.float16)      # doc vectors used as queries
    D, I = p.search_dense_unit("qrecc_ance_mini", Q, topN=5)
    assert D.shape == (3, 5) and I.shape == (3, 5)
    for k, di in enumerate(qi):
        assert ids[di] in set(I[k])                          # each doc retrieves itself (top-5)


def test_search_dense_unit_requires_resident():
    p = _pipe()
    try:
        p.search_dense_unit("qrecc_ance_mini", np.zeros((1, 768), np.float16), 5)
        assert False, "expected KeyError (unit not resident)"
    except KeyError:
        pass


def test_ram_for_routes_to_resident_unit():
    from apcir.interactive.pipeline import RetrieverSpec
    p = _pipe()
    p.set_active(["qrecc_ance_mini"])
    ram, faiss = p._ram_for(RetrieverSpec("ance", "raw", unit="qrecc_ance_mini"))
    assert ram is p._dense["qrecc_ance_mini"]
    assert faiss is None                      # fp16 unit -> no faiss


def test_ram_for_unresident_unit_raises():
    from apcir.interactive.pipeline import RetrieverSpec
    p = _pipe()
    try:
        p._ram_for(RetrieverSpec("ance", "raw", unit="qrecc_ance_mini"))
        assert False, "expected KeyError (unit not resident)"
    except KeyError:
        pass


def test_ram_for_no_unit_uses_legacy_ram():
    from apcir.interactive.pipeline import RetrieverSpec
    p = _pipe()
    ram, faiss = p._ram_for(RetrieverSpec("ance", "raw"))   # unit=None -> legacy single index
    assert ram is p._ram and faiss is p._faiss


def test_ram_for_rejects_non_fp16_unit():
    # codex#3 #2: dynamic units must be fp16 (index=None path); fp32 must fail clearly, not crash
    from apcir.interactive.capacity import IndexFootprint
    from apcir.interactive.pipeline import RetrieverSpec
    mini = "/part/01/Tmp/yuchenhui/indexes/qrecc_ance_mini_merged"
    reg = IndexRegistry({"mini32": IndexFootprint("mini32", "dense", 0.01, 0.02, index_dir=mini,
                                                  dtype="float32", embed_dim=768, block_num=1)})
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    p.set_active(["mini32"])
    try:
        p._ram_for(RetrieverSpec("ance", "raw", unit="mini32"))
        assert False, "expected ValueError for a non-fp16 dynamic unit"
    except ValueError as e:
        assert "fp16" in str(e)


def test_dense_group_search_rejects_mixed_units():
    # codex#3 #1: a shared-corpus group must target ONE unit; reject (not assert) mixed units
    from apcir.interactive.pipeline import RetrieverSpec
    p = _pipe()
    specs = [RetrieverSpec("ance", "raw", unit="a"), RetrieverSpec("ance", "raw", unit="b")]
    try:
        p._dense_group_search(specs, None, "q")
        assert False, "expected ValueError for mixed units"
    except ValueError as e:
        assert "unit" in str(e).lower()


def test_search_dense_unit_rejects_non_fp16():
    # codex #2: index=None is only valid for the fp16 GPU path; an fp32 unit must be rejected
    # with a clear error, not crash deep in ram_index's faiss branch.
    from apcir.interactive.capacity import IndexFootprint
    mini = "/part/01/Tmp/yuchenhui/indexes/qrecc_ance_mini_merged"
    reg = IndexRegistry({"mini32": IndexFootprint("mini32", "dense", 0.01, 0.02, index_dir=mini,
                                                  dtype="float32", embed_dim=768, block_num=1)})
    cap = CapacityManager(reg, free_ram_fn=lambda: 200.0, free_vram_fn=lambda: [24.0])
    p = InteractivePipeline(PipelineConfig(), registry=reg, capacity=cap)
    p.set_active(["mini32"])
    try:
        p.search_dense_unit("mini32", np.zeros((1, 768), np.float32), 5)
        assert False, "expected a clear fp16-only error"
    except (ValueError, NotImplementedError) as e:
        assert "fp16" in str(e) or "float16" in str(e)

class _TinySrc:
    """Minimal RamBlockSource stand-in: one fp16 block."""
    def __init__(self, emb, ids):
        self._emb = np.ascontiguousarray(emb, dtype=np.float16)
        self._ids = np.array(ids, dtype=object)
    def iter_blocks(self):
        yield 0, self._emb, self._ids


def test_fp16_search_scores_are_fp32_and_resolve_sub_ulp_ties():
    """At ANCE score magnitude (~700) fp16 ulp is 0.5: an fp16-output GEMM quantizes near-tied
    docs together and can rank them wrongly. Scores must be computed/emitted in fp32 so docs whose
    true IPs differ by < 0.5 still rank correctly."""
    from apcir.interactive.ram_index import _search_ram_fp16_gpu
    rng = np.random.default_rng(0)
    dim = 768
    q = rng.standard_normal((1, dim)).astype(np.float32)
    q /= np.linalg.norm(q)
    base = q[0] * 700.0                    # doc collinear with q -> IP ~= 700 (ANCE-like magnitude)
    # three docs with true IPs 700.0 / 700.35 / 700.70 — gaps BELOW the fp16 ulp (0.5) at this
    # magnitude, injected via ONE component so fp16 DOC storage still distinguishes the docs
    # (per-component change is large); only an fp16 SCORE cannot resolve the gap.
    j = int(np.argmax(np.abs(q[0])))
    docs = np.stack([base.copy() for _ in range(3)])
    for k in range(3):
        docs[k, j] += 0.35 * k / q[0, j]
    filler = rng.standard_normal((61, dim)).astype(np.float32)   # low-score fillers
    emb = np.concatenate([docs, filler])
    ids = [f"d{i}" for i in range(len(emb))]
    D, I = _search_ram_fp16_gpu(np.ascontiguousarray(q), _TinySrc(emb, ids), topN=3, gpus=[0])
    assert D.dtype == np.float32
    assert list(I[0]) == ["d2", "d1", "d0"]            # correct sub-ulp ordering
    assert D[0][0] > D[0][1] > D[0][2]                 # strictly descending fp32 scores


def test_int8_store_rescore_matches_fp32_ranking():
    """int8 rescore store (pq_refine): per-row symmetric quantization must preserve IP RANKING
    (scores within ~0.5% of fp32) — the octal31 ClueWeb-Qwen enabler (235G fp16 -> 119G int8)."""
    from apcir.interactive.ram_index import quantize_int8, dequantize_int8
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 256)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)              # qwen-like normalized rows
    q, scales = quantize_int8(X)
    assert q.dtype == np.int8 and scales.shape == (500, 1)
    Xr = dequantize_int8(q, scales)
    Q = X[:8]
    exact = Q @ X.T
    approx = Q @ Xr.T
    assert np.abs(exact - approx).max() < 0.01                 # tight for normalized vectors
    # top-10 rankings essentially identical
    for r in range(8):
        a = np.argsort(-exact[r])[:10]
        b = np.argsort(-approx[r])[:10]
        assert len(set(a) & set(b)) >= 9
