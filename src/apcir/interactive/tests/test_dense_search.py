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
