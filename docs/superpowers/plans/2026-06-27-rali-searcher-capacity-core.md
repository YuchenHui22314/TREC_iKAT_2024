# RALI Searcher — Capacity Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure, fully-unit-testable capacity planner (`capacity.py`) + its config (`capacity_config.yaml`) that decides, for a requested active set of indexes, which to load / which to evict / whether it fits — using **live free RAM/VRAM** and a **load-peak** (not just resident) check, with **active-set** semantics.

**Architecture:** A single new module `apcir/interactive/capacity.py` with three plain dataclasses + two classes (`IndexRegistry`, `CapacityManager`). All host-state reads (free RAM, free VRAM) are **injected functions** defaulting to psutil / torch, so unit tests pass fakes and never touch real hardware. This is sub-plan **2A.1** of the larger RALI Searcher Phase 2 (`/u/huiyuche/.claude/plans/ok-project-wild-candy.md`); it produces a working, tested planner with NO coupling to the pipeline. The next sub-plan (2A.2) wires `CapacityManager` into `pipeline.py` (`set_active`) + the FastAPI endpoints.

**Tech Stack:** Python 3.12 (`trec_ikat` env), stdlib `dataclasses` + `pyyaml` (already a dep), `pytest`. psutil/torch only at runtime via the default injected fns (not needed by tests).

## Global Constraints

- Python env: `/data/rech/huiyuche/envs/trec_ikat/bin/python` (3.12). Run tests from `src/`.
- New module path: `src/apcir/interactive/capacity.py`. Tests: `src/apcir/interactive/tests/test_capacity.py`.
- Config path: `src/apcir/interactive/capacity_config.yaml`.
- **CPU faiss is forbidden** — irrelevant here (no search), but the capacity model treats dense indexes as RAM-resident (fp16) units, consistent with the `fp16_torch` backend.
- Footprint values in the yaml are **measured estimates** refined by a documented `du` step; unit tests use **synthetic** footprints so they never depend on the real numbers.
- Memory units are **GB (gibibyte-ish, float)** throughout; one `ram_safety_gb` margin (default 8.0) is reserved on RAM, one `vram_safety_gb` (default 1.0) on each GPU.

---

### Task 1: `IndexFootprint` + `IndexRegistry`

**Files:**
- Create: `src/apcir/interactive/capacity.py`
- Create: `src/apcir/interactive/tests/__init__.py` (empty)
- Test: `src/apcir/interactive/tests/test_capacity.py`

**Interfaces:**
- Produces:
  - `@dataclass IndexFootprint(name: str, kind: str, resident_ram_gb: float, load_peak_ram_gb: float, vram_gb: float = 0.0, index_dir: Optional[str] = None, dtype: Optional[str] = None)`
  - `class IndexRegistry(footprints: Dict[str, IndexFootprint])` with `@classmethod from_yaml(path: str) -> IndexRegistry`, `get(name: str) -> IndexFootprint` (raises `KeyError` with the known names on miss), `names() -> List[str]`.

- [ ] **Step 1: Write the failing test**

Create `src/apcir/interactive/tests/__init__.py` (empty), then `src/apcir/interactive/tests/test_capacity.py`:

```python
from apcir.interactive.capacity import IndexFootprint, IndexRegistry


def _reg():
    return IndexRegistry({
        "dense_qwen": IndexFootprint("dense_qwen", "dense", 245.0, 330.0, index_dir="/x/qwen", dtype="float16"),
        "bm25": IndexFootprint("bm25", "sparse", 8.0, 8.0, index_dir="/x/bm25"),
        "reranker_qwen3": IndexFootprint("reranker_qwen3", "reranker", 1.0, 1.0, vram_gb=9.0),
    })


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError: cannot import name 'IndexFootprint'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/apcir/interactive/capacity.py`:

```python
"""Capacity planner for the RALI Searcher: decide which indexes to load/evict for a
requested active set, using LIVE free RAM/VRAM and a LOAD-PEAK (not just resident) check.

Pure logic + injected host-state readers (default psutil/torch) so it is fully unit-testable.
Active-set semantics: the requested set becomes resident; anything not in it is evicted; the
request is refused only if the set itself cannot be loaded given current free memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import yaml


@dataclass
class IndexFootprint:
    name: str
    kind: str                       # "dense" | "sparse" | "splade" | "reranker"
    resident_ram_gb: float          # steady-state RAM once loaded
    load_peak_ram_gb: float         # transient peak DURING load (>= resident)
    vram_gb: float = 0.0            # steady-state VRAM (rerankers); dense fp16 search VRAM is transient -> 0 here
    index_dir: Optional[str] = None
    dtype: Optional[str] = None


class IndexRegistry:
    def __init__(self, footprints: Dict[str, IndexFootprint]):
        self._fp = dict(footprints)

    @classmethod
    def from_yaml(cls, path: str) -> "IndexRegistry":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        fps = {}
        for name, d in (raw.get("indexes") or {}).items():
            fps[name] = IndexFootprint(name=name, **d)
        return cls(fps)

    def get(self, name: str) -> IndexFootprint:
        if name not in self._fp:
            raise KeyError(f"unknown index {name!r}; known: {sorted(self._fp)}")
        return self._fp[name]

    def names(self) -> List[str]:
        return list(self._fp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
cd /data/rech/huiyuche/TREC_iKAT_2024 && git add src/apcir/interactive/capacity.py src/apcir/interactive/tests/__init__.py src/apcir/interactive/tests/test_capacity.py && git commit -m "feat(capacity): IndexFootprint + IndexRegistry"
```

---

### Task 2: `CapacityManager.plan` — active-set + RAM load-peak

**Files:**
- Modify: `src/apcir/interactive/capacity.py` (append `CapacityPlan` + `CapacityManager`)
- Test: `src/apcir/interactive/tests/test_capacity.py` (append)

**Interfaces:**
- Consumes: `IndexRegistry`, `IndexFootprint` from Task 1.
- Produces:
  - `@dataclass CapacityPlan(to_load: List[str], to_unload: List[str], fits: bool, reason: str = "", projected_resident_ram_gb: float = 0.0)`
  - `class CapacityManager(registry, free_ram_fn: Callable[[], float] = <psutil default>, free_vram_fn: Callable[[], List[float]] = <torch default>, ram_safety_gb: float = 8.0, vram_safety_gb: float = 1.0)`
  - `CapacityManager.plan(active_set: List[str], resident: List[str]) -> CapacityPlan`
- RAM model (verbatim): `to_unload = [r for r in resident if r not in active_set]`; `to_load = [a for a in active_set if a not in resident]`. `free_after_unload = free_ram_fn() + sum(resident_ram_gb of to_unload)`. Load `to_load` ordered by **descending** `(load_peak_ram_gb - resident_ram_gb)` (biggest transient first, when most RAM is free); for each, require `load_peak_ram_gb + ram_safety_gb <= free`, then `free -= resident_ram_gb`. `fits` iff every load passes. `projected_resident_ram_gb = sum(resident_ram_gb for a in active_set)`.

- [ ] **Step 1: Write the failing test**

Append to `src/apcir/interactive/tests/test_capacity.py`:

```python
from apcir.interactive.capacity import CapacityManager, CapacityPlan


def _mgr(free_ram_gb, reg=None):
    return CapacityManager(reg or _reg(), free_ram_fn=lambda: free_ram_gb,
                           free_vram_fn=lambda: [24.0, 24.0, 24.0, 24.0])


def test_plan_fits_on_empty_octal40():
    # octal40: ~480G free, loading qwen (peak 330) from nothing -> fits
    plan = _mgr(480.0).plan(active_set=["dense_qwen", "bm25"], resident=[])
    assert plan.fits
    assert set(plan.to_load) == {"dense_qwen", "bm25"}
    assert plan.to_unload == []
    assert plan.projected_resident_ram_gb == 253.0


def test_plan_refuses_qwen_on_octal31_by_load_peak():
    # octal31: ~237G free; qwen RESIDENT 245 already > free, and PEAK 330 >> free -> refuse
    plan = _mgr(237.0).plan(active_set=["dense_qwen"], resident=[])
    assert not plan.fits
    assert "dense_qwen" in plan.reason and "330" in plan.reason


def test_plan_evicts_then_loads_active_set():
    # bm25 resident; switch to {dense_qwen}: must evict bm25, load qwen.
    plan = _mgr(200.0).plan(active_set=["dense_qwen"], resident=["bm25"])
    assert plan.to_unload == ["bm25"]
    assert plan.to_load == ["dense_qwen"]
    # free after evicting bm25 = 200 + 8 = 208 < peak 330 -> still refuses on this small host
    assert not plan.fits


def test_plan_evict_makes_it_fit():
    # big host: 100 free + evict a 245G qwen (resident) frees 345 -> ance (peak 220) fits
    reg = IndexRegistry({
        "dense_qwen": IndexFootprint("dense_qwen", "dense", 245.0, 330.0),
        "dense_ance": IndexFootprint("dense_ance", "dense", 184.0, 220.0),
    })
    plan = _mgr(100.0, reg).plan(active_set=["dense_ance"], resident=["dense_qwen"])
    assert plan.to_unload == ["dense_qwen"] and plan.to_load == ["dense_ance"]
    assert plan.fits  # 100 + 245 = 345 free >= 220 peak + 8 safety


def test_plan_noop_when_already_resident():
    plan = _mgr(50.0).plan(active_set=["bm25"], resident=["bm25"])
    assert plan.to_load == [] and plan.to_unload == [] and plan.fits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: FAIL — `ImportError: cannot import name 'CapacityManager'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/apcir/interactive/capacity.py`:

```python
def _default_free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / (1024 ** 3)


def _default_free_vram_gb() -> List[float]:
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        out.append(free / (1024 ** 3))
    return out


@dataclass
class CapacityPlan:
    to_load: List[str]
    to_unload: List[str]
    fits: bool
    reason: str = ""
    projected_resident_ram_gb: float = 0.0


class CapacityManager:
    def __init__(self, registry: IndexRegistry,
                 free_ram_fn: Callable[[], float] = _default_free_ram_gb,
                 free_vram_fn: Callable[[], List[float]] = _default_free_vram_gb,
                 ram_safety_gb: float = 8.0, vram_safety_gb: float = 1.0):
        self.reg = registry
        self.free_ram_fn = free_ram_fn
        self.free_vram_fn = free_vram_fn
        self.ram_safety_gb = ram_safety_gb
        self.vram_safety_gb = vram_safety_gb

    def plan(self, active_set: List[str], resident: List[str]) -> CapacityPlan:
        active, res = list(active_set), list(resident)
        to_unload = [r for r in res if r not in active]
        to_load = [a for a in active if a not in res]
        projected = sum(self.reg.get(a).resident_ram_gb for a in active)

        free = self.free_ram_fn() + sum(self.reg.get(n).resident_ram_gb for n in to_unload)
        order = sorted(to_load,
                       key=lambda n: self.reg.get(n).load_peak_ram_gb - self.reg.get(n).resident_ram_gb,
                       reverse=True)
        fits, reason = True, ""
        for n in order:
            fp = self.reg.get(n)
            if fp.load_peak_ram_gb + self.ram_safety_gb > free:
                fits = False
                reason = (f"{n} needs ~{fp.load_peak_ram_gb:.0f}G peak RAM but only "
                          f"~{free:.0f}G free"
                          + (f" after evicting {to_unload}" if to_unload else ""))
                break
            free -= fp.resident_ram_gb
        return CapacityPlan(to_load=to_load, to_unload=to_unload, fits=fits,
                            reason=reason, projected_resident_ram_gb=projected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: PASS (8 passed total).

- [ ] **Step 5: Commit**

```bash
cd /data/rech/huiyuche/TREC_iKAT_2024 && git add -A && git commit -m "feat(capacity): CapacityManager.plan with active-set + RAM load-peak"
```

---

### Task 3: VRAM check for GPU-resident components (rerankers)

**Files:**
- Modify: `src/apcir/interactive/capacity.py` (`plan` adds a VRAM pass)
- Test: `src/apcir/interactive/tests/test_capacity.py` (append)

**Interfaces:**
- Extends `plan`: after the RAM pass, for each `to_load` with `vram_gb > 0`, require some GPU's free VRAM `>= vram_gb + vram_safety_gb`; greedily place each on the GPU with the most free VRAM and decrement that GPU's running free. If none fits, `fits=False` with a VRAM reason.

- [ ] **Step 1: Write the failing test**

Append to `src/apcir/interactive/tests/test_capacity.py`:

```python
def test_plan_reranker_fits_on_a_gpu():
    mgr = CapacityManager(_reg(), free_ram_fn=lambda: 100.0,
                          free_vram_fn=lambda: [2.0, 12.0])  # gpu1 has 12G
    plan = mgr.plan(active_set=["reranker_qwen3"], resident=[])
    assert plan.fits  # 9G + 1 safety <= 12


def test_plan_reranker_refused_when_no_gpu_room():
    mgr = CapacityManager(_reg(), free_ram_fn=lambda: 100.0,
                          free_vram_fn=lambda: [2.0, 6.0])   # neither has 9+1
    plan = mgr.plan(active_set=["reranker_qwen3"], resident=[])
    assert not plan.fits
    assert "VRAM" in plan.reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: FAIL — `test_plan_reranker_refused_when_no_gpu_room` fails (no VRAM check yet, returns fits=True).

- [ ] **Step 3: Write minimal implementation**

In `capacity.py`, inside `plan`, after the RAM loop and before `return`, insert:

```python
        if fits:
            gpu_free = sorted(self.free_vram_fn(), reverse=True)  # most-free first
            for n in to_load:
                need = self.reg.get(n).vram_gb
                if need <= 0:
                    continue
                if not gpu_free or gpu_free[0] < need + self.vram_safety_gb:
                    fits = False
                    reason = (f"{n} needs ~{need:.0f}G VRAM but no GPU has that free "
                              f"(free per GPU: {[round(g, 1) for g in gpu_free]})")
                    break
                gpu_free[0] -= need
                gpu_free.sort(reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: PASS (10 passed total).

- [ ] **Step 5: Commit**

```bash
cd /data/rech/huiyuche/TREC_iKAT_2024 && git add -A && git commit -m "feat(capacity): VRAM placement check for GPU-resident components"
```

---

### Task 4: `capacity_config.yaml` + from_yaml smoke test

**Files:**
- Create: `src/apcir/interactive/capacity_config.yaml`
- Test: `src/apcir/interactive/tests/test_capacity.py` (append)

**Interfaces:**
- Consumes: `IndexRegistry.from_yaml` (Task 1).
- The yaml top-level key is `indexes:`; each child maps a component name → footprint fields (minus `name`). Values are **measured estimates** (refine with the `du` step below); the registry keys are the loadable units the pipeline resolves retriever names to (next sub-plan): `dense_qwen` (qwen3 / conv-qwen3), `dense_ance` (ance / conv-ance), `splade` (splade_v3), `bm25` (BM25 + doc-fetch), `reranker_qwen3` (qwen3_reranker).

- [ ] **Step 1: Write the failing test**

Append to `src/apcir/interactive/tests/test_capacity.py`:

```python
import os


def test_capacity_config_yaml_loads_and_has_expected_units():
    path = os.path.join(os.path.dirname(__file__), "..", "capacity_config.yaml")
    reg = IndexRegistry.from_yaml(path)
    for unit in ("dense_qwen", "dense_ance", "splade", "bm25", "reranker_qwen3"):
        fp = reg.get(unit)
        assert fp.load_peak_ram_gb >= fp.resident_ram_gb >= 0
    # qwen's load peak must exceed octal31's ~237G free so the guard refuses it there
    assert reg.get("dense_qwen").load_peak_ram_gb > 237
    assert reg.get("reranker_qwen3").vram_gb > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py::test_capacity_config_yaml_loads_and_has_expected_units -v`
Expected: FAIL — `FileNotFoundError` (yaml not created yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/apcir/interactive/capacity_config.yaml` (values = starting estimates from the index dtypes/sizes; refine via Step 4b):

```yaml
# Loadable capacity units for the RALI Searcher. resident_ram_gb = steady-state RAM held once
# loaded; load_peak_ram_gb = transient peak DURING load (dense = read fp32 block + cast to fp16,
# block by block, so peak = resident + largest fp32 block + its fp16 cast). vram_gb = steady-state
# VRAM (rerankers only; dense fp16 search VRAM is transient). Refine numbers with the du step.
indexes:
  dense_qwen:
    kind: dense
    resident_ram_gb: 245.0
    load_peak_ram_gb: 330.0
    vram_gb: 0.0
    index_dir: /part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged
    dtype: float16
  dense_ance:
    kind: dense
    resident_ram_gb: 184.0
    load_peak_ram_gb: 220.0
    vram_gb: 0.0
    index_dir: /part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_ance_merged_2
    dtype: float16
  splade:
    kind: splade
    resident_ram_gb: 176.0
    load_peak_ram_gb: 200.0
    vram_gb: 0.0
    index_dir: /part/01/Tmp/yuchen/indexes/splade_v3_clueweb22B
    dtype: int16
  bm25:
    kind: sparse
    resident_ram_gb: 8.0
    load_peak_ram_gb: 8.0
    vram_gb: 0.0
    index_dir: /part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2
    dtype: null
  reranker_qwen3:
    kind: reranker
    resident_ram_gb: 1.0
    load_peak_ram_gb: 1.0
    vram_gb: 9.0
    index_dir: null
    dtype: bf16
```

- [ ] **Step 3b (optional refinement, not a test): measure on-disk sizes**

Run (records real sizes to refine the estimates above; commit the yaml after editing):
`du -sh /part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged /part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_ance_merged_2 /part/01/Tmp/yuchen/indexes/splade_v3_clueweb22B 2>/dev/null`
Note: on-disk is fp32 for dense; resident fp16 ≈ on-disk/2; load_peak ≈ resident + (on-disk/blocks) + (cast = that/2). Leave the yaml as-is if measurement is unavailable.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/rech/huiyuche/TREC_iKAT_2024/src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v`
Expected: PASS (11 passed total).

- [ ] **Step 5: Commit**

```bash
cd /data/rech/huiyuche/TREC_iKAT_2024 && git add -A && git commit -m "feat(capacity): capacity_config.yaml with measured-estimate footprints"
```

---

## Self-Review

- **Spec coverage:** Plan §2A "NEW `capacity_config.yaml` + NEW `capacity.py` (IndexRegistry + CapacityManager: live free RAM via psutil, VRAM via mem_get_info; plan(active_set) → {to_load, to_unload, fits, reason}, active-set semantics, load-peak check)" → Task 1 (registry), Task 2 (plan + RAM load-peak + active-set), Task 3 (VRAM), Task 4 (config). The octal31 gotcha (qwen peak > free → refuse) → `test_plan_refuses_qwen_on_octal31_by_load_peak` + the config assertion. ✅ Covered. `set_active` + endpoints are explicitly the NEXT sub-plan (2A.2), not here.
- **Placeholder scan:** No TBD/"handle errors"/"similar to" — every step has complete code. ✅
- **Type consistency:** `IndexFootprint(name, kind, resident_ram_gb, load_peak_ram_gb, vram_gb, index_dir, dtype)` used identically in tests, `from_yaml`, config, and `CapacityManager`. `CapacityPlan(to_load, to_unload, fits, reason, projected_resident_ram_gb)` consistent across Tasks 2–3. `plan(active_set, resident)` signature stable. ✅

## Verification (end of sub-plan)
`cd src && /data/rech/huiyuche/envs/trec_ikat/bin/python -m pytest apcir/interactive/tests/test_capacity.py -v` → 11 passed. The planner now: maps a requested active set to to_load/to_unload (active-set), refuses sets that won't fit by **load-peak** against **live** free RAM (qwen refused on octal31, accepted on octal40), and places GPU rerankers by free VRAM. Ready for 2A.2 to call `CapacityManager.plan` inside `pipeline.set_active`.
