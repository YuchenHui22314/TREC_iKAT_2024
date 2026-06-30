"""Capacity planner for the RALI Searcher: decide which indexes to load/evict for a
requested active set, using LIVE free RAM/VRAM and a LOAD-PEAK (not just resident) check.

Pure logic + injected host-state readers (default psutil/torch) so it is fully unit-testable.
Active-set semantics: the requested set becomes resident; anything not in it is evicted; the
request is refused only if the set itself cannot be loaded given current free memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Footprints + registry
# --------------------------------------------------------------------------- #
@dataclass
class IndexFootprint:
    name: str
    kind: str                       # "dense" | "sparse" | "splade" | "reranker"
    resident_ram_gb: float          # steady-state RAM once loaded
    load_peak_ram_gb: float         # transient peak DURING load (>= resident)
    vram_gb: float = 0.0           # steady-state VRAM (rerankers); dense fp16 search VRAM is transient -> 0
    index_dir: Optional[str] = None
    index_dir_alts: Optional[List[str]] = None  # fallback dirs tried (in order) if index_dir is missing
                                   # -> prefer fast local SSD (/part on octal31/40), fall back to NFS.
    dtype: Optional[str] = None
    embed_dim: Optional[int] = None      # dense units: query/doc embedding dim
    block_num: Optional[int] = None      # dense units: number of doc_emb blocks
    corpus: Optional[str] = None         # corpus tag; one active set must share a single corpus
    query_encoder: Optional[str] = None  # dense units: HF id / local path of the QUERY encoder this
                                         # index was built with (a leg routed here uses it unless the
                                         # leg sets an explicit encoder_path). None -> global default.

    def _candidate_dirs(self) -> List[str]:
        return [p for p in [self.index_dir, *(self.index_dir_alts or [])] if p]

    def resolved_index_dir(self) -> Optional[str]:
        """The dir to actually load from: the FIRST existing of index_dir then index_dir_alts (so a
        unit is served from fast local SSD when present, else the NFS copy). Falls back to index_dir
        (possibly missing) so a caller can still report a path."""
        import os
        for p in self._candidate_dirs():
            if os.path.isdir(p):
                return p
        return self.index_dir

    @property
    def is_available(self) -> bool:
        """True if no index dir is needed (e.g. reranker) or ANY candidate dir exists on disk."""
        import os
        return self.index_dir is None or any(os.path.isdir(p) for p in self._candidate_dirs())


class CapacityError(RuntimeError):
    """Raised when a requested active set cannot fit in available memory."""


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


# --------------------------------------------------------------------------- #
# Live host-state readers (defaults; injectable for tests)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def _dedup(seq):
    """Order-preserving de-duplication."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
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
        active, res = _dedup(active_set), _dedup(resident)
        to_unload = [r for r in res if r not in active]
        to_load = [a for a in active if a not in res]
        projected = sum(self.reg.get(a).resident_ram_gb for a in active)

        # RAM: evicting to_unload frees their resident; load to_load biggest-transient-first
        # (so the largest transient peak happens when the most RAM is free).
        free = self.free_ram_fn() + sum(self.reg.get(n).resident_ram_gb for n in to_unload)
        order = sorted(
            to_load,
            key=lambda n: self.reg.get(n).load_peak_ram_gb - self.reg.get(n).resident_ram_gb,
            reverse=True,
        )
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

        # VRAM: place each GPU-resident component (reranker) on the GPU with the most free VRAM,
        # largest-need first. Credit VRAM freed by evicting GPU-resident units ONLY when there is a
        # single GPU (unambiguous). On multi-GPU we don't track which GPU each unit sits on, so
        # crediting the most-free GPU could false-accept (free [8,1,1], evict two 9G on the full
        # GPUs, load 20G: real post-evict is [8,10,10], no GPU fits 20G). Per-GPU placement tracking
        # is a TODO for when reranker residency lands.
        if fits:
            gpu_free = sorted(self.free_vram_fn(), reverse=True)
            evicted_vram = sum(self.reg.get(n).vram_gb for n in to_unload)
            if len(gpu_free) == 1 and evicted_vram:
                gpu_free[0] += evicted_vram
            for n in sorted(to_load, key=lambda m: self.reg.get(m).vram_gb, reverse=True):
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

        return CapacityPlan(to_load=to_load, to_unload=to_unload, fits=fits,
                            reason=reason, projected_resident_ram_gb=projected)
