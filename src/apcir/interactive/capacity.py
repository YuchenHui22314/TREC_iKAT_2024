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
    query_encoders: Optional[List[dict]] = None  # dense units: OPTIONAL list of selectable query-
                                         # encoder checkpoints that all search this same doc index
                                         # (query-encoder-only fine-tunes). Each entry:
                                         # {label, path, leg_name, default_query_type}.
    num_docs: Optional[int] = None       # dense units: total vectors (sizes the PQ index)
    pq_index_path: Optional[str] = None  # dense units: prebuilt IVF-PQ64 faiss file (pq_refine
                                         # mode); None -> {resolved_index_dir}/ivfpq64.faiss

    # -- load modes (dense units) ------------------------------------------- #
    # ram_fp16     : fp16 blocks in CPU RAM, streamed to GPU per request (exact; default)
    # gpu_resident : fp16 shards uploaded ONCE to GPUs, GEMM per request (exact, ~100x faster;
    #                needs VRAM = the fp16 store, sharded)
    # pq_refine    : prebuilt IVF-PQ64 on ONE GPU for candidates + fp16 RAM store for exact
    #                rescore (~2% NDCG@3 tax; tiny VRAM; RAM = same as ram_fp16)
    MODES = ("ram_fp16", "gpu_resident", "pq_refine")

    @property
    def pq_vram_gb(self) -> float:
        """IVF-PQ64 GPU footprint: 64 B/vec codes + ids/lists overhead (~15%) + coarse tables."""
        n = self.num_docs or 0
        return round(n * 64 * 1.15 / 1e9 + 0.3, 2)

    def resolved_pq_path(self) -> Optional[str]:
        import os
        if self.pq_index_path:
            return self.pq_index_path
        base = self.resolved_index_dir()
        return os.path.join(base, "ivfpq64.faiss") if base else None

    @property
    def has_pq_index(self) -> bool:
        import os
        p = self.resolved_pq_path()
        return bool(p) and os.path.exists(p)

    def mode_requirements(self, mode: str) -> Dict[str, float]:
        """Memory needs of loading this unit under `mode`:
        ram_gb/load_peak_ram_gb (CPU) + vram_total_gb (sharded across GPUs) / vram_single_gb."""
        if mode == "ram_fp16" or self.kind != "dense":
            return dict(ram_gb=self.resident_ram_gb, load_peak_ram_gb=self.load_peak_ram_gb,
                        vram_total_gb=0.0, vram_single_gb=self.vram_gb)
        if mode == "gpu_resident":
            # blocks stream disk->fp16->GPU one at a time; CPU holds ~one fp32 block transiently
            blk = self.resident_ram_gb / max(1, self.block_num or 1) * 2
            ram = max(2.0, round(blk, 1))
            return dict(ram_gb=ram, load_peak_ram_gb=ram,
                        vram_total_gb=self.resident_ram_gb, vram_single_gb=0.0)
        if mode == "pq_refine":
            # rescore store is INT8 (half of the fp16 resident figure); peak = int8 store + the
            # same fp32-block transient the fp16 load has (load_peak - resident).
            ram = self.resident_ram_gb / 2
            peak = ram + max(0.0, self.load_peak_ram_gb - self.resident_ram_gb)
            return dict(ram_gb=round(ram, 1), load_peak_ram_gb=round(peak, 1),
                        vram_total_gb=0.0, vram_single_gb=self.pq_vram_gb)
        raise ValueError(f"unknown load mode {mode!r} (known: {self.MODES})")

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
        # unit -> {gpu_id: gb} — where each resident unit's VRAM actually sits. Lets plan() credit
        # evictions per-GPU on multi-GPU hosts (the old single-GPU-only limitation is gone).
        self._placements: Dict[str, Dict[int, float]] = {}
        # unit -> load mode it was ACTUALLY loaded under (evicting a gpu_resident unit frees its
        # small streaming RAM, not the full fp16 store — codex review #4)
        self._loaded_modes: Dict[str, str] = {}

    # ------------------------------ VRAM fitting --------------------------- #
    def _vram_fit(self, gpu_free: List[float], singles: List[tuple],
                  shards: List[tuple]) -> tuple:
        """Dry-run placement. singles=[(name, gb)] need ONE gpu each; shards=[(name, total_gb)]
        spread over as many GPUs as needed. Mutates gpu_free. Returns (ok, reason, placements)
        where placements[name] = {gpu_id: gb}."""
        placed: Dict[str, Dict[int, float]] = {}
        for name, need in sorted(singles, key=lambda x: -x[1]):
            if need <= 0:
                continue
            best = max(range(len(gpu_free)), key=lambda g: gpu_free[g], default=None)
            if best is None or gpu_free[best] < need + self.vram_safety_gb:
                return False, (f"{name} needs ~{need:.1f}G VRAM on one GPU but free per GPU is "
                               f"{[round(g, 1) for g in gpu_free]}"), placed
            gpu_free[best] -= need
            placed[name] = {best: need}
        for name, total in sorted(shards, key=lambda x: -x[1]):
            if total <= 0:
                continue
            usable = sorted(range(len(gpu_free)), key=lambda g: -gpu_free[g])
            got, alloc = 0.0, {}
            for g in usable:
                room = gpu_free[g] - self.vram_safety_gb
                if room <= 0:
                    continue
                take = min(room, total - got)
                alloc[g] = take
                got += take
                if got >= total - 1e-6:
                    break
            if got < total - 1e-6:
                return False, (f"{name} needs ~{total:.0f}G VRAM sharded but only "
                               f"~{got:.0f}G is free across GPUs "
                               f"({[round(g, 1) for g in gpu_free]})"), placed
            for g, take in alloc.items():
                gpu_free[g] -= take
            placed[name] = alloc
        return True, "", placed

    def _needs(self, name: str, mode: Optional[str]) -> Dict[str, float]:
        return self.reg.get(name).mode_requirements(mode or "ram_fp16")

    def plan(self, active_set: List[str], resident: List[str],
             modes: Optional[Dict[str, str]] = None) -> CapacityPlan:
        active, res = _dedup(active_set), _dedup(resident)
        modes = modes or {}
        to_unload = [r for r in res if r not in active]
        to_load = [a for a in active if a not in res]
        req = {n: self._needs(n, modes.get(n)) for n in to_load}
        projected = sum(self._needs(a, modes.get(a))["ram_gb"] for a in active)

        # RAM: evicting to_unload frees their resident; load to_load biggest-transient-first
        # (so the largest transient peak happens when the most RAM is free).
        free = self.free_ram_fn() + sum(
            self._needs(n, self._loaded_modes.get(n))["ram_gb"] for n in to_unload)
        order = sorted(to_load, key=lambda n: req[n]["load_peak_ram_gb"] - req[n]["ram_gb"],
                       reverse=True)
        fits, reason = True, ""
        for n in order:
            if req[n]["load_peak_ram_gb"] + self.ram_safety_gb > free:
                fits = False
                reason = (f"{n} needs ~{req[n]['load_peak_ram_gb']:.0f}G peak RAM but only "
                          f"~{free:.0f}G free"
                          + (f" after evicting {to_unload}" if to_unload else ""))
                break
            free -= req[n]["ram_gb"]

        # VRAM: per-GPU simulation. Start from LIVE free, credit the TRACKED placements of
        # evicted units on their actual GPUs, then place singles (pq/reranker) and shards
        # (gpu_resident) with the same fitter used at load time.
        if fits:
            gpu_free = list(self.free_vram_fn())
            for n in to_unload:
                placement = self._placements.get(n)
                if placement:
                    for g, gb in placement.items():
                        if g < len(gpu_free):
                            gpu_free[g] += gb
                elif len(gpu_free) == 1:
                    # untracked legacy unit (loaded before placement tracking): crediting is
                    # unambiguous only on a single-GPU host — same conservatism as before.
                    gpu_free[0] += self.reg.get(n).vram_gb
            singles = [(n, req[n]["vram_single_gb"]) for n in to_load]
            shards = [(n, req[n]["vram_total_gb"]) for n in to_load]
            ok, why, _ = self._vram_fit(gpu_free, singles, shards)
            if not ok:
                fits, reason = False, why

        return CapacityPlan(to_load=to_load, to_unload=to_unload, fits=fits,
                            reason=reason, projected_resident_ram_gb=projected)

    # --------------------------- live placement ---------------------------- #
    def allocate_gpus(self, name: str, mode: str) -> List[tuple]:
        """Assign concrete GPUs for loading `name` under `mode` NOW (live free VRAM). Records the
        placement so later plans credit it on eviction. Returns [(gpu_id, gb), ...]."""
        req = self._needs(name, mode)
        gpu_free = list(self.free_vram_fn())
        ok, why, placed = self._vram_fit(
            gpu_free,
            [(name, req["vram_single_gb"])],
            [(name, req["vram_total_gb"])],
        )
        if not ok:
            raise CapacityError(why)
        alloc = placed.get(name, {})
        self._placements[name] = dict(alloc)
        self._loaded_modes[name] = mode
        return sorted(alloc.items())

    def note_loaded(self, name: str, mode: str) -> None:
        """Record the mode a unit was loaded under (RAM-only modes never call allocate_gpus)."""
        self._loaded_modes[name] = mode

    def release_gpus(self, name: str) -> None:
        self._placements.pop(name, None)
        self._loaded_modes.pop(name, None)
