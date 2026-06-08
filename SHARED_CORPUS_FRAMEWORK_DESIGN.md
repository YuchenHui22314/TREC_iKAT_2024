# Shared-Corpus Dense-Retrieval Eval Framework — Design

This is the ground truth for the validation plan. The qwen_conv config is exactly the "14 qwen + 3 conv-ance" source: `qwen3` and `conv-qwen3` (2 models) × `{qwen_conversation, qwen_conversation_ptkb}` (2 query types) × `{ikat_23, ikat_24, ikat_25}` (3 topics) = 12 qwen-index experiments, all sharing the **same 6-block, 1024-dim qwen index** — the exact stacking case. ANCE uses a different 12-block 768-dim index. I now have every load-bearing fact verified. Here is the final design.

---

# Shared-Corpus Dense-Retrieval Eval Framework — Final Design

**Author:** lead architect • **Target repo:** `/data/rech/huiyuche/TREC_iKAT_2024`
**Verified against:** `dense_search.py`, `search.py`, `evaluation.py`, `run_experiments.py`, the `qwen_conv` config, and the live machine (251G RAM / 224G avail / 216G cache, 64 cores, merged qwen index = **6 blocks / 450G / dim 1024**, ANCE index = **12 blocks / dim 768**).

---

## 0. Thesis and the one fact that changes the math

The block-streaming loop `search_one_by_one_with_faiss` (`dense_search.py:71`) already calls `index.search(query_embeddings, topN)` on a **matrix** of queries — stacking more query rows is a near-free GEMM widening. The entire waste is structural: `run_experiments.py:107`'s `process.wait()` serializes N subprocesses, and each one re-streams **all blocks** of the same corpus from disk. For the canonical paper sweep, **12 experiments share one 450G qwen index** — that's 12 full 450G disk reads where **one** suffices.

**The win is "stream once, fan out":** group all jobs that share a corpus, encode each job's queries up front (cheap, GPU, MB-sized), vertically stack the query matrices, stream the corpus **once**, then slice merged results back per job. Disk cost collapses from `N × read` to `1 × read` per distinct corpus.

> **Critical machine-specific correction to all four candidates:** the merged qwen index is **6 blocks of ~75G each**, not 116 blocks of ~4G. This kills aggressive block prefetching (depth-2 = 150G resident, tight against 224G avail and competing with the 216G page cache) and means **fp16/SHM/daemon are unnecessary** for the common case. The single-process grouped runner is the right answer; the daemon is explicitly deferred (§8).

---

## 1. Architecture

Named components and responsibilities:

| Component | Responsibility | Reuses |
|---|---|---|
| **`run_experiments_grouped.py`** (new entry point) | Expand the SAME yaml into specs, partition by corpus key, drive groups. Sibling of `run_experiments.py`; old path untouched. | shared `expand_specs()` |
| **`ExperimentSpec`** | Frozen dataclass: the full arg set one experiment needs (every CLI key). | — |
| **`Scheduler`** | Partition specs into `JobGroup`s by `CorpusKey`; order/co-schedule groups (§4). | — |
| **`JobGroup`** | All specs sharing one `CorpusKey`. Owns exactly **one** corpus stream. | — |
| **`Encoder`** (pluggable) | `encode(texts, qids) -> (emb, ids)`. Adapters wrap the two existing branches of `get_test_query_embedding`. | `get_test_query_embedding` branches |
| **`BlockSource`** (pluggable) | Streamable block-partitioned doc index; the **only** disk-touching abstraction. Default = pickle blocks. | `dense_search.py:98-113` load logic |
| **`Index` / `IndexFactory`** | Fresh empty FAISS index per group stream. | `build_faiss_index` |
| **`GroupRunner`** | The core: encode → one stream (fan-out search per block) → slice → emit. | `search_one_by_one_with_faiss` body, `get_dense_ranking_list` |
| **`ResultSink`** (pluggable) | Build run-dict, write ranking list, evaluate, dump metrics. Default = iKAT TREC eval. | `get_run_object_and_save_ranking_list` tail + `evaluate` |

### Data flow

```
 YAML config (fixed + iterate + param_mapping)
        │
        ▼
 expand_specs()  ──►  List[ExperimentSpec]          # SHARED with run_experiments.py
        │
        ▼
 Scheduler.partition(specs)  ──► groups keyed by CorpusKey=(index_dir, embed_dim, block_num)
        │
        │   group A: {qwen3×qconv×23, qwen3×qconv×24, ... conv-qwen3×qconv_ptkb×25}  (12 jobs, ONE 450G qwen index)
        │   group B: {ance×oracle×23, ×24, ×25}                                      ( 3 jobs, ONE ance index)
        │
        ▼  for each group (Scheduler decides order/parallelism §4):
 ┌──────────────────────  GroupRunner.run(group)  ──────────────────────────┐
 │                                                                          │
 │  PHASE A  ENCODE (GPU, per-spec, cheap)                                   │
 │    for spec in group:                                                     │
 │        emb, qids = Encoder.for(spec).encode(spec.queries, spec.qids)      │
 │        record slice [cursor : cursor+len(qids)]                           │
 │    Q = np.concatenate(all emb)        # (ΣN, dim)  ~ few MB               │
 │                                                                          │
 │  PHASE B  ONE CORPUS STREAM (disk-bound, the win)                         │
 │    index = IndexFactory.new_empty(dim)                                    │
 │    topN  = max(spec.retrieval_top_k for spec in group)                    │
 │    for block in BlockSource.iter_blocks():     # 6 reads, ONCE            │
 │        index.add(block.emb)                                               │
 │        D, I = index.search(Q, topN)            # ALL stacked queries      │
 │        merge_topk(D, block.ids[I])             # in-place, no deepcopy    │
 │        index.reset()                                                      │
 │                                                                          │
 │  PHASE C  SLICE + EMIT (CPU, per-spec, parallel)                          │
 │    for spec, lo, hi in slices:                                            │
 │        hits = get_dense_ranking_list(qids[lo:hi], D[lo:hi], I[lo:hi], k)  │
 │        ResultSink.for(spec).emit(spec, hits)   # run-dict→eval→JSON       │
 └──────────────────────────────────────────────────────────────────────────┘
```

Three phases per group: **encode (GPU, ms)** → **stream+search (disk, the floor)** → **sink+eval (CPU)**. Only phase B touches disk, and only once per corpus.

---

## 2. Core abstractions (interfaces)

All in a new package `src/apcir/search/grouped/`. Method signatures:

```python
# spec.py
@dataclass(frozen=True)
class CorpusKey:
    index_dir: str          # args.dense_index_dir_path (abspath)
    embed_dim: int          # args.embed_dim
    block_num: int          # args.passage_block_num
    # NOTE: encoder is deliberately NOT in the key -> different encoders, same index, one group

@dataclass
class ExperimentSpec:
    args: SimpleNamespace   # the full arg namespace one evaluation.main() would build
    corpus_key: CorpusKey
    file_name_stem: str
    # convenience views populated during run:
    query_embeddings: np.ndarray = None
    query_embedding2id: list[str] = None

def expand_specs(config: dict) -> list[ExperimentSpec]: ...
    # itertools.product over iterate + param_mapping overlay (extracted from run_experiments.py:78-110)
```

```python
# encoder.py
class Encoder(Protocol):
    dim: int
    def encode(self, texts: list[str], qids: list[str]) -> tuple[np.ndarray, list[str]]: ...
    # returns (N, dim) float32 + parallel id list. Pure function of texts.

# Built-in adapters wrap the existing branches VERBATIM:
class AnceEncoder(Encoder):       dim = 768    # wraps get_test_query_embedding ance branch
class QwenEncoder(Encoder):       dim = 1024   # wraps qwen3 branch
# conv-ance / conv-qwen3 are the SAME encoder classes (same dim, same doc space) selected
# with a different checkpoint path + retrieval_query_type — see §5.
```

```python
# block_source.py
class Block(NamedTuple):
    emb: np.ndarray         # (n_i, dim) float32
    ids: np.ndarray         # (n_i,) docids
    block_id: int

class BlockSource(Protocol):
    num_blocks: int
    dim: int
    def iter_blocks(self) -> Iterator[Block]: ...    # caller drops each Block before next
    def validate(self) -> None: ...                  # assert all num_blocks files exist (pre-flight)

class PickleBlockSource(BlockSource):
    # reproduces dense_search.py:98-113: doc_emb_block.{i}.pb / doc_embid_block.{i}.pb
```

```python
# index.py
class IndexFactory(Protocol):
    def new_empty(self, dim: int) -> "faiss.Index": ...
class FaissFlatIPFactory(IndexFactory):   # wraps build_faiss_index(args) verbatim
```

```python
# sink.py
class ResultSink(Protocol):
    def emit(self, spec: ExperimentSpec, hits: dict[str, list]) -> dict: ...  # returns metrics

class TrecEvalSink(ResultSink):
    # build run-dict (get_run_object_and_save_ranking_list logic), write ranking list,
    # call evaluate(run, spec.args.qrel_file_path, ranking_list_path, metrics, key_form),
    # dump metrics JSON at the ikat-results-layout stem path.
```

```python
# runner.py
class GroupRunner:
    def run(self, group: JobGroup, index_factory, block_source, gpu_lock) -> None: ...

# scheduler.py
class Scheduler:
    def partition(self, specs: list[ExperimentSpec]) -> list[JobGroup]: ...
    def run_all(self, groups: list[JobGroup]) -> None: ...   # §4 policy
```

**Registries** (mirror the repo's `registry.py` / `model_factory.py` idiom):
```python
ENCODER_REGISTRY   = {"qwen3": QwenEncoder, "conv-qwen3": QwenEncoder,
                      "ance": AnceEncoder, "conv-ance": AnceEncoder}
BLOCK_SRC_REGISTRY = {"pickle_blocks": PickleBlockSource}
INDEX_REGISTRY     = {"faiss_flat_ip": FaissFlatIPFactory}
SINK_REGISTRY      = {"trec_eval": TrecEvalSink}
```
A new project = register one `Encoder` + (if its on-disk format differs) one `BlockSource` + a YAML group. Everything else is reused.

---

## 3. Exact integration with existing code

**Reuse AS-IS (zero edits — call from new code):**

| Function | Location | Used in |
|---|---|---|
| `build_faiss_index(args)` | `dense_search.py:28` | `FaissFlatIPFactory.new_empty` |
| `get_test_query_embedding(args)` | `dense_search.py:270` | `QwenEncoder`/`AnceEncoder.encode` (set `args.retrieval_query_list/qid_list_string/retrieval_model/dense_query_encoder_path/retrieval_query_type/query_gpu_id` per spec, call it, capture `(emb, ids)` — it already returns exactly this pair) |
| `get_dense_ranking_list(qids, D, I, k)` | `dense_search.py:363` | `GroupRunner` phase C, per slice |
| `get_query_list(args)` | `evaluation_util.py:29` | `expand_specs` / spec builder (produces `retrieval_query_list` + `qid_list_string`) |
| `evaluate(run, qrel, path, metrics, key_form)` | `evaluation_util.py:191` | `TrecEvalSink.emit` |
| `get_run_object_and_save_ranking_list(hits, args)` | `search.py:45` | `TrecEvalSink.emit` (build run-dict + write ranking list) |

**Thin adapters to write (new files, no logic change — just hoist branch bodies):**
- `QwenEncoder.encode` / `AnceEncoder.encode` → call `get_test_query_embedding` with a per-spec arg namespace. **No copy of encoder logic** — call the real function.
- `PickleBlockSource.iter_blocks` → the `dense_search.py:98-113` pickle-load body, yielding one `Block` at a time.
- `GroupRunner` phase B → a corpus-stream loop that is the **merge math of `search_one_by_one_with_faiss:120-212` lifted into a `Merger` with the per-block `copy.deepcopy` (line 171) removed** (replaced by pre-allocated `2*topN`-wide arrays). This is the one place we don't call the original verbatim, because the original couples one Q to one stream and pays the deepcopy. See §6 for the bit-identical guarantee.

**The ONE edit to an existing file** — a pure extract-function (behavior-identical, diff-reviewable):
- `run_experiments.py:78-110` (the `itertools.product` + `param_mapping` overlay) → extract into `grouped/spec.py::expand_specs(config)`. `run_experiments.py` then calls it. Both entry points share one expander; no drift. **The legacy subprocess path (`Popen` + `process.wait()`) is unchanged**, so every current run stays bit-for-bit identical.

**New entry point:** `src/apcir/evaluate/run_experiments_grouped.py` — opt-in, invoked as `python -m apcir.evaluate.run_experiments_grouped --config <same yaml>`. The existing `run_experiments.py`, `search.py`, `evaluation.py`, `dense_search(args)`, `evaluation_util.py` are **all untouched** except the one extract-function. The iKAT pipeline keeps working unchanged.

---

## 4. Concurrency & scheduling model

Three resource domains, three policies. The hard fact: **the disk (~150 MB/s, one device) is the wall-clock floor and is single-threaded by physics.** Two streams on one disk halve each other → no win. So:

**(1) Stacking = the primary concurrency (kills the N× blow-up).** Within a group, all jobs' queries ride **one** stream. Disk reads go from `N×` to `1×`. This is "concurrent experiment execution" done right: every job advances per block, sharing the resident block. For the qwen group, **12 jobs share 1 stream** → ~12× on the dominant term.

**(2) Encode/eval pipelining across groups (hide GPU+CPU under disk).** While group K is in disk-bound phase B, group K+1's phase A (encode, GPU) and group K-1's phase C (sink+eval, CPU) run concurrently — they don't touch the disk. A `gpu_lock` serializes encoder GPU use vs. FAISS GPU use. Net wall-clock ≈ `Σ_groups(one stream) + one encode + one eval`.

**(3) Block prefetch — DISABLED by default on this machine.** With 6 blocks of ~75G, a depth-2 prefetch needs 150G resident, fighting the 216G page cache and 224G avail. Default `prefetch_depth=1` (synchronous load→search). The `BlockSource` exposes `prefetch_depth` as a config knob: **only raise it when blocks are small** (e.g., a future 116-block re-dump), where `2 × 4G = 8G` is free. The framework auto-checks `prefetch_depth × block_bytes < ram_budget` and refuses otherwise (pre-flight, per CLAUDE.md).

**(4) Cross-corpus / multi-GPU parallelism — gated on `st_dev`.** Two groups on the **same disk** serialize (disk-bound; parallel = thrash). Two groups on **different physical disks** (detected via `os.stat(index_dir).st_dev`) run in parallel, splitting the 4 A5000s (`faiss_n_gpu=2` each, distinct `query_gpu_id`). The qwen index is on `/part/01` (local SSD); the ANCE index is on `/data/rech/...` — if these are distinct devices, the qwen group and ANCE group **can stream concurrently**. Scheduler policy:

```
groups = partition_by_corpus_key(specs)
for g in groups: g.disk = os.stat(g.index_dir).st_dev ; g.gpu_cost = blocks × Σjobs
lanes = bin_pack(groups, key=disk)        # one lane per physical disk
within a lane: run groups SEQUENTIALLY (single stream at a time)
across lanes (distinct st_dev): run in PARALLEL, split GPUs
within a group: one stream, gpu_lock serializes encode vs FAISS, phase C on ThreadPool
```

**Expected wall-clock** for the canonical sweep: old = `15 experiments × one full stream each`. New = `max(qwen_stream, ance_stream)` if on distinct disks, else `qwen_stream + ance_stream` — i.e. effectively **one or two streams total** instead of fifteen.

---

## 5. Stacking: different encoders, same doc index

This is the headline capability and the reason `CorpusKey` **excludes the encoder**. Worked against the real config:

The qwen group has 12 jobs spanning `qwen3` (base Qwen3-Embedding) **and** `conv-qwen3` (the instruct3 fine-tune checkpoint) — **two different query encoders / checkpoints** — across `qwen_conversation` and `qwen_conversation_ptkb` query types. All 12 target `clueweb22b_ikat23_qwen_merged` (same `embed_dim=1024`, same 6 blocks). They are **one group**:

- Phase A encodes each job **independently** with its own checkpoint + query type via `Encoder.encode` (which calls the real `get_test_query_embedding` branch). Job *j* → `Q_j (N_j, 1024)`.
- Stack: `Q = np.concatenate([Q_0, ..., Q_11])`, remember `slices[j] = (lo_j, hi_j)` and `qids_j`.
- Phase B: **one** stream over 6 blocks; per block, `index.search(Q, topN)` scores **all 12 jobs' queries** against the same resident block in one GEMM; `index.reset()` fires **once per block** (not per job).
- Phase C: `hits_j = get_dense_ranking_list(qids_j, D[lo_j:hi_j], I[lo_j:hi_j], k_j)`. Slicing is by contiguous row-range, so qid collisions across jobs are impossible (each job owns its rows and emits to its own output stem).

**Hard compatibility rule (asserted at partition time):** stacking is valid iff query vectors live in the **same space as the doc blocks** — `Encoder.dim == BlockSource.dim` is necessary. Crucially, `conv-qwen3` and `qwen3` **share the 1024-dim qwen doc space** (the fine-tune is query-side only; the frozen doc index is the base qwen index — confirmed in the config: both point at `clueweb22b_ikat23_qwen_merged`), so they legitimately co-stack. Likewise `conv-ance`/`ance` share the 768-dim ANCE space. An ANCE query (768) against the qwen index (1024) has a **different `CorpusKey`** (different dir AND dim) → different group, separate stream — the grouping key *is* the safety check. The `dim` assert turns the silent-garbage failure into a startup error; the framework **cannot** catch "right dim, wrong model family," so convention must bind each `BlockSource` to its producing encoder (store/check an encoder tag in the index dir — recommended).

---

## 6. Correctness / validation plan

**Goal:** reproduce the known **12 qwen + 3 conv-ance numbers** (the `qwen_conv` config: 2 qwen-family models × 2 query types × 3 years on the qwen index = 12; ANCE on its own index = 3) **EXACTLY** via the new path before trusting it for the paper.

**Why exact reproduction is achievable:** `IndexFlatIP` is **exact** inner-product search — stacking changes `search(Q_single)` to `search(Q_stacked)` but each row's scores are identical (no approximation, no cross-row interaction in a flat IP GEMM). The block order is identical. The only risk is the **merge**: float reduction order must match. Therefore:

1. **Bit-identical merge.** The `Merger` must reproduce `search_one_by_one_with_faiss`'s two-pointer (`:171-196`) exactly — same comparison (`>=`), same tie-breaking, same `2*topN` width — only removing the `copy.deepcopy` (which is a perf artifact, not a numerical one; deep-copying tuples doesn't change values). **Unit test:** feed the same `(D, I)` block sequence through the original loop and the `Merger`; assert `np.array_equal` on `merged_D, merged_I`.

2. **Single-job equivalence gate.** Run **one** spec (e.g. `qwen3 × qwen_conversation × ikat_23`) through BOTH paths:
   - legacy: `python -m apcir.evaluate.evaluation` with that spec's args (the current path), producing its ranking list + metrics JSON.
   - new: through `GroupRunner` with a group of size 1.
   `diff` the two ranking-list files. **They must be byte-identical** (same docids, same scores, same order). If not, stop — the merge or slicing is wrong.

3. **Stacked equivalence gate.** Run the same spec **inside a full 12-job stacked group**; diff its sliced ranking list against the size-1 run from step 2. **Byte-identical** — proves stacking + slicing doesn't perturb any single job.

4. **Full-sweep metric reproduction.** Run the whole `qwen_conv` config through `run_experiments_grouped.py`. For each of the 15 runs, compare the metrics JSON against the legacy run's JSON (located via the `ikat-results-layout` skill's filename stem). **All metrics equal to full float precision** (not just 4 decimals). Use `recip_rank, ndcg_cut_3, ndcg_cut_10, recall_100, recall_1000` (the config's `metrics_to_print`) as the gate, but assert on the full metric set.

5. **Pre-flight sweep (per CLAUDE.md), before any GPU work:** for every group assert — `index_dir` exists with `block_num` contiguous `doc_emb_block.{0..n-1}.pb` (a **merged** index, not raw per-rank blocks); each `qrel_file_path` exists per year; `Encoder.dim == BlockSource.dim == build_faiss_index dim` (probe block 0's `emb.shape[1]`); output dirs creatable; CWD is `src/`; env python is `trec_ikat` py3.12.

Only after gates 1-4 pass byte/precision-identical do we use the new path for the paper. Until then it runs **alongside** legacy, never replacing it.

---

## 7. Phased implementation plan

**Phase 0 — Extract + scaffolding (~0.5 day).**
Extract `expand_specs()` from `run_experiments.py:78-110`; have the legacy entry call it (verify a full legacy run is unchanged). Create `grouped/` package skeleton + registries. **Deliverable:** legacy path still bit-identical; specs expandable in-process.

**Phase 1 — MVP: single-process grouped eval, one corpus (~2 days).**
Implement `PickleBlockSource`, `FaissFlatIPFactory`, `QwenEncoder`/`AnceEncoder` adapters, the `Merger` (deepcopy-free, unit-tested for bit-identity vs original), `GroupRunner` (phases A/B/C, prefetch_depth=1), `TrecEvalSink`, and `run_experiments_grouped.py`. Scheduler partitions but runs groups **sequentially**. Run validation gates 1-4 on the `qwen_conv` config. **Deliverable:** 12 qwen + 3 ance numbers reproduced exactly, qwen group from **1 stream** instead of 12. This is the whole paper-grade win.

**Phase 2 — Full framework / extensibility (~1.5 days).**
Promote `Encoder`/`BlockSource`/`Index`/`ResultSink` to clean ABCs + registries; add the `framework:` YAML block (declarative jobs/encoders/sources, §2). Add the `dim` assert and encoder-tag check. Phase-C `ThreadPoolExecutor`. **Deliverable:** a new model/corpus plugs in via one registry line + YAML; reusable beyond iKAT.

**Phase 3 — Concurrency (optional, ~1.5 days).**
Cross-group encode/eval pipelining (`gpu_lock`); `st_dev`-gated cross-disk parallelism with GPU splitting; config-knob prefetch (only for small-block indexes). **Deliverable:** qwen + ANCE groups overlap when on distinct disks; further wall-clock cut.

**Phase 4 — Persistent daemon (DEFER; build only if justified, ~3+ days).** See §8 — not needed for the current machine/corpus.

Total to paper-grade win (Phases 0-1): **~2.5 days**. Full framework: **~4 days**.

---

## 8. Explicit limits — the capacity wall and when daemon/SHM/fp16 matter

**The wall (unavoidable):** 478G corpus > 251G RAM > 96G GPU. **There is no residency strategy.** The framework drives disk reads to **one stream per distinct corpus** but **cannot beat one full pass** — abstraction adds no bandwidth. Max speedup = average group size (12× for the qwen sweep). The stacked `Q` and one ~75G block + FAISS index must coexist; queries are MB-sized, so this is never the binding constraint here.

**When a persistent daemon (`CorpusHost` scatter-gather) is worthwhile — and when NOT:**
- **NOT worth it for this project's common case.** A daemon's win over the single-process grouped runner appears only when query-sets **arrive at different times across separate processes** and can't be batched in one launch. But the iKAT sweep is one config, one launch, all specs known up front → the in-process `GroupRunner` already achieves stream-once with zero IPC, zero socket protocol, zero crash-recovery surface. The daemon adds operational risk (stale sockets/locks, head-of-line blocking on a registration barrier) for **no additional disk savings**.
- **Worth it only if:** you need a long-lived service answering ad-hoc eval requests from independent processes/users over time against a hot corpus, where you cannot enumerate all query-sets in one launch. Then the inverted scatter-gather host (collect registrations in a window → one stacked stream → scatter slices) is the right pattern, with a **hard fallback invariant**: the host is never on the correctness path; any failure degrades to legacy `dense_search`. Until that requirement is real, **don't build it** (Phase 4, deferred).

**When `/dev/shm` / shared-memory matters — and when NOT:**
- **NOT for the corpus** (it's RAM; 478G doesn't fit; even fp16 239G leaves no headroom against the 216G page cache).
- **NOT needed in-process at all** — the single-process `GroupRunner` shares the resident block across all jobs **by reference**, for free, no IPC. SHM only becomes relevant as a *per-block transport* if you later go multi-process daemon (Phase 4) and profile the host→client result copy as hot (usually MB, so it won't be).

**When fp16 matters — and when NOT:**
- **NOT as a residency trick** (239G doesn't fit usefully).
- **Worth it as an on-disk dtype** to halve the disk read (~450G→~225G, the dominant term) and halve block footprint — but only if paired with a re-dump and only when read time dominates after stacking. For the current 6-block index, a re-dump is a large one-time cost; defer unless the single remaining stream's wall-clock is the bottleneck. It compounds with, but is orthogonal to, the stream-once win.

**Other limits:**
- **`topN = max(top_k)` per group** slightly inflates merge width (`2*max_topN`); bounded, cheap (config uses uniform `retrieval_top_k=1000`).
- **Lost per-experiment process isolation:** an in-process group shares a fate. Mitigation: wrap each phase-A encode and phase-C emit in try/except so one bad spec/qrel doesn't sink the group (the expensive phase B is shared and already done); optionally run each **group** in its own subprocess (keeps cross-group isolation, matching today's process model, while sharing the stream **within** a group).
- **`index.reset()` placement is load-bearing:** it must fire once per block *after* the stacked search, never per job — getting this wrong co-resides vectors from multiple blocks and silently corrupts results. Assert `index.ntotal == n_block` before search and `== 0` after reset.

---

### Relevant files
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/search/dense_search.py` — reuse `build_faiss_index:28`, `get_test_query_embedding:270`, `get_dense_ranking_list:363`; lift merge math from `search_one_by_one_with_faiss:71-212` (drop `copy.deepcopy:171`); legacy `dense_search:453` untouched.
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/search/search.py` — reuse `get_run_object_and_save_ranking_list:45`; dense dispatch `dense_search(args):614` untouched.
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/evaluate/evaluation.py` — eval seam `evaluate(run, args.qrel_file_path, ranking_list_path, ...)` ~`:549`; untouched.
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/evaluate/evaluation_util.py` — reuse `get_query_list:29`, `evaluate:191`; untouched.
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/evaluate/run_experiments.py` — **only edit:** extract `expand_specs` from `:78-110`; subprocess loop `:103-107` untouched.
- `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/evaluate/fuse_then_eval_config_qwen_conv.yaml` — the validation config (12 qwen + 3 ance).
- **New:** `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/search/grouped/{spec,encoder,block_source,index,runner,scheduler,sink}.py` + entry `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/evaluate/run_experiments_grouped.py`.