# Dense vector search in apcir/Magpie: memory & speed vs professional vector-search approaches

*Analysis + micro-benchmark, 2026-07-03, octal31 (4× RTX A5000 24 GB, 251 GB RAM, faiss 1.8.0, torch 2.5.1+cu124).*
Harness: `src/apcir/benchmarks/bench_dense.py`; raw results: `/part/01/Tmp/yuchenhui/bench_dense/results.jsonl`.

## Executive summary

- The apcir dense retriever is **exact brute-force inner product** (no ANN), over L2-normalized
  embeddings, in two production paths: an offline faiss path and an interactive `fp16_torch` path.
  Both **re-stream the whole corpus per query batch** (offline: from disk; interactive: CPU→GPU
  over PCIe).
- The single biggest *interactive* speedup that keeps results **the same** is to stop
  re-streaming: keep the fp16 index **GPU-resident** and only run the query per turn (configs
  **c/d**). Measured p50 on the 26 M-doc unit dropped **2.73 s → 0.033 s (config c, ≈ 82×)** at
  recall@10 0.98, for corpora that fit VRAM. The current production streaming path (config a) is
  **2.73 s/query** at 26 M — and ~12 s extrapolated at ClueWeb-116 M — which is the "search is slow"
  the demo showed.
- **Two measured findings that change the recommendation:**
  1. **Keep the query in fp32.** The production `_search_ram_fp16_gpu` casts the *query* to fp16
     too (ram_index.py:188). That alone costs recall: config **d** (query fp16) = recall@10 **0.83**
     vs config **c** (faiss `useFloat16`, query fp32) = **0.98** — same speed class, +0.15 recall,
     free. The interactive fp16 path therefore already **diverges ~17% in top-10 docids from the
     exact offline eval** (largely tie-shuffling among QReCC's near-duplicate URLs, but real).
  2. **Product Quantization (PQ) fails on these ANCE embeddings.** IVF-**PQ96** recall@10 = **0.05–0.07**
     (and *inverts* with nprobe). Normalizing helps only ~2× (0.10→0.24 on 1 M) — still unusable.
     The viable GPU compression is **Scalar Quantization**: IVF-**SQ8** = 0.93 (4×), IVF-**SQfp16** =
     0.95 (2×). So the naive "IVF-PQ96 ≈ 11 GB fits one GPU" plan is **wrong for ANCE** — use SQ8.
- For **ClueWeb-116 M** the exact fp16 index (ANCE 178 GB / Qwen 237 GB) does **not** fit 96 GB total
  VRAM, so GPU-resident-exact is impossible there; options are (i) keep streaming (exact, slow), or
  (ii) an **ANN** index — **IVF-SQ8** ≈ 89 GB (ANCE, sharded across the 4 GPUs, recall ~0.93); PQ is
  out (recall collapse). Whether 0.93 recall is acceptable is a research-quality decision.
- **Reproducibility constraint**: the offline eval feeds papers, so it must stay **fp32 exact**
  (bit-exact `merge_compat`). The fp16 finding above shows even fp16 already reshuffles top-10 — ANN
  and fp16 are acceptable only for the interactive **demo**, never for reported numbers.
- Professional vector DBs (Elasticsearch, Milvus, Qdrant, …) are **not** a fit here and mostly do
  **not** use faiss — see §6. Their query path is CPU; adopting one throws away the GPUs that make
  exact search feasible.

## 1. What the code does today

### 1.1 On-disk format
Merged blocks `doc_emb_block.{i}.pb` (pickled `np.ndarray (n_i, dim)` **float32**) +
`doc_embid_block.{i}.pb` (docid list). Typical merged block ≈ 1 M docs; ClueWeb-Qwen is merged
coarse (6 blocks × ~80 GB). Search is inner product (`IndexFlatIP`). Note: **Qwen embeddings are
L2-normalized** (norm 1.0 → IP = cosine), but **ANCE embeddings are NOT** (measured doc-vector norm
≈ **26.8**, IP scores ≈ 700). That un-normalized, large-magnitude regime is exactly what makes fp16
storage and PQ lossy for ANCE (§5) — a normalized encoder (Qwen) would fare better on both.

### 1.2 Interactive path (`apcir/interactive/ram_index.py`)
`RamBlockSource` loads **all** blocks into CPU RAM once at startup as **fp16**
(`store_dtype="float16"`; `malloc_trim` returns freed fp32 buffers to the OS; docids kept as
`np.array(dtype=object)` — a fixed-width unicode array of QReCC's ≤6335-char URL docids would be
~25 GB/block). Per turn, `search_query_against_ram`:
- **fp16 units → `_search_ram_fp16_gpu`** (ram_index.py:173): each block's rows are split across
  GPUs, streamed H2D in 4 M-row chunks, scored `Q @ shard.T` (fp16 in, fp32 accumulate), per-chunk
  `torch.topk`, running merge — **no faiss, no fp32 cast**. But the whole corpus is re-streamed
  CPU→GPU **every request**.
- fp32 units → faiss `IndexFlatIP` add→search→reset per block (also re-uploaded per request).

### 1.3 Offline eval path (`apcir/search/dense_search.py`, `grouped/runner.py`)
`build_faiss_index` → `IndexFlatIP` cloned to GPUs with `GpuMultipleClonerOptions(shard=True)`;
per block `add → search(Q_all, topN) → reset`; cross-block merge via `merge_compat` (bit-identical
legacy two-pointer) or `merge_topk` (numpy). All topics are stacked into one `Q`, so per-query cost
is amortized — but every experiment re-streams every block from disk. CPU faiss is **hard-forbidden**
(dense_search.py:66) — GPU only.

### 1.4 No ANN anywhere
Only `IndexFlatIP` (exact). No IVF/PQ/HNSW in the main path (the lone `IndexHNSWFlat` is in unused
`convdr` DPR code). This is a deliberate exact-search system optimized for recall/reproducibility.

## 2. Where the memory goes (measured)

Dense embeddings dominate; docids and glibc arenas are the footnotes the code already handles.
Per corpus (fp32 on disk → **fp16 resident** → load peak):

| Corpus | encoder | #docs | dim | blocks | fp32 disk | fp16 resident | load peak |
|---|---|---:|---:|---:|---:|---:|---:|
| ClueWeb22-B | ANCE | ~117 M | 768 | 12 | 336 GB | 168 GB | 210 GB |
| ClueWeb22-B | Qwen3-0.6B | ~117 M | 1024 | 6 | 450 GB | 235 GB | 320 GB |
| QReCC | ANCE | ~56 M | 768 | 55 | 161 GB | 80 GB | 85 GB |
| QReCC | Qwen3-0.6B | ~56 M | 1024 | 55 | 214 GB | 107 GB | 112 GB |
| TopiOCQA | ANCE | ~26 M | 768 | 26 | 74 GB | 37 GB | 42 GB |

fp16 already halves memory (the physical floor: `N × dim × 2 B`). The **320 GB ClueWeb-Qwen load
peak** is an artifact of the 6 coarse ~80 GB blocks (resident 235 GB + one 80 GB fp32 block being
cast); **re-merging to ~1 M-doc blocks drops the peak to ≈ 239 GB** — a cheap, exact-preserving win
that lets it load on a 251 GB node. Measured docid overhead (object array): **3.9 GB for 26 M** long
QReCC URL docids — real but ~10× smaller than the 40 GB embeddings, and ~200× smaller than the
fixed-width-unicode trap would have been. fp16 load of the 26 blocks (~40 GB) from /part took
**125 s** (≈ 320 MB/s).

## 3. Where the time goes (measured)

The interactive headline is **per-request PCIe re-streaming**: the 26 M×768 fp16 index is ~40 GB;
streaming it CPU→GPU every turn is the floor on the streaming configs — measured 2.73 s/query (config a) vs 0.023–0.033 s once resident (configs c/d), a ~82–119× gap that is entirely the re-streaming. Keeping the
index GPU-resident removes that entirely — the query becomes one batched GEMM. Batched queries are
nearly free on GPU (config latency at batch 1 vs 16 below): scoring 16 queries costs ≈ scoring 1,
which is exactly why the offline eval stacks all topics.

## 4. Benchmark setup

Unit: first **26 blocks of `qrecc_ance_merged`** (26 M × 768-d — same scale+dim as TopiOCQA-ANCE,
local SSD; QReCC's long URL docids also exercise the object-array cost). 128 real QReCC
`Truth_rewrite` queries encoded once with the production ANCE encoder. Metrics per (config, param,
batch∈{1,16}): build/load time, steady RSS Δ, per-GPU VRAM, latency p50/p95 (warmup 5, ≥40 reps for
resident configs / 8 for streaming), QPS, and **recall@10/@100 vs the exact fp32 ground truth**
(config b, rep 1). Baselines a/b call the **real production functions** unmodified. All GPU; ANN CPU
fallbacks are labelled.

## 5. Trade-off table

### 5.1 Measured (26 M × 768-d, QReCC-ANCE, batch 1; 128 real queries)

| config | approach | build/load | RAM Δ | VRAM | p50 | R@10 | R@100 | vs paper |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **a** stream_fp16_torch | **production interactive** (re-stream fp16, query fp16) | 125 s load | 44 G | 0.6 G/req | **2.73 s** | 0.83 | 0.81 | near-fp16 |
| **b** stream_faiss_fp32 | **production offline** (= ground truth) | — | 84 G | 2.2 G | 3.67 s | **1.00** | 1.00 | bit-exact |
| **c** resident_faiss_fp16 | GPU-resident, **query fp32** | 5 s | 87 G | 11.7 G | **0.033 s** | **0.98** | 0.99 | near-fp16 |
| **d** resident_torch_fp16 | GPU-resident, query fp16 | 3 s | 4 G | 10.8 G | **0.023 s** | 0.83 | 0.81 | near-fp16 |
| **e** IVF-SQfp16 nprobe=128 | ANN, 2 B/vec | 38 s | 88 G | 18.6 G | 0.001 s | 0.95 | 0.92 | approx |
| **e** IVF-SQfp16 nprobe=32 | | 38 s | 88 G | 18.6 G | 0.001 s | 0.89 | 0.82 | approx |
| **f** IVF-SQ8 nprobe=128 | ANN, 1 B/vec | 38 s | 89 G | 10.2 G | 0.001 s | 0.93 | 0.91 | approx |
| **f** IVF-SQ8 nprobe=32 | | 38 s | 89 G | 10.2 G | 0.001 s | 0.87 | 0.82 | approx |
| **f** IVF-PQ96 nprobe=128 | ANN, 96 B/vec | 48 s | 90 G | **2.8 G** | 0.001 s | **0.05** | 0.07 | **broken (ANCE)** |
| **g** CPU-HNSW32 ef=256 | CPU-ANN (ES/Qdrant proxy), 5 M subset | **486 s** build | 108 G | 0 | 0.013 s | 0.84\* | 0.75\* | approx |
| **g** CPU-HNSW32 ef=64 | 5 M subset | 486 s build | 108 G | 0 | 0.013 s | 0.76\* | 0.57\* | approx |

\* HNSW recall is vs a 5 M-subset exact GT (not head-to-head with the 26 M rows). Its 486 s build for
5 M extrapolates to ~40 min for 26 M / ~3 h for ClueWeb-scale — the CPU-ANN build tax. Even so it is
slower per query (13 ms) AND lower recall than GPU IVF-SQ (<1 ms, 0.93) — **GPU IVF-SQ dominates CPU
HNSW here**, which is exactly the point about CPU-query vector DBs (§6).

**Reading the table:** at 26 M, GPU-resident **exact** (c: 33 ms, recall 0.98) is *already*
interactive-fast — ANN buys sub-ms latency but pays recall, so it is **not worth it at this scale**.
ANN only becomes *necessary* at ClueWeb-116 M where exact fp16 no longer fits VRAM.

### 5.2 ClueWeb-116 M extrapolation

Costs scale ~linearly in N (GEMM + PCIe) and index storage in N×bytes:
- **Streaming exact (current)**: ~12 s/query (2.73 s × 116/26) — the status quo, correct but slow.
- **GPU-resident exact fp16**: ANCE 116 M×768×2 = **178 GB**, Qwen 116 M×1024×2 = **237 GB** — both
  exceed 96 GB total VRAM → **impossible on octal31**; stays a streaming (or CPU) job.
- **IVF-SQ8** (the viable ANN): ANCE ≈ **89 GB** → fits *sharded* across the 4 GPUs (89/96, tight),
  recall ≈ **0.93**, sub-ms. Qwen 116 M×1024 = 116 GB SQ8 → doesn't fit even sharded; stays streaming
  or needs a smaller-code scheme.
- **IVF-PQ96** (11 GB, one GPU): **ruled out** — recall 0.05 on ANCE (0.24 even normalized). Would
  need OPQ + normalized/whitened embeddings to be worth revisiting.
- IVF **build** at 116 M (train 2 M sample + add) extrapolates to ~5–15 min — a one-time cost.

## 6. Are the RAG/vector-DB libraries "FAISS underneath"? (verified, 2026)

Mostly **no**. Verified against official docs + GitHub source for 10 systems:

| System | Engine | FAISS? |
|---|---|---|
| **Elasticsearch** | Lucene Java HNSW + BBQ quantization + native SIMD | **No** (benchmarks *against* faiss) |
| **OpenSearch** | k-NN plugin, **default faiss** (JNI C++) / Lucene / nmslib(dep.) | **Yes** (default) |
| **Milvus** | Knowhere (C++): patched faiss fork for IVF/flat only; HNSW/DiskANN/GPU-cuVS separate | **Partial** |
| **Qdrant** | custom Rust filterable HNSW | No |
| **Weaviate** | custom Go HNSW | No |
| **Chroma** | Rust core + hnswlib fork | No |
| **pgvector** | custom C HNSW + IVFFlat | No |
| **Vespa** | custom C++ modified HNSW | No |
| **Pinecone** | proprietary Rust (LSM slabs, anti-HNSW) | No |
| **LanceDB** | Rust, Lance IVF-PQ on disk | No |

Takeaways: the **industry default is HNSW**, but as *N independent re-implementations*, because
production DBs need mutable graphs (live insert/delete), filter-aware traversal, and durability —
things a static library index (faiss/hnswlib) doesn't give (Qdrant, Pinecone explicitly rejected
faiss). Convergent 2026 trends: 1-bit-ish binary quantization + rescoring (~32×), disk-resident
IVF/DiskANN for beyond-RAM, and **GPU used only for index *building*** (cuVS/CAGRA → converted to a
CPU HNSW) — **no system does GPU query-time search**.

## 7. Recommendations

### 7.1 Magpie interactive (latency-sensitive demo)
- **Small corpora (QReCC/TopiOCQA, fit VRAM): switch the resident search to keep the index
  GPU-resident** (config c or d) instead of re-streaming per request — measured ≈ `82`× p50,
  zero recall change (and switch the query to fp32 for +0.15 recall — config c). Cost: a modest engineering change to `ram_index` (upload shards once at
  activate, reuse per turn).
- **ClueWeb (does NOT fit VRAM)**: keep the streaming exact path as default; optionally offer an
  **IVF-PQ** demo index (clearly labelled approximate, recall ≈ `~0.93 (SQ8; PQ collapses)`) for snappy latency.
  This is a product decision, not a correctness one.
- Cheap, exact wins regardless: re-merge the ClueWeb-Qwen index into ~1 M-doc blocks (load peak
  320→239 GB); consider mmap'd safetensors/npy blocks for faster, safer loads than pickled fp32.

### 7.2 Offline eval (feeds papers)
Must stay **exact** — ANN is not acceptable for reported numbers. But building a GPU-resident
fp16 index **once per corpus** and reusing it across a multi-config sweep amortizes the disk stream;
break-even ≈ `~2` configs sharing a corpus. Keep `merge_compat` for bit-exact paper repro.

### 7.3 Do NOT adopt a vector DB
For static, read-only, single-node research indexes on GPUs, every surveyed system is a worse fit:
ANN-first (recall<100%, nondeterministic), **CPU-only query**, and value-adds (CRUD/filtering/
sharding/multitenancy) that Magpie doesn't need. Only revisit if Magpie becomes a multi-user product
needing metadata filters or hybrid BM25+dense at scale — then Qdrant/Vespa (filter-aware,
self-hostable, no faiss baggage) alongside, not replacing, the exact eval path.

## Appendix
Env manifest + raw rows: `results.jsonl` (`_env`/`_fp16_load` records). Config definitions:
`bench_dense.py`. Vector-DB survey sources: workflow `vector-db-landscape` (per-system primary docs).
