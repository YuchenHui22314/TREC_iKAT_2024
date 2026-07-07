# Dense load modes & the PQ64+refine-INT8 design (2026-07-06/07)

How Magpie serves 116M-doc ClueWeb-Qwen interactively on octal31 (4×A5000 24G, 251G RAM), and why
every alternative lost. Companion measurement report: `dense_search_benchmark_report.md`
(§5.1a latency, §5.1b NDCG on QReCC, §5.1c NDCG on iKAT'23). Code: `apcir/interactive/`
(`capacity.py` modes, `ram_index.py` containers, `apcir/indexing/build_ivfpq_index.py`).

## 1. The problem

Exact fp16 search over ClueWeb-Qwen (116.8M × 1024-d) needs 237G — more than total VRAM (96G),
and streaming it from CPU RAM per query costs seconds (measured 2.73s at 26M ⇒ ~16s at 116M).
Goal: interactive latency, near-exact top-10 quality, on this box.

## 2. Solution space — what we measured and why each lost

| # | candidate | verdict | measured reason |
|---|---|---|---|
| 1 | `gpu_resident` exact fp16 shards | ✅ kept as a MODE (small corpora) | 82–119× faster, zero loss; needs VRAM = fp16 store → fits QReCC-ANCE (80G), NOT ClueWeb (178/237G) |
| 2 | IVF-SQfp16 (2 B/dim) | ✗ for ClueWeb | quality fine (NDCG parity at nprobe≥32) but 237G VRAM |
| 3 | IVF-SQ8 (1 B/dim) | ✗ for ClueWeb-Qwen | NDCG@3 = exact on QReCC; 119G > 96G VRAM (would fit ClueWeb-ANCE 89G, tight) |
| 4 | IVF-SQ4 (0.5 B/dim, 59G ✓) | ✗ alone | NDCG@3 −5.7% (qwen); fine **with refine** (−2%) but PQ64 is smaller still |
| 5 | IVF-PQ96/PQ128 alone | ✗✗ | ANCE: collapses (NDCG@3 0.02 — unnormalized, ‖doc‖≈27); qwen: −12.5% (QReCC) / **−28% (iKAT)**; PQ128 physically impossible on A5000 (needs 64K shared mem, SM86 has 48K, any faiss version) |
| 6 | **IVF-PQ64 + exact refine** | ✅ **WINNER** | candidates only from PQ (8.55G, ONE GPU); order fixed by exact rescore → iKAT'23 oracle NDCG@3 **0.1892 vs exact 0.1902 (−0.5%)**, NDCG@10 parity; only deep R@100 −9.7% (demo-irrelevant, eval-forbidden anyway) |
| 7 | refine store fp16 | ✗ on octal31 | 235G > free RAM |
| 8 | **refine store INT8 + per-row scale** | ✅ **shipped** | 119G fits; IP error <0.01 on normalized rows; the iKAT numbers above ARE int8-rescored |
| 9 | CPU HNSW (faiss HNSW32) | ✗ | build 486s per 5M (~3h at 116M), 13ms/query, recall *below* GPU IVF-SQ; industry vector-DBs are all CPU-query HNSW — adopting one throws away our GPUs (see report §6) |
| 10 | disk-gather refine (no RAM store) | ✗ on octal31 | /part HDD random reads <5 IOPS under load (measured); viable only on NVMe-class storage |
| 11 | /dev/shm resident store | ✗ | 120.6G into a 126G tmpfs — 5G headroom, and it pins half the machine's RAM |
| 12 | resident store daemon (attach on restart) | deferred | NVMe migration cut restarts to 122s; revisit only if that ever hurts |

Two cross-cutting findings that shaped everything:
- **recall-vs-GT ≠ quality.** The fp16 path's recall@10 0.83 was tie-shuffling among equally-relevant
  near-duplicates — NDCG unchanged. Conversely the pre-fix fp16-**score** path really lost −1.6 pts
  NDCG@3: fp16 output quantizes IP≈700 (unnormalized ANCE) at ulp 0.5. Fix (`dd499a8`): docs stored
  fp16, scores computed fp32. On normalized qwen even the old path was harmless.
- **Hard datasets amplify quantization damage in the CANDIDATE stage but not after refine**
  (PQ-alone −12.5% QReCC → −28% iKAT; +refine −2% → −0.5%). Refine quality is bounded by candidate
  coverage, not codec precision.

## 3. Hyperparameters (and why)

| param | value | rationale |
|---|---|---|
| PQ sub-quantizers m | **64** (×8 bit, 16-dim subvectors, 64 B/vec) | SM86 shared-mem ceiling (m=96/128 need >48K; cloner fp16-LUT flag = `GpuClonerOptions.useFloat16` — there is NO `useFloat16LookupTables` field, SWIG silently swallows it); PQ64+refine ties PQ128+refine (0.4887) |
| nlist | `2^round(log2(4·√N))` = **32768** @116.8M | faiss guideline lower bound 4√N; k-means needs ≥39·nlist samples — 32768×39≈1.28M < our 2M training sample |
| nprobe | **64** (runtime-tunable) | 64 vs 128 identical on iKAT (0.1892) — candidates saturate at 64 buckets |
| cand_k | **512** default, auto-raised to ≥topN, capped **2048** | GPU PQ64 verified OK to k=2048 (0.9–1.7ms); SQ-family GPU aborts at k>~100 on faiss 1.8 (uncatchable C++ assert; fixed in faiss 1.14 — env kept at 1.8 for eval stability, PQ64 unaffected) |
| int8 quantization | per-row symmetric, scale = max\|x\|/127 (float32 (n,1)) | IP err <0.01, top-10 overlap ≥9/10 on normalized rows (unit test); corpus-level: iKAT §5.1c |
| train sample | 2M vectors, GPU k-means (`clustering_index`) | ~42s train @116M |
| rescore | fp32 GEMM over dequantized rows | 3.6 ms/query steady-state (RAM-resident store) |
| thread safety | all searches on one GPU index hold a lock | concurrent `GpuIndexIVFPQ.search()` corrupts faiss `StackDeviceMemory` → process abort (hit live 2026-07-07, `6e4af09`) |

## 4. Storage layout & the NVMe migration (octal31)

Root cause of the old "BM25 takes tens of seconds": the ClueWeb Lucene index had NO local copy and
resolved to **NFS** (~10–150MB/s + per-read network latency). /part is a 2.7T **HDD** (150MB/s seq,
~12ms random). The unused **system NVMe** (`/`, 450G, 384G free) fixes both:

```
/var/tmp/yuchenhui/indexes/                      (NVMe, 283G, static data)
├── clueweb22b_ikat23_fengran_sparse_index_2/    Lucene 159G  (BM25 + doc-fetch)
└── clueweb22b_ikat23_qwen_merged/               124G: int8 cache (doc_emb_int8_block.*.npy
        + scales) + docid pickles + ivfpq64.faiss(+.meta.json docid fingerprint)
        — INTENTIONALLY no fp32 blocks (validate() accepts cache-only dirs for int8 loads)
```

- **Fallback chain** (`index_dir` → `index_dir_alts`, first existing dir wins): NVMe → /part →
  octal40-/part → NFS. If an admin wipes /var/tmp, nothing breaks — activation falls back to the
  13-min HDD path (int8 cache also lives on /part) and BM25 to NFS. octal40 resolution unchanged
  (the /var/tmp path doesn't exist there).
- Sources of truth stay on /part + NFS; the NVMe copies are disposable (rebuild: rsync ~20min).
- Measured wins: activation **70min (cold quantize) → 13min (HDD int8 cache) → 122s (NVMe)**;
  full query (dense+BM25+RAG) **~34s → 4.5s steady / 9.7s cold** — the ~30s delta was BM25-over-NFS
  I/O; the LLM (gpt-5-mini, reasoning_effort=minimal) costs ~2–3s of what remains.
- Stewardship: nobody else stores data on this disk (checked); 103G left; /var/tmp may have a
  30-day-unused cleanup policy → if files vanish, re-copy. Copies were made with an atomic
  tmp-name + rename so a half-copied dir can never be resolved as a valid index.

## 5. Runbook

```bash
# 1. prebuild the PQ index (once per corpus; ~75 min for ClueWeb-Qwen, GPU k-means)
python -m apcir.indexing.build_ivfpq_index \
    --index-dir /part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged --dim 1024 --blocks 6

# 2. prebuild the int8 cache offline (once; else the FIRST pq_refine activation does it, slowly)
python -c "from apcir.interactive.ram_index import RamBlockSource; \
           RamBlockSource('<index_dir>', 6, 1024, store_dtype='int8')"

# 3. (optional, octal31) copy hot data to NVMe + point capacity_config index_dir there
rsync -a <index_dir>/doc_emb_int8_* <index_dir>/doc_embid_* <index_dir>/ivfpq64.faiss* \
      /var/tmp/yuchenhui/indexes/<name>/

# 4. activate from the panel: "load as: PQ64+refine · ≈2% quality tax · fast"
#    (or POST /activate {"units":[...], "modes":{"<unit>":"pq_refine"}})
```

Mode cheat-sheet (the panel's "load as" dropdown; feasibility computed live per machine):
- `ram_fp16` — exact, slow (streams store per query); the offline-eval-faithful default.
- `gpu_resident` — exact, ~100× faster; needs VRAM = the fp16 store; small corpora only.
- `pq_refine` — ~ −0.5% NDCG@3 / −10% R@100; ~9G VRAM + int8-store RAM; ClueWeb-capable.
- NEVER use fp16/pq modes for offline eval numbers (papers) — fp32 exact only (house rule).

## 6. Night-shift addenda (2026-07-07)

**Shared-mmap store + keeper daemon (shipped).** `RamBlockSource(int8_mmap=True)` attaches the
int8 npy cache via file-backed mmap (pages = shared OS page cache, surviving restarts);
`apcir.interactive.store_daemon` keeps them warm (1 byte/page touch per `--interval`). Measured
activation chain for ClueWeb-Qwen `pq_refine`: **70 min (cold quantize) → 13 min (HDD int8 cache)
→ 122 s (NVMe read) → 31 s (warm mmap attach)** — the remaining 31 s is docid pickles (~13 s) +
PQ GPU clone + BM25 open. Capacity note: mmap'd store pages count as *available* to psutil, so the
`ram_gb` requirement stays conservative; over-admission degrades to NVMe re-reads, never OOM.

**Format measurements (deferred work, decided by data):**
- *pickle vs npy, cold sequential 3.07G on HDD*: pickle 22.1 s (139 MB/s) vs npy 26.0 s (118 MB/s)
  — **a wash; both disk-bound**. npy's real value is mmap-ability (18 ms attach), not read speed.
  The only lever that cuts load wall-clock is FEWER BYTES (fp16 = ½, int8 = ¼).
- *docid storage prototype* (offsets-uint64 + utf8-bytes npys + decode-on-access view, vs the
  current pickle→object-array): clueweb 20M-id block: load 2.2 s → **1 ms**, RAM 1.77 G → **~0**
  (×6 blocks ≈ −10.6 G, −13 s activation); gather of 1000 ids 0.14 ms → 2.46 ms (Python decode
  loop — absolute cost still invisible next to the 3.6 ms rescore). Verdict: worthwhile, same
  batch as the fp16-npy cache work — **both deferred by user decision (2026-07-07)**.

**Per-unit "search" toggle (frontend, shipped).** A resident unit can now sit out of retrieval
(checkbox next to the badge) while still serving doc-fetch — fixes "a resident BM25 unit always
runs a 1000-doc search". State is pruned to resident units on every activation (codex fix).
