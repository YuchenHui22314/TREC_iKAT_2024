# `apcir.interactive` — the Magpie search server

A long-lived **personalized conversational search service** over the APCIR retrieval stack:
a logged-in user picks corpora/retrievers/rerankers at runtime, the server **dynamically loads and
evicts** the matching in-RAM indexes (capacity-guarded), and each turn runs
QR → multi-retriever search → fusion → (rerank) → cited RAG → PTKB extraction.
The web frontend lives in a separate repo: **[pica / Magpie](https://github.com/YuchenHui22314/pica)**
(React SPA, talks to this server over HTTP/JSON only).

```
 pica (React SPA)  ──HTTP/JSON──►  search_server.py (FastAPI)
                                        │ auth/sessions/PTKB ──► store.py (SQLite)
                                        ▼
                                   pipeline.py (InteractivePipeline)
                                        │ set_active / can_serve / process_turn
                                        ▼
                     capacity.py (IndexRegistry + CapacityManager, capacity_config.yaml)
                                        │ load/evict, load-peak guarded
                          ┌─────────────┼──────────────┬────────────┐
                          ▼             ▼              ▼            ▼
                    ram_index.py   splade_index.py   Lucene      LLM client
                    (dense fp16)   (learned sparse)  (BM25+doc)  (openai gpt-5-mini / vllm)
```

`driver.py` is a separate concern: the iKAT'26 **Sim.API** conversation driver that talks to this
server for TREC submissions (see `logs/runs_2_3_commands.sh` for the submitted configurations).

## Files

| File | Owns |
|---|---|
| `search_server.py` | FastAPI app factory `create_app(...)`: all endpoints, auth dependency, `_persist_turn` |
| `pipeline.py` | `InteractivePipeline`: transactional `set_active`, per-turn `process_turn`, leg routing (`_encoder_for`, `_dense_group_search`), enrichment, `models_status()` |
| `capacity.py` | `IndexFootprint` (+ `index_dir_alts` first-existing resolver, `query_encoder`), `IndexRegistry.from_yaml`, `CapacityManager.plan` (checks **load-peak**, not resident, vs live free RAM/VRAM) |
| `capacity_config.yaml` | every loadable unit: dense/sparse/splade indexes per corpus + reranker; paths with NFS fallbacks |
| `ram_index.py` | `RamBlockSource` (all blocks → CPU RAM, fp16), GPU search (`search_query_against_ram`, `_search_ram_fp16_gpu`) |
| `splade_index.py` | in-RAM SPLADE inverted index |
| `store.py` | SQLite: users (PBKDF2), bearer tokens, sessions/turns, per-user PTKB |
| `driver.py` | iKAT'26 Sim.API driver (online QR + RAG + dynamic PTKB) |
| `tests/` | 76+ unit tests (`pytest apcir/interactive/tests` — **run from `src/`**) |

## Endpoints

| Route | What |
|---|---|
| `POST /auth/login` · `POST /auth/logout` · `GET /auth/me` | bearer-token auth (SQLite-backed) |
| `GET /models` | every registered unit: kind, corpus, resident?, `available` (any path exists), query encoder(s) |
| `POST /activate` → `GET /activate/status/{id}` | change the active set (one at a time, capacity-guarded, background load with per-block progress) |
| `POST /search` | one conversational turn: legs (`{name, unit, query_type, encoder_path}`), fusion, rerank, cited RAG; optional `session_id` + `extract_ptkb` persist the turn |
| `GET /doc?docid=...` | passage text via the resident Lucene doc-fetch (409 if no sparse unit resident); query param because docids contain `/` |
| `GET/POST /sessions` · `PATCH/DELETE /sessions/{id}` · `GET /sessions/{id}/turns` | session CRUD + faithful reload (turn payload stores fused `hits`, per-retriever lists, citations, `extracted_ptkb`) |
| `GET/POST /ptkb` · `PUT/DELETE /ptkb/{id}` | user-profile facts (manual + auto-extracted) |
| `GET /health` | liveness + `llm_ready` |

## Run it

```bash
# from the pica repo (serves the built SPA + this API on one port):
MAGPIE_PORT=8500 MAGPIE_ADMIN_PW=... CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python server/run_magpie.py
```

Env: `MAGPIE_PORT` (default 8500) · `MAGPIE_ADMIN_PW` (seeds the admin user) · `MAGPIE_DB`
(SQLite path) · `openai_key` (gpt-5-mini RAG + PTKB extraction) · `CUDA_VISIBLE_DEVICES`.
The launcher configures `PipelineConfig(llm_backend="openai", llm_model="gpt-5-mini",
cite_passages=True, llm_reasoning_effort="minimal")` — the OpenAI client is resource-free, so it is
built eagerly even in dynamic mode.

## Corpus units & what they cost (see `capacity_config.yaml`)

Dense doc embeddings are fp32 on disk, loaded **fp16** into CPU RAM; GPU search streams fp16
(`fp16_torch` backend). Load peak ≈ resident + one fp32 block being cast. Paths prefer the local
SSD (`/part/01/...`) and fall back to the NFS store (`data/embeddings/`, `data/indexes/`) via
`index_dir_alts` (first existing path wins; `available` in `/models` reflects it).

| Corpus | Doc encoder | #docs | dim | blocks | fp32 disk | fp16 resident | load peak |
|---|---|---:|---:|---:|---:|---:|---:|
| ClueWeb22-B | ANCE | ~117 M | 768 | 12 | 336 GB | 168 GB | 210 GB |
| ClueWeb22-B | Qwen3-Embedding-0.6B | ~117 M | 1024 | 6 | 450 GB | 235 GB | 320 GB |
| ClueWeb22-B | SPLADE-v3 (inverted) | ~117 M | — | — | — | 235 GB | 250 GB |
| ClueWeb22-B | BM25 (Lucene, on-disk) | ~117 M | — | — | — | 5 GB | 5 GB |
| QReCC | ANCE | ~56 M | 768 | 55 | 161 GB | 80 GB | 85 GB |
| QReCC | Qwen3-Embedding-0.6B | ~56 M | 1024 | 55 | 214 GB | 107 GB | 112 GB |
| TopiOCQA | ANCE | ~26 M | 768 | 26 | 74 GB | 37 GB | 42 GB |

(The ClueWeb-Qwen 320 GB peak comes from its 6 coarse ~80 GB blocks — re-merging into ~1M-doc
blocks would drop the peak to ≈ resident + 4 GB.)

## Load modes (dense units)

Each dense unit loads in one of three modes (panel dropdown "load as"; `/activate` `modes`):
`ram_fp16` (exact, streams the fp16 store per query — slow, eval-faithful), `gpu_resident`
(exact, fp16 shards resident on GPUs, ~100x faster, needs VRAM = the store), and `pq_refine`
(prebuilt IVF-PQ64 candidates on ONE GPU + exact fp32 rescore from an INT8 RAM store — measured
NDCG@3 within 0.5% of exact on iKAT'23, ~9G VRAM, ClueWeb-capable on octal31). Full design
rationale, alternatives considered, hyperparameters, NVMe storage layout and runbook:
**[`docs/dense_load_modes.md`](../../../docs/dense_load_modes.md)** + measurements in
**[`docs/dense_search_benchmark_report.md`](../../../docs/dense_search_benchmark_report.md)**.

## Invariants worth knowing

- **One corpus per active set**, and at most one unit per singleton kind (sparse / splade /
  reranker / llm). Enforced in `set_active`; `can_serve` preflights `/search` (409 with a reason).
- **Capacity checks the load PEAK** against *live* free RAM/VRAM, not the steady state.
- **Per-unit query-encoder routing**: a dense leg's encoder resolves as
  leg `encoder_path` → unit footprint `query_encoder` → global default (`_encoder_for`).
  This exists because the global default once silently encoded Qwen units with ANCE (bug #1).
  Query encoders are cached per (path, device) in VRAM (~1–2 GB each) and are **not**
  capacity-accounted.
- **`_persist_turn` extracts first**, then saves — the turn payload carries `extracted_ptkb`
  (plus full fused `hits`, per-retriever lists, citations) so a reloaded session re-renders
  identically, including the "🪶 learned" line. An extraction failure never loses the turn.
- Dense docids are stored as `np.array(dtype=object)` — a fixed-width unicode array of QReCC's
  long URL docids would cost ~25 GB/block (the OOM we hit).
- CPU faiss is **forbidden** for dense search (far too slow at ClueWeb scale) — GPU only,
  sharded across `faiss_n_gpu`.

## Tests

```bash
cd src && CUDA_VISIBLE_DEVICES="" python -m pytest apcir/interactive/tests -q
```

Unit tests inject an in-memory `Store` and fake capacity probes; no GPU or index needed.
The full-pipeline e2e (real encoder + index) is exercised via `qrecc_ance_mini` + `qrecc_bm25`
(tiny units registered exactly for this).
