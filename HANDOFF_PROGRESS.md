# HANDOFF / PROGRESS (resume guide)

Context was running low; this captures everything to resume. Two parallel threads:
**(A) finish the paper table** (the /goal), **(B) implement the shared-corpus framework**.

---

## A. The table (GOAL — almost done)

New table **"(Personalized) Conversational Dense Retrieval"** = `\label{table: pers_conv_dense}`
in `/data/rech/huiyuche/Continual_learning_IR_Data/main.tex` (right after Table 4 / `main_results`).
4 row-groups: base-qwen3 original prompt (3 rows), base-qwen3 v3 prompt (3), conv-qwen3 instruct3 (3),
conv-ANCE (1). **qwen numbers already filled.** The conv-ANCE row is the placeholder
`FullConv & -- & -- ... --` (12 cells).

**LAST STEP:** when conv-ance finishes (3 runs: `full_conversation` × ikat_23/24/25), parse
`results/ClueWeb_ikat/ikat_2{3,4,5}_test/metrics/S1[full_conversation]-S2[none]-g[none]-[conv-ance]-[none_4_1_none]-[s2_top50].json`
(keys recip_rank, ndcg_cut_3, recall_10, recall_100, ×100), replace the `--` row in main.tex, then:
standalone-compile to verify (the full doc has pre-existing hyperref/`\custo` breakage — compile the
table in isolation like before), `git add main.tex && commit && push` (branch `main`, repo
`Continual_learning_IR_Data`). **Restore `main.bbl` after compiling** (bibtex empties it — `git checkout -- main.bbl`).

conv-ance was running under a watchdog (`b4oakbqjh`; ~47min/experiment, ANCE index on shared disk).
If the run died/hung, re-launch (auto-reuse skips done):
`cd src && export PATH=/data/rech/huiyuche/envs/trec_ikat/bin:$PATH && python -m apcir.evaluate.run_experiments --config ./apcir/evaluate/fuse_then_eval_config_continual_ir_ft.yaml 2>&1 | tee ../logs/run_convance.log`

### Parsed result grid (MRR / NDCG@3 / R@10 / R@100, ×100) — qwen all done
```
base-qwen3 OLD prompt   Conv      23 13.7/6.3/2.2/7.4   24 26.7/10.9/1.8/9.6   25 40.8/16.3/4.7/16.2
                        +PTKB     23 18.1/7.3/2.6/8.1   24 20.6/7.7/1.4/8.2    25 33.1/11.7/3.2/12.8
                        +PrevConv 25 17.6/7.6/2.0/7.6
base-qwen3 NEW(v3)      Conv      23 20.9/9.1/2.8/9.4   24 32.2/12.4/2.5/13.6  25 44.3/18.9/5.1/19.5
                        +PTKB     23 25.0/10.8/3.2/11.3 24 33.0/14.9/3.0/14.9  25 46.6/19.4/5.7/19.2
                        +PrevConv 25 32.5/13.0/4.1/14.8
conv-qwen3 (instruct3) Conv      23 32.3/15.7/4.0/16.6 24 58.4/27.7/5.7/26.5  25 52.2/23.3/6.9/24.9
                        +PTKB     23 31.6/15.7/4.1/16.9 24 59.1/28.6/6.0/26.3  25 57.9/26.6/7.1/28.5
                        +PrevConv 25 56.7/27.0/7.3/28.6
conv-ance               FullConv  <PENDING>
```
Story: v3 prompt > old prompt; conv-qwen3 (FT) >> base-qwen3.

---

## B. Shared-corpus framework (STARTED)

Full design: **`SHARED_CORPUS_FRAMEWORK_DESIGN.md`** (in this repo). Thesis = "stream once, fan out":
group experiments by corpus, encode each job's queries, **stack** into one matrix, stream the corpus
**once**, slice results per job. ~Nx (12x for the qwen sweep) on the dominant disk term.
Hard constraint: corpus 478G > RAM 251G > GPU 96G ⇒ no residency; daemon/SHM/fp16 NOT needed (deferred).
Both disks slow (~150 MB/s) ⇒ disk-bound ⇒ stream-once is the win. Cross-corpus parallel only helps on
DIFFERENT physical disks AND if still disk-bound after stacking (marginal; Phase 3, optional).

Savepoint before refactor: **tag `avant_refactoration`** (commit c59942e, pushed to `yuchen`).

### Done
- `src/apcir/search/grouped/__init__.py`
- `src/apcir/search/grouped/merge.py` — `merge_compat` (byte-identical to legacy two-pointer,
  deepcopy removed as proven-noop) + `merge_topk` (fast numpy running top-K, ~100x; FAISS-pipeline opt).
- `src/apcir/search/grouped/test_merge.py` — **GATE 1 PASSED** (compat==legacy byte-identical; topk matches).

### Next (Phase 0/1 of the design)
1. `grouped/spec.py`: `CorpusKey(index_dir, embed_dim, block_num)` (encoder EXCLUDED → diff encoders same
   index = one group), `ExperimentSpec(args namespace + key + file_name_stem)`, `expand_specs(config)`.
   **Extract** the Cartesian-product + param_mapping overlay from `run_experiments.py:78-99` into this
   (the ONE edit to an existing file; subprocess loop `:103-107` stays). Build the per-experiment arg
   Namespace by feeding the merged config dict through `evaluation.py`'s argparse (reuse exact types/defaults;
   factor `evaluation.py`'s parser into `build_parser()` if needed).
2. `grouped/block_source.py`: `PickleBlockSource` = lift `dense_search.py:98-113` load (`doc_emb_block.{i}.pb`
   / `doc_embid_block.{i}.pb`), yield one `Block(emb, ids, block_id)` at a time. `prefetch_depth=1` default
   (6×75G blocks too big to prefetch; assert `depth*block_bytes < ram_budget`).
3. `grouped/encoder.py`: `QwenEncoder`/`AnceEncoder` wrap `get_test_query_embedding(args)` (set per-spec
   args, call it, return its `(embeddings, embedding2id)`). conv-qwen3==Qwen (dim 1024), conv-ance==Ance (768).
4. `grouped/index.py`: `FaissFlatIPFactory.new_empty(dim)` wraps `build_faiss_index(args)` (OPT-2: once per group).
5. `grouped/sink.py`: `TrecEvalSink` = `get_run_object_and_save_ranking_list(hits,args)` (search.py:45) +
   `evaluate(...)` (evaluation_util.py:191) → metrics JSON at the ikat-results-layout stem.
6. `grouped/runner.py`: `GroupRunner.run(group)` = PHASE A encode each spec → stack `Q`; PHASE B one stream
   over blocks (index.add → index.search(Q,topN) → merge → index.reset ONCE/block); PHASE C slice
   `D[lo:hi],I[lo:hi]` per spec → `get_dense_ranking_list` → sink. **`index.reset()` once per block, never
   per job** (assert `index.ntotal==n_block` before search, `==0` after reset).
7. `grouped/scheduler.py` + `evaluate/run_experiments_grouped.py` (new opt-in entry, same yaml).

### Validation gates (design §6) — must pass before paper use
- G1 merge byte-identical — **DONE**.
- G2 single-job: run one spec via legacy `evaluation` AND via `GroupRunner` (group size 1); ranking files
  **byte-identical**.
- G3 stacked: same spec inside the full 12-job group; sliced ranking == G2's. **Byte-identical**.
- G4 full sweep: run `fuse_then_eval_config_qwen_conv.yaml` (12 qwen) + continual_ir_ft (3 ance) via
  `run_experiments_grouped.py`; all metrics JSON == legacy at full float precision.
Use `merge_compat` for G2-G4 (byte-identity); switch default to `merge_topk` after.

### Key existing functions (reuse AS-IS)
- `dense_search.py`: `build_faiss_index:28`, `get_test_query_embedding:270` (returns `(emb, embedding2id)`),
  `get_dense_ranking_list:363`, merge math `search_one_by_one_with_faiss:162-212` (the 2*topN two-pointer + deepcopy).
- `search.py`: `get_run_object_and_save_ranking_list:45`; dense dispatch `:604` (`in ["ance","dpr","qwen3","conv-ance","conv-qwen3"]`).
- `evaluation.py`: `get_query_list:462`, `evaluate(...):549`, file_name_stem `:379`.
- `evaluation_util.py`: `get_query_list:29` (8-tuple), `evaluate:191`.
- `run_experiments.py`: expand at `:78-99`, subprocess loop `:103-107`.

### FINDINGS (bugs / optimizations to report)
- **OPT-1** merge: Python two-pointer + `copy.deepcopy` + list-of-tuples → numpy `merge_topk` (~100x). DONE in merge.py.
- **NOTE-1** the legacy `2*topN` "drop lower half each block" is NOT a correctness bug for the used
  top-`retrieval_top_k` (verified); just wasted buffer (only top_k needed).
- **OPT-2** `build_faiss_index` builds `StandardGpuResources` per call → grouped runner builds once per group.
- (Already fixed in the refactor: duplicate shadowed `Retrieval_topiocqa` class removed; `"[SEP]"`-join
  placeholder hack → JSON; the `max_query_length=self.max_length` cap bug gone.)
- TODO while implementing: keep scanning for more (per user: not just faiss).

### Machine / data facts
- Host octal31 (4× A5000 24G, 251G RAM, 64 cores). `/part/01` local sda1 **~158 MB/s (slow!)**, scratch
  `/part/01/Tmp/yuchenhui`. `/data/rech` shared NFS ~140 MB/s, 12T free. env `/data/rech/huiyuche/envs/trec_ikat` (py3.12).
- qwen index: `/part/01/Tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged` (6 blk, dim1024) AND shared
  `data/embeddings/clueweb22b_ikat23_qwen_merged`. ance index: `data/embeddings/clueweb22b_ikat23_ance_merged`
  (12 blk, dim768, shared). Both merged via `src/apcir/indexing/dense/merge_clueweb_index.py`.
- Configs: `fuse_then_eval_config_qwen_conv.yaml` (qwen3+conv-qwen3 × {qwen_conversation,qwen_conversation_ptkb} × 3yr),
  `_qwen_prevconv.yaml` (× previous_conv × 25), `continual_ir_ft.yaml` (conv-ance full_conversation × 3yr).
- conv-qwen3 encoder = `huggingface/continual_ir/instruct3_qwen_nosched/checkpoint-step-1880`;
  conv-ance encoder = `huggingface/continual_ir/ance_topiocqa_nosched/checkpoint-step-1900`.
- Watchdog idiom for long runs (avoid silent overnight hangs): poll metrics-file count + eval-log staleness
  (>25min = hang) in a `run_in_background` bash loop; **never `pkill -f` a pattern that matches your own shell**.
