Now I have confirmed all the disputed facts. Key findings from the code:

1. **`save_ranking_list`**: Review 1's BLOCKER 1 is downgraded — legacy gates the ranking write on `args.save_ranking_list`, and the runner never sets it. But the config's `fixed` block typically carries `--save_ranking_list`. The runner should force it True to not rely on config. Confirmed real risk.

2. **`file_name_stem_without_group`**: Confirmed at evaluation.py:384 — it's `file_name_stem` WITHOUT the personalization_group suffix. For `personalization_group="all"`, the suffix is empty, so `file_name_stem_without_group == file_name_stem`. The runner never sets `a.file_name_stem_without_group`, so `vars(a)` will be missing this key. Confirmed.

3. **`vars(args)` parity** (evaluation.py:472-500): legacy sets `ranking_list_path`, `file_name_stem`, `file_name_stem_without_group`, then deletes the 6 query-list keys before the dump. Runner sets only the first two, deletes none. Confirmed.

4. **Metrics format** (evaluation.py:606-648): all four reviews agree exactly with the code. Confirmed.

5. **`save_results_to_object`** (evaluation.py:565-600): confirmed present in legacy, absent in runner.

6. **merge_compat**: all reviews agree it's correct; deepcopy removal is a verified no-op.

7. **`run_eval` vs `run_rag`**: Note the runner gates on `a.run_eval` (line 105) but legacy's `del` block and the search live under `if args.run_rag`. The runner doesn't have a `run_rag` gate — it always searches. This is fine for these specs (retrieval-only) but worth noting the `del` must happen regardless.

Let me confirm one final disputed point — whether `query_metrics_dic` keys are qids (for the `save_results_to_object` loop) and verify the args_view subset count discrepancy between reviews (Review 1 says 9-key, Review 3 says 7-key). From the code, `args_view` has 8 keys. Minor, doesn't change the fix.

I have everything needed. Here is the synthesized fix-list.

---

# Synthesized Fix-List — `grouped/` framework byte-identity gate

All four reviews independently converge: **`merge_compat` and the ranking-file path are byte-faithful; `runner._save_metrics` and the surrounding PHASE-C arg bookkeeping are NOT.** Every divergence is in **one file**: `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/search/grouped/runner.py`. The legacy reference to mirror is `evaluation.py:472-500` (the set+del block) and `:606-648` (the metrics/per-query/per-metric dump). I re-checked all cited lines against the actual code; disagreements resolved inline.

## (1) BLOCKERS — break byte-identity (none crash; all are silent output divergence)

Ordered by how much of the deliverable they corrupt. **All live in `runner.py`'s `_save_metrics` (114-132) and PHASE C (92-111).**

**B1 — Metrics JSON format is wholly wrong.** `runner.py:127-130` writes a 3-line file (`"Print this line to your latex table\n"` + compact `averaged_metrics` + compact 8-key `args_view`). Legacy (`evaluation.py:618-632`, verified) writes: header **with trailing colon**, two `---` rules, `"    " + formatted_metrics + "\n"` (from `print_formatted_latex_metrics(averaged_metrics, args.metrics_to_print)`), then `"\n"` + `json.dump(averaged_metrics, f, indent=4)`, then `"\n"` + `json.dumps(vars(args), indent=4)`. **Fix:** rewrite `_save_metrics` to reproduce `:618-632` verbatim; import `print_formatted_latex_metrics` from `evaluation_util`. (All 4 reviews agree; R1/R3/R4 quote it identically.)

**B2 — `vars(args)` dumped by B1 will still mismatch: query-list keys never deleted.** `_encode_spec` (`runner.py:31-36`) sets `retrieval_query_list / reranking_query_list / fusion_query_lists / qid_list_string / qid_personalized_level_dict / qid_weights_dict` on `a` and never deletes them. Legacy deletes exactly these (`evaluation.py:493-500`, the last two guarded with `if ... in args`) **after** `search()` returns and **before** the dump. **Fix:** in PHASE C, after `get_run_object_and_save_ranking_list(hits, a)` (which still needs `a.qid_list_string` at `search.py:77`) and before `_save_metrics`, `del` those six keys with the same two guards. Order matters: delete after the ranking write, exactly like legacy.

**B3 — `file_name_stem_without_group` never set on `a`.** Legacy sets it at `evaluation.py:474` (verified: `:384` computes it as the stem *without* the `_{group}` suffix; for `personalization_group="all"` it equals `file_name_stem`). Runner sets `ranking_list_path` and `file_name_stem` (`:98-99`) but not this third key, so `vars(a)` is missing it. **Fix:** `a.file_name_stem_without_group = spec.file_name_stem` in PHASE C (equal for all 15 target specs since group is `"all"`; compute the no-group variant if a grouped config ever uses a/b/c).

**B4 — Per-query JSON missing `indent=4`.** `runner.py:132` does `json.dump(query_metrics_dic, f)`; legacy (`evaluation.py:636`, verified) uses `indent=4`. **Fix:** add `indent=4`. (All 4 reviews agree.)

**B5 — Per-metric append files `metrics/{name}.txt` never written.** Legacy (`evaluation.py:640-648`, verified) appends `file_name_stem + f"-[{averaged_metrics[m]}]\n"` to `metrics/{m}.txt` for each `metrics_list_key_form` entry. Runner writes none. **Fix:** add that loop to `_save_metrics`. These are **append-only**, so a true byte-comparison must start from a clean `metrics/` dir. (R1/R2/R3/R4 agree.)

**B6 — `save_ranking_list` not forced True.** The ranking write is gated on `args.save_ranking_list` (`search.py:70`, verified), an `action="store_true"` defaulting False. The runner relies on the config's `fixed` passing `--save_ranking_list`; it does for these configs, but the runner doesn't guarantee it, and if absent the ranking `.txt` is silently empty/missing while metrics still compute (because `evaluate` uses the in-memory `run`). **Resolution of R1↔others:** only R1 flagged this; it's real but config-dependent — downgrade to a **forced-True guard**, not a confirmed break. **Fix:** `a.save_ranking_list = True` in PHASE C before the ranking write. (`a.run_name`/`run_name` token: verified OK — parser default `"none"` → writer falls back to `file_name_stem`, which the runner sets correctly. No fix.)

**B7 — `save_results_to_object` write-back to the input topic JSON is entirely missing.** Legacy (`evaluation.py:565-600`, verified) re-reads turns from `args.input_query_path`, calls `turn.add_result(...)` per qid (`response = "rag_not_run, no response."` when `generation_model=="none"`), and `save_turns_to_json(...)` — **mutating the input topic file**. Runner does none of it. **Resolution of disagreement:** R1 calls this a blocker, R3/R4 call it "not a blocker for ranking+metrics bytes." Both are right depending on scope — if the gate's byte-comparison includes the input topic JSON (and the configs set `save_results_to_object: true`), it **is** a blocker; for ranking+metrics-only it is not. **Verdict: treat as a BLOCKER** because the config enables it and legacy mutates a tracked file. **Fix:** replicate `:565-600` in PHASE C guarded by `a.save_results_to_object`; preserve spec order (legacy appends sequentially). Note: with multiple specs sharing one topic file, append in the same group order legacy would run them.

## (2) Real BUGS (latent / correctness, do not bite the 15 target specs)

**BUG-A — `topN = max(retrieval_top_k)` over the group is unsound for mixed top-k.** `runner.py:73` searches/merges all specs at the group-max `topN`, slicing to each spec's own `[:k]` only in `get_dense_ranking_list`. **Resolution R2↔R4:** R4 argues `IndexFlatIP` is exact so truncation gives identical top-k; R2 argues a larger per-block pool can change results near block-boundary ties. R4 is correct for the *final* top-k set (exact index, every candidate that would be in legacy's smaller pool is also in the larger), but R2's tie-order concern is real only if scores tie across the boundary — negligible and not exercised. **Inert for all 15 specs** (uniform `retrieval_top_k=1000`). **Fix (cheap guard):** `assert len({int(s.args.retrieval_top_k) for s in group}) == 1` in `run_group` so a future config can't silently diverge.

**BUG-B — GPU index/resources not freed between groups.** `run_experiments_grouped.py` loops `run_group` per group; `build_faiss_index` (`runner.py:77`) allocates 4× `StandardGpuResources` + a sharded GPU index each call, freed only by nondeterministic GC. Two co-resident sets (qwen-1024 then ance-768) risk OOM. Not a byte issue. **Fix:** at end of `run_group`: `index.reset(); del index; gc.collect()` (and drop the resource list) before returning. (Only R4 caught this; verified plausible.)

**BUG-C — runner gates the whole emit on `a.run_eval`, but the `del`/save-object work should not be conditioned wrongly.** Minor structural note: legacy splits `run_rag` (search + del) from `run_eval` (metrics). The runner has no `run_rag` gate (always searches) and does the `del` nowhere. When you add B2's `del`, place it outside the `if a.run_eval` block (right after the ranking write), matching legacy. Not a divergence for these specs (both flags effectively on), but get the placement right.

## (3) Optimizations / nits

- **N1** `runner.py:101` `a.retrieval_query_list = a.retrieval_query_list` — dead self-assignment; delete. (R1, R4)
- **N2** `runner.py:103` comment "returns (hits, run)" describes `get_dense_ranking_list`, which returns a single dict; the code is correct, only the comment is wrong. Fix the comment. (R2, R3)
- **N3** `merge.py:27` docstring "numpy breaks ties by stable sort" understates it — `merge_topk` runs `argpartition` *before* the stable `argsort`, so tie order is the post-partition order, not block order. Tighten wording so no one mistakes `merge_topk` for byte-identical. (R2)
- **N4** `block_source.py:31-41` `iter_blocks` ignores `self.dim`; a wrong-dim index fails only later in `index.add`. Add `assert emb.shape[1] == self.dim` on block 0 (design §6 gate-5 pre-flight). (R4)
- **N5** `spec.py:88` `parse_known_args` silently swallows misspelled eval args into `_unknown`. Safe for these configs (only `--machine` is non-eval), but log/assert `_unknown ⊆ {known-non-eval}`. (R3, R4)
- **N6** `spec.py` nargs='+' argparse defaults are shared mutable lists across specs; safe here (all three list args set in `fixed`), landmine otherwise — deepcopy the namespace or re-instantiate per spec. (R4)
- **N7** Optional literal-parity nit: legacy `D.tolist()` makes Python floats; runner passes numpy float32 into `merge_compat`. Values/strings identical (`float()` cast downstream), so no byte change — only add `.tolist()` if you want intermediate-type parity. (R3)
- **Confirmed correct, no action** (do not re-litigate): merge fold order + `>=` tie rule + `2*topN` width + first-block skip + deepcopy-removal no-op; `ids[I]` mapping with asserted reset discipline; `Q` float32 concat + slice↔qid alignment; `make_file_name_stem` (== `evaluation.py:382`, pre-routing); the 4 subdir creates; `evaluate` not reading the file since `run` is non-None.

## (4) Verdict — ready for G2 empirical validation?

**Not yet for a full byte-identity gate — but the merge/search/ranking core is ready, and the gap is small, mechanical, and confined to `runner.py`.** The four reviews unanimously confirm the load-bearing science (`merge_compat`, reset discipline, encoder-agnostic grouping, stem computation, ranking-file bytes) is faithful. Every failure is in the metrics/args/side-effect bookkeeping.

**Minimal changes required before G2 (all in `/data/rech/huiyuche/TREC_iKAT_2024/src/apcir/search/grouped/runner.py`):**
1. Rewrite `_save_metrics` to mirror `evaluation.py:618-648` exactly — header (colon + 2 rules + `formatted_metrics`), `indent=4` on both `averaged_metrics` and `vars(a)`, the per-metric `.txt` appends (**B1, B4, B5**).
2. In PHASE C, before `_save_metrics`: set `a.save_ranking_list = True` and `a.file_name_stem_without_group = spec.file_name_stem`; after the ranking write, `del` the six query-list keys (**B2, B3, B6**).
3. Add the `save_results_to_object` write-back block from `evaluation.py:565-600`, guarded by `a.save_results_to_object` (**B7**).
4. Delete the dead line `runner.py:101` and fix the comment at `:103` (**N1, N2**).

**Then run the validation:** one spec legacy + grouped, `diff` all artifacts that legacy emits — ranking `.txt`, metrics `.json` (incl. `vars(args)` key set/order), per-query `_dict.json`, the per-metric `.txt` appends (from a clean `metrics/` dir), and `git diff` the input topic JSON. They must be byte-identical. Add the `assert len({retrieval_top_k}) == 1` guard (**BUG-A**) and the inter-group `del index; gc.collect()` (**BUG-B**) before the multi-group qwen+ance run; defer N3-N7 (non-blocking).