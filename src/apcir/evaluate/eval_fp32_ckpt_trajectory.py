"""Per-checkpoint trajectory eval for conv-qwen3-fp32 (instruct3fp32_infonce_lr1e5):
evaluate ALL 20 per-epoch checkpoints with the `qwen_conversation` (Conv) query on
iKAT 23/24/25. All 60 specs share ONE qwen CorpusKey -> grouped run_group streams the
corpus ONCE (encoder differs per spec; CorpusKey excludes it). Each spec gets a
step-tagged file_name_stem ([conv-qwen3-fp32_step{N}]) so the 20 ckpts don't collide,
while retrieval_model stays "conv-qwen3-fp32" (valid for the topics.py asserts + dispatch).

Run from src/:  PATH=.../trec_ikat/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m apcir.evaluate.eval_fp32_ckpt_trajectory
"""
import os, copy, yaml
from apcir.search.grouped.spec import expand_specs, ExperimentSpec, make_file_name_stem, partition_by_corpus
from apcir.search.grouped.runner import run_group
from apcir.search.grouped.merge import merge_topk

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--ckpt_dir", default="/data/rech/huiyuche/huggingface/continual_ir/instruct3fp32_infonce_lr1e5")
_p.add_argument("--tag", default="conv-qwen3-fp32",
                help="stem model-token base; per-epoch output stem = S1[...]-[{tag}_epoch{k}]-...")
_a = _p.parse_args()
CKPT_DIR = _a.ckpt_dir
TAG = _a.tag
EPOCHS = list(range(1, 21))   # epoch k  <->  checkpoint-step-{94*k}  (94..1880, 20 per-epoch ckpts)
CFG = "apcir/evaluate/fuse_then_eval_config_convqwen3fp32_A.yaml"

# 3 template specs (Conv on 23/24/25) carry all the correct fixed args from the config.
cfg = yaml.safe_load(open(CFG))
templates = [s for s in expand_specs(cfg) if s.args.retrieval_query_type == "qwen_conversation"]
assert len(templates) == 3, f"expected 3 Conv templates, got {len(templates)}"

specs = []
for k in EPOCHS:
    N = 94 * k
    ckpt = f"{CKPT_DIR}/checkpoint-step-{N}"
    assert os.path.isdir(ckpt), f"missing ckpt {ckpt}"
    for t in templates:
        a = copy.deepcopy(t.args)
        a.dense_query_encoder_path = ckpt
        a.save_results_to_object = False          # don't bloat the topic JSONs with 60 runs
        stem = make_file_name_stem(a).replace("[conv-qwen3-fp32]", f"[{TAG}_epoch{k}]")
        specs.append(ExperimentSpec(args=a, corpus_key=t.corpus_key, file_name_stem=stem))

groups = partition_by_corpus(specs)
print(f"[fp32-traj] {len(specs)} specs ({len(EPOCHS)} ckpts x 3 datasets), {len(groups)} corpus group(s)")
assert len(groups) == 1, "expected ONE corpus group (all share the qwen index)"
for ck, g in groups.items():
    print(f"[fp32-traj] streaming corpus once for {len(g)} specs ...")
    run_group(g, merge_fn=merge_topk)
print("[fp32-traj] DONE")
