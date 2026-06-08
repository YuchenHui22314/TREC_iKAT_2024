"""Opt-in shared-corpus entry point. Same YAML as run_experiments.py, but experiments that
share a doc index are streamed ONCE (grouped) instead of N separate subprocess re-reads.

Run (from src/):
  python -m apcir.evaluate.run_experiments_grouped --config ./apcir/evaluate/fuse_then_eval_config_qwen_conv.yaml
  # --merge compat  -> byte-identical to legacy (validation); --merge topk -> fast numpy (default after gates pass)

The legacy run_experiments.py is untouched and remains the reference path.
"""

import argparse
import yaml

from apcir.search.grouped.spec import expand_specs, partition_by_corpus
from apcir.search.grouped.runner import run_group
from apcir.search.grouped.merge import merge_compat, merge_topk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="fuse_then_eval YAML (same as run_experiments.py)")
    ap.add_argument("--merge", choices=["compat", "topk"], default="compat",
                    help="compat = byte-identical to legacy (validate with this); topk = fast numpy")
    a = ap.parse_args()

    config = yaml.safe_load(open(a.config))
    specs = expand_specs(config)
    groups = partition_by_corpus(specs)
    merge_fn = merge_compat if a.merge == "compat" else merge_topk

    print(f"[grouped] {len(specs)} experiments in {len(groups)} corpus group(s); merge={a.merge}")
    for i, (key, group) in enumerate(groups.items(), 1):
        print(f"[grouped] === group {i}/{len(groups)}: dim={key.embed_dim} blocks={key.block_num} "
              f"idx=.../{key.index_dir.split('/')[-1]} : {len(group)} jobs ===")
        run_group(group, merge_fn=merge_fn)
    print("[grouped] DONE")


if __name__ == "__main__":
    main()
