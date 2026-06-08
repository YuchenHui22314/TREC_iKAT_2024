"""ExperimentSpec / CorpusKey / expand_specs for the shared-corpus framework.

Expands a fuse_then_eval YAML (the SAME config run_experiments.py consumes) into a list of
ExperimentSpec, each carrying the exact arg Namespace one `evaluation.main()` would build.
Specs sharing a CorpusKey (index_dir, embed_dim, block_num) can be streamed together — note
the ENCODER is deliberately NOT part of the key, so e.g. qwen3 + conv-qwen3 (different
checkpoints, same 1024-dim qwen doc index) form one group.
"""

import os
import itertools
from copy import deepcopy
from dataclasses import dataclass
from argparse import Namespace

from apcir.evaluate.evaluation import build_parser


@dataclass(frozen=True)
class CorpusKey:
    index_dir: str      # abspath of dense_index_dir_path
    embed_dim: int
    block_num: int


@dataclass
class ExperimentSpec:
    args: Namespace
    corpus_key: CorpusKey
    file_name_stem: str


def make_file_name_stem(args):
    """Exact replica of evaluation.py:372-382. Computed with the ORIGINAL
    retrieval_query_type (before get_query_list routes full_conversation ->
    full_conversation_dense), matching evaluation.py's ordering."""
    qe = "_rm3" if args.qe_type == "rm3" else ""
    pg = args.personalization_group
    pg = "" if pg == "all" else f"_{pg}"
    return (f"S1[{args.retrieval_query_type}{pg}]-S2[{args.reranking_query_type}]"
            f"-g[{args.generation_query_type}]-[{args.retrieval_model}{qe}]"
            f"-[{args.reranker}_{args.window_size}_{args.step}_{args.rerank_quant}]"
            f"-[s2_top{args.rerank_top_k}]")


def _merged_dicts(config):
    """Cartesian product of `iterate`, with `param_mapping` overlaid onto a copy of `fixed`,
    iterate values applied LAST (they win). Mirrors run_experiments.py:78-99."""
    fixed = config["fixed"]
    it = config["iterate"]
    pm = config.get("param_mapping", {})
    keys = list(it.keys())
    for combo in itertools.product(*[it[k] for k in keys]):
        param_dict = dict(zip(keys, combo))
        d = deepcopy(fixed)
        for pname, mapping in pm.items():
            for pval, assoc in mapping.items():
                if pname in param_dict and param_dict[pname] == pval:
                    d.update(assoc)
        d.update(param_dict)
        yield d


def _dict_to_cli(d):
    """Same surface as run_experiments.extend_command, but store_true bools become a bare
    flag (no trailing '') so argparse.parse_args(list) works without a shell."""
    cli = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        elif isinstance(v, (list, tuple)):
            # nargs='+' args (QRs_to_rank, fuse_weights, metrics_to_print): separate tokens
            # (run_experiments joins them but relies on shell word-splitting; parse_args(list) does not)
            cli.append(f"--{k}")
            cli += [str(x) for x in v]
        else:
            cli += [f"--{k}", str(v)]
    return cli


def expand_specs(config):
    """config: the loaded YAML dict (fixed/iterate/param_mapping). Returns list[ExperimentSpec]."""
    parser = build_parser()
    specs = []
    for d in _merged_dicts(config):
        # parse_known_args tolerates iterate-only keys that aren't eval args (e.g. --machine)
        args, _unknown = parser.parse_known_args(_dict_to_cli(d))
        key = CorpusKey(
            index_dir=os.path.abspath(args.dense_index_dir_path),
            embed_dim=int(args.embed_dim),
            block_num=int(args.passage_block_num),
        )
        specs.append(ExperimentSpec(args=args, corpus_key=key,
                                    file_name_stem=make_file_name_stem(args)))
    return specs


def partition_by_corpus(specs):
    """Group specs by CorpusKey. Returns dict[CorpusKey, list[ExperimentSpec]]."""
    groups = {}
    for s in specs:
        groups.setdefault(s.corpus_key, []).append(s)
    return groups
