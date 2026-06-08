"""
Merge per-rank ClueWeb dense embeddings (doc_emb_block.rank_{r}.{b}.pb) into the consecutive
doc_emb_block.{id}.pb layout the searcher needs. Reusable CLI wrapper around
distributed_dense_index.merge_blocks_to_large_blocks. See skill: merge-clueweb-dense-index.

Block size by RAM budget: expected ~= RAM_budget / (2 * dim * 4 bytes).
  200 GB, qwen dim 1024 -> ~24M -> use 20M -> 6 blocks (~164 GB peak)
  200 GB, ance dim  768 -> ~32M -> use 10M -> 12 blocks (~61 GB peak)
expected MUST be a multiple of the 250K per-rank sub-block size so each flush lands on a
boundary (num_remain == 0) and the function's latent residual list/ndarray bug never fires.

Example (qwen, input = local /part unmerged, output = shared /data/rech embeddings):
  python src/apcir/indexing/dense/merge_clueweb_index.py \
    --input  /part/01/Tmp/yuchenhui/indexes/clueweb22B_qwen_emb_0.6 \
    --output /data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/clueweb22b_ikat23_qwen_merged \
    --expected 20000000
"""
import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distributed_dense_index import merge_blocks_to_large_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-rank source dir (doc_emb_block.rank_{r}.{b}.pb)")
    ap.add_argument("--output", required=True, help="merged output dir (doc_emb_block.{id}.pb)")
    ap.add_argument("--num_block", type=int, default=117)
    ap.add_argument("--num_rank", type=int, default=4)
    ap.add_argument("--expected", type=int, required=True,
                    help="docs per merged block; MUST be a multiple of the per-rank sub-block size (250K)")
    args = ap.parse_args()

    assert args.expected % 250_000 == 0, \
        f"--expected ({args.expected}) must be a multiple of 250000 to keep num_remain==0 (avoids the merge bug)"

    os.makedirs(args.output, exist_ok=True)
    print(f"[merge] input={args.input}\n[merge] output={args.output}\n"
          f"[merge] num_block={args.num_block} num_rank={args.num_rank} expected={args.expected}",
          flush=True)
    merge_blocks_to_large_blocks(
        input_folder=args.input,
        output_folder=args.output,
        num_block=args.num_block,
        num_rank=args.num_rank,
        expected_num_doc_per_block=args.expected,
    )
    n = len([f for f in os.listdir(args.output)
             if f.startswith("doc_emb_block.") and f.endswith(".pb") and "rank_" not in f])
    print(f"[merge] DONE — {n} merged blocks written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
