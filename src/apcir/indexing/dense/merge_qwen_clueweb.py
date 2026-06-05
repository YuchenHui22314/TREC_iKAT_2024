"""
Merge the UNMERGED per-rank Qwen3-Embedding ClueWeb22-B blocks into the searchable
`doc_emb_block.{id}.pb` layout the retriever expects.

The encoder produced 4 ranks x 117 blocks (doc_emb_block.rank_{r}.{b}.pb, 1M docs/block,
116,838,987 total) under data/embeddings/qwen_clueweb22B. The searcher
(search/dense_search.py) reads consecutive `doc_emb_block.{id}.pb`. We merge into
**6 blocks** of 20M docs each (last ~16.8M). Block size is chosen for a ~200GB-RAM
"portable" budget: merge RAM peak ~= 2 * expected * 1024 * 4B ~= 164GB at 20M (fits a
200GB server with headroom), vs 12 blocks/82GB at 10M. Search loads one block at a time
(~82GB at 20M). Set passage_block_num=6 for qwen3 accordingly (ANCE index stays 12).

We call merge_blocks_to_large_blocks DIRECTLY (not the __main__ --do_merge path) because
__main__ couples num_block to total/num_docs_per_block, which can't read all 117 input
blocks AND emit 12 output blocks at once. Calling the function lets us set them independently.

With expected_num_doc_per_block=10M and uniform 1M input blocks, each flush lands exactly on
10M (10 blocks) -> num_remain==0 -> the function's residual branch never fires (it has a latent
list/ndarray bug only when num_remain>0). RAM peak ~ 10M*1024*4B ~= 41GB (x2 at concat) — fine
on this 503GB node.

Run (py3.12 env; heavy I/O ~1-3h, no GPU; in tmux):
  /data/rech/huiyuche/envs/trec_ikat/bin/python \
    src/apcir/indexing/dense/merge_qwen_clueweb.py 2>&1 | tee logs/merge_qwen_clueweb.log
Done-check: ls .../clueweb22b_ikat23_qwen_merged/doc_emb_block.[0-9]*.pb | wc -l  == 12
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distributed_dense_index import merge_blocks_to_large_blocks

INPUT  = "/data/rech/huiyuche/TREC_iKAT_2024/data/embeddings/qwen_clueweb22B"
OUTPUT = "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_qwen_merged"
NUM_BLOCK = 117          # per-rank block_ids 0..116
NUM_RANK  = 4
EXPECTED_PER_BLOCK = 20_000_000   # -> 6 output blocks (5 x 20M + 1 x ~16.8M); ~164GB RAM peak

if __name__ == "__main__":
    print(f"[merge] input  = {INPUT}", flush=True)
    print(f"[merge] output = {OUTPUT}", flush=True)
    print(f"[merge] num_block={NUM_BLOCK} num_rank={NUM_RANK} expected_per_block={EXPECTED_PER_BLOCK}", flush=True)
    t0 = time.time()
    merge_blocks_to_large_blocks(
        input_folder = INPUT,
        output_folder = OUTPUT,
        num_block = NUM_BLOCK,
        num_rank = NUM_RANK,
        expected_num_doc_per_block = EXPECTED_PER_BLOCK,
    )
    print(f"[merge] DONE in {(time.time()-t0)/60:.1f} min", flush=True)
