"""Resident-store keeper daemon for `pq_refine` units.

The INT8 rescore store is served via SHARED file-backed mmap (RamBlockSource(int8_mmap=True)):
its pages live in the OS page cache, so they are shared across processes and survive Magpie
server restarts — attaching is ~instant while the pages stay warm. This daemon's only job is to
keep them warm: it faults every page in once (first pass = one sequential NVMe read) and then
re-touches periodically so memory pressure can't silently evict the store.

Run (octal31, ClueWeb-Qwen):
  nohup python -m apcir.interactive.store_daemon \
      --index-dir /var/tmp/yuchenhui/indexes/clueweb22b_ikat23_qwen_merged --blocks 6 \
      > ../logs/store_daemon.log 2>&1 &

A touch pass that completes near RAM speed (seconds) means the store was already resident; a slow
pass means pages had been evicted and were re-read from disk. Stop the daemon any time — nothing
depends on it except warmth.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

_PAGE = 4096


def _touch_file(path: str) -> int:
    """Fault every page of `path` into the page cache (read 1 byte per page). Returns bytes."""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    arr = np.memmap(path, dtype=np.uint8, mode="r")
    int(arr[::_PAGE].sum())          # one byte per page -> every page faulted, minimal CPU
    del arr
    return size


def touch_store(index_dir: str, num_blocks: int) -> int:
    """One warm pass over the pq_refine artifacts of `index_dir`: int8 blocks + scales (+ the
    PQ index file and docid pickles when present). Returns total bytes touched. Idempotent."""
    total = 0
    for i in range(num_blocks):
        for pat in (f"doc_emb_int8_block.{i}.npy", f"doc_emb_int8_scale.{i}.npy",
                    f"doc_embid_block.{i}.pb"):
            p = os.path.join(index_dir, pat)
            if os.path.exists(p):
                total += _touch_file(p)
    for p in glob.glob(os.path.join(index_dir, "ivfpq64.faiss*")):
        total += _touch_file(p)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--interval", type=int, default=600, help="seconds between warm passes")
    args = ap.parse_args()
    while True:
        t0 = time.time()
        n = touch_store(args.index_dir, args.blocks)
        dt = time.time() - t0
        print(f"[store_daemon {time.strftime('%H:%M:%S')}] touched {n / 1e9:.1f} GB in {dt:.1f}s "
              f"({n / 1e9 / max(dt, 1e-9):.1f} GB/s{'  (was warm)' if dt < 30 else '  (re-read)'})",
              flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
