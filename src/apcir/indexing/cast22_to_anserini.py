"""Flatten the official CAsT-2022 jsonlines output into Anserini-indexable passage records.

The official `corpus_processing/main.py --output_type jsonlines` writes ONE LINE PER DOCUMENT
with the passages nested:

    {"id": "KILT_20988744", "url": ..., "title": "Worlingworth",
     "contents": [{"body": "...", "id": 1}, {"body": "...", "id": 2}, ...]}

Anserini's JsonCollection instead needs one record per indexable unit with `contents` as a
STRING. CAsT-2022 judges passages, so each passage becomes its own record with the official
id form `<docid>-<passage_id>` (hyphen, matching the 2022 qrels, e.g. MARCO_02_1687136851-3;
note 2021's released collection used an underscore instead).

The title is prepended to every passage's text, for the same reason as in 2021: the body does
not repeat it, so a passage is otherwise unreachable by a query naming its document. The url is
kept as a separate JSON key and therefore stays out of the index.

Usage
-----
    cd src
    python -m apcir.indexing.cast22_to_anserini \
        --in  /part/01/Tmp/yuchen/cast22_build/jsonlines \
        --out /part/01/Tmp/yuchen/cast22_build/anserini --workers 8
"""
import argparse
import glob
import gzip
import json
import os
from multiprocessing import Pool


def convert(job):
    src, dst, index_title = job
    n_doc = n_pas = 0
    tmp = dst + ".part"
    with open(src, encoding="utf-8") as fi, \
         gzip.open(tmp, "wt", encoding="utf-8", compresslevel=4) as fo:
        for line in fi:
            d = json.loads(line)
            did = d["id"]
            title = (d.get("title") or "").strip()
            n_doc += 1
            for p in d.get("contents") or []:
                body = (p.get("body") or "").strip()
                if not body:
                    continue
                text = f"{title} {body}" if (index_title and title and title != ".") else body
                fo.write(json.dumps({
                    "id": f"{did}-{p['id']}",          # hyphen: the 2022 qrel id form
                    "contents": text,
                    "title": title,
                    "url": d.get("url", ""),
                }, ensure_ascii=False) + "\n")
                n_pas += 1
    os.replace(tmp, dst)
    return os.path.basename(src), n_doc, n_pas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no_index_title", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for src in sorted(glob.glob(os.path.join(a.inp, "*.jsonl"))):
        dst = os.path.join(a.out, os.path.basename(src) + ".gz")
        if os.path.exists(dst):          # resumable
            continue
        jobs.append((src, dst, not a.no_index_title))
    print(f"{len(jobs)} files to convert", flush=True)

    td = tp = 0
    with Pool(a.workers) as pool:
        for name, nd, np_ in pool.imap_unordered(convert, jobs):
            td += nd; tp += np_
            print(f"  {name}: {nd:,} docs -> {np_:,} passages", flush=True)
    print(f"done: {td:,} documents -> {tp:,} passages")


if __name__ == "__main__":
    main()
