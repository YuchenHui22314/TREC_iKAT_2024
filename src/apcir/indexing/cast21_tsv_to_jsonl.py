"""Convert the CAsT 2021 collection tarball into gzipped JSONL shards for Anserini.

Record format (verified by inspection, NOT assumed):

    <docid>\t<body>\t<title>\t<url>

i.e. FOUR tab-separated fields, not two. The passage id suffix is `_<n>`
(e.g. MARCO_D1167206_1).

TITLES ARE REAL AND MUST BE INDEXED. Measured over a ~600k-passage sample: MARCO 90.6% and
WAPO 99.9% of rows carry a genuine title (only a minority carry the placeholder "."), and every
KILT row carries its Wikipedia page name. The body does NOT repeat the title -- only 6.5% of
MARCO and 0.1% of WAPO first-passages start with it -- so a passage such as KILT_727245_2
("Pseudomorph") has a body that never contains the word "Pseudomorph" and is unreachable by a
query for it unless the title is indexed. Indexing the title is a standard IR default, not an
experiment: run with --index_title. The url is NOT indexed (kept as a JSON field only).

Two things make a naive line-by-line split wrong:

1. The body may contain a literal newline, which splits one logical record across two
   physical lines: the first holds `docid \t body-head`, the second `body-tail \t title
   \t url`. There are 28 such rows out of 40,235,494 (plus 349 blank lines). Dropping the
   continuation would silently truncate those documents' text, and treating it as its own
   record would invent a bogus docid from a headline. So a record is accumulated from a
   line matching the docid pattern up to (not including) the next such line.
2. The body may itself contain tabs, so fields are taken from the ENDS: field 0 is the
   docid, the last field is the url, the second-to-last is the title, and everything
   between is the body.

What gets indexed: `contents` = "<title> <body>" when --index_title is given (the intended
setting; see above), otherwise the body alone. `title` and `url` are also written as separate
JSON keys; Anserini's JsonCollection ignores unknown keys, so the url stays out of the index.

Usage
-----
    cd src
    python -m apcir.indexing.cast21_tsv_to_jsonl \
        --tar    ../data/collections/cast21_collection.tsv.tar.gz \
        --out    ../data/collections/cast21_jsonl \
        --shards 16
    # sanity-check the parser on the first N records without writing 40M docs:
    python -m apcir.indexing.cast21_tsv_to_jsonl --tar ... --out /tmp/probe --limit 2100000
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import sys

DOCID = re.compile(r"^(MARCO|WAPO|KILT)_\S*\t")
EXPECTED = 40_235_494


def parse_record(lines):
    """['docid\\tbody-head', 'body-tail\\ttitle\\turl'] -> (docid, body, title, url)"""
    rec = "\n".join(lines)
    f = rec.split("\t")
    if len(f) < 4:
        # no title/url present; treat everything after the docid as body
        return f[0], ("\t".join(f[1:]) if len(f) > 1 else ""), "", ""
    return f[0], "\t".join(f[1:-2]), f[-2], f[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="probe mode: stop after N COMPLETE records (never mid-record)")
    ap.add_argument("--index_title", action="store_true",
                    help="prepend the title to `contents` so Anserini searches it too. "
                         "Off by default: `contents` is then exactly the canonical passage "
                         "text the passage id denotes. Turn on if recall on WaPo/KILT is "
                         "poor -- headlines carry entity terms the body may omit.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    writers = [gzip.open(os.path.join(a.out, f"part-{i:03d}.jsonl.gz"), "wt",
                         encoding="utf-8", compresslevel=4)
               for i in range(a.shards)]

    proc = subprocess.Popen(["tar", "xzOf", a.tar], stdout=subprocess.PIPE, bufsize=1 << 22)

    n_docs = n_multiline = n_blank = 0
    buf = []
    index_title = a.index_title

    def flush():
        nonlocal n_docs, n_multiline
        if not buf:
            return
        if len(buf) > 1:
            n_multiline += 1
        did, body, title, url = parse_record(buf)
        contents = f"{title} {body}" if (index_title and title and title != ".") else body
        writers[n_docs % a.shards].write(json.dumps(
            {"id": did, "contents": contents, "title": title, "url": url},
            ensure_ascii=False) + "\n")
        n_docs += 1
        buf.clear()

    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if DOCID.match(line):
            # a new docid line means the buffered record is complete -> safe stop point
            if a.limit and n_docs >= a.limit:
                buf.clear()          # do NOT flush a record we may have truncated
                break
            flush()
            buf.append(line)
        elif not line.strip():
            n_blank += 1
        elif buf:
            buf.append(line)          # continuation of the current record's body
        else:
            n_blank += 1              # unattachable stray line (should not happen)
        if n_docs and n_docs % 5_000_000 == 0 and not buf:
            print(f"  {n_docs:,} docs written", flush=True)
    flush()

    for w in writers:
        w.close()
    proc.stdout.close()
    proc.terminate() if a.limit else proc.wait()

    print(f"wrote {n_docs:,} documents into {a.shards} shards -> {a.out}")
    print(f"  records reassembled from multiple physical lines: {n_multiline}")
    print(f"  blank/stray lines skipped: {n_blank}")
    if not a.limit and n_docs != EXPECTED:
        print(f"  WARNING: expected {EXPECTED:,} passages, got {n_docs:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
