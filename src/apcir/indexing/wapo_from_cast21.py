"""Reconstruct Washington Post V4 DOCUMENTS from the CAsT-2021 passage collection.

Why this exists. CAsT 2022 uses "the same KILT and Washington Post V4 collections from year 3"
(2022 overview), and WaPo is the only sub-corpus behind a signed NIST agreement. We do have it,
but only in the CAsT-2021 release form: already split into passages
(`WAPO_<id>_<n> \t body \t title \t url`). The official CAsT-2022 chunker needs whole documents,
so we glue the passages of each document back together in passage order.

What this produces: one JSON object per line, mimicking the fields the official
`WaPoGenerator` reads out of `TREC_Washington_Post_collection.v4.jl`, so the SAME
`PassageChunker` (spaCy 3.3.0 / en_core_web_sm-3.3.0, <=250-word passages) can be applied and
the resulting passage ids are produced by official code rather than by us.

Fidelity caveat, stated up front. The 2021 passages are themselves the output of a chunker, so
re-joining them recovers the document text only up to whatever the 2021 chunking normalised
(whitespace at passage boundaries, in particular). Measured on the CAsT-2021 collection, 1.05%
of its passages exceed the official 250-word limit and its passage numbering starts at 1 while
official CAsT-2021 run files start at 0 -- i.e. the tarball we have is a third-party
re-serialisation, not the byte-exact official release. Expect near-identical but not guaranteed
byte-identical passages. Verify by checking that the passage ids referenced by the CAsT-2022
qrels exist in the rebuilt collection.

Usage
-----
    cd src
    python -m apcir.indexing.wapo_from_cast21 \
        --tar ../data/collections/cast21_collection.tsv.tar.gz \
        --out /part/01/Tmp/yuchen/cast22_build/wapo_docs.jl
"""
import argparse
import json
import re
import subprocess
import sys

DOCID = re.compile(r"^(MARCO|WAPO|KILT)_\S*\t")


def parse_record(lines):
    rec = "\n".join(lines)
    f = rec.split("\t")
    if len(f) < 4:
        return f[0], ("\t".join(f[1:]) if len(f) > 1 else ""), "", ""
    return f[0], "\t".join(f[1:-2]), f[-2], f[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    proc = subprocess.Popen(["tar", "xzOf", a.tar], stdout=subprocess.PIPE, bufsize=1 << 22)
    out = open(a.out, "w", encoding="utf-8")

    cur_doc = None            # (docid, title, url)
    parts = []                # [(passage_no, body)]
    n_docs = n_pas = 0
    buf = []

    def flush_doc():
        nonlocal cur_doc, parts, n_docs
        if cur_doc is None or not parts:
            return
        did, title, url = cur_doc
        parts.sort(key=lambda x: x[0])
        body = " ".join(b.strip() for _, b in parts if b.strip())
        # field names mirror TREC_Washington_Post_collection.v4.jl as read by WaPoGenerator
        out.write(json.dumps({
            "id": did[len("WAPO_"):],
            "title": title,
            "article_url": url,
            # WaPoGenerator keeps ONLY items whose subtype == "paragraph" (it also strips
            # HTML and drops documents whose body ends up empty), so the subtype must be set
            # or every document is silently skipped.
            "contents": [{"type": "sanitized_html", "subtype": "paragraph", "content": body}],
        }, ensure_ascii=False) + "\n")
        n_docs += 1
        cur_doc, parts = None, []

    def handle(rec_lines):
        nonlocal cur_doc, parts, n_pas
        did, body, title, url = parse_record(rec_lines)
        if not did.startswith("WAPO_"):
            return
        base, _, num = did.rpartition("_")
        try:
            num = int(num)
        except ValueError:
            base, num = did, 0
        if cur_doc is not None and cur_doc[0] != base:
            flush_doc()
        if cur_doc is None:
            cur_doc = (base, title, url)
        parts.append((num, body))
        n_pas += 1

    for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if DOCID.match(line):
            if buf:
                handle(buf)
            buf = [line]
        elif line.strip() and buf:
            buf.append(line)
    if buf:
        handle(buf)
    flush_doc()

    out.close()
    proc.stdout.close()
    proc.wait()
    print(f"wrote {n_docs:,} WaPo documents (from {n_pas:,} passages) -> {a.out}")
    if n_docs != 724509:
        print(f"  NOTE: the 2021 collection holds 724,509 WaPo documents; got {n_docs:,}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
