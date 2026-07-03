"""Render bench_dense results.jsonl into markdown trade-off tables.

  python -m apcir.benchmarks.report_table /part/01/Tmp/yuchenhui/bench_dense/results.jsonl
"""
import json
import sys
from collections import defaultdict

FIDELITY = {  # "identical to the paper eval?"
    "a_stream_fp16_torch": "near (fp16)", "b_stream_faiss_fp32": "bit-exact (GT)",
    "c_resident_faiss_fp16": "near (fp16)", "d_resident_torch_fp16": "near (fp16)",
    "e_gpu_ivf_sqfp16": "approximate", "f_gpu_ivf_sq8": "approximate",
    "f_ivf_pq96": "approximate", "f_ivf_pq64": "approximate", "g_cpu_hnsw32": "approximate",
}


def main(path: str):
    rows = [json.loads(l) for l in open(path)]
    recall_fill = {}
    for r in rows:
        if r.get("config", "").endswith("_recall_backfill"):
            recall_fill[r["config"][0]] = (r.get("recall10"), r.get("recall100"))
    by_unit = defaultdict(list)
    for r in rows:
        c = r.get("config", "")
        if c.startswith("_") or c.endswith("_recall_backfill") or r.get("batch") is None:
            continue
        if c[0] in recall_fill and r.get("recall10") is None:
            r["recall10"], r["recall100"] = recall_fill[c[0]]
        by_unit[r["unit"]].append(r)
    for unit, rs in by_unit.items():
        print(f"\n## {unit}\n")
        print("| config | param | batch | build/load s | RAM ΔG | VRAM G | p50 s | p95 s | QPS | R@10 | R@100 | vs paper |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in sorted(rs, key=lambda x: (x["config"], str(x.get("param")), x.get("batch", 0))):
            print(f"| {r['config']} | {r.get('param','')} | {r.get('batch','')} | "
                  f"{r.get('build_s','')} | {r.get('rss_extra_gb','')} | {r.get('vram_gb','')} | "
                  f"{r.get('p50_s','')} | {r.get('p95_s','')} | {r.get('qps','')} | "
                  f"{r.get('recall10','')} | {r.get('recall100','')} | "
                  f"{FIDELITY.get(r['config'],'')} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/part/01/Tmp/yuchenhui/bench_dense/results.jsonl")
