import json
from evaluation.compute_stats import wilcoxon_safe, cliffs_delta

with open("data/charts/per_query_stats.json") as f:
    d = json.load(f)

comp_keys = {
    "BM25-Only": "BM25-Only",
    "Vector-Only": "Vector-Only",
    "Hybrid (no MMR)": "Hybrid (no MMR)",
    "Hybrid+MMR": "Hybrid+MMR"
}

with open("tables.md", "a", encoding="utf-8") as out:
    out.write("\n### Wilcoxon Signed-Rank Tests + Cliff's Delta\n\n")
    out.write("| System A | System B | Metric | W | p-value | Cliff's delta | Magnitude | Sig? |\n")
    out.write("|----------|----------|--------|---|---------|---------------|-----------|------|\n")
    
    comps = [
        ("Vector-Only", "BM25-Only",         "per_query_recall"),
        ("Vector-Only", "Hybrid+MMR",   "per_query_recall"),
        ("Vector-Only", "Hybrid (no MMR)", "per_query_recall"),
        ("BM25-Only",   "Hybrid+MMR",   "per_query_recall"),
        ("Hybrid (no MMR)","Hybrid+MMR","per_query_recall"),
        ("Vector-Only", "BM25-Only",         "per_query_mrr"),
        ("Vector-Only", "Hybrid+MMR",   "per_query_mrr"),
    ]
    
    for a, b, key in comps:
        av = d["systems"][a][key]
        bv = d["systems"][b][key]
        met = "Recall@5" if "recall" in key else "MRR"
        W, p = wilcoxon_safe(av, bv)
        delta, mag = cliffs_delta(av, bv)
        sig = "Y p<0.05" if (p is not None and p < 0.05) else "N"
        w_str = f"{W:.1f}" if W is not None else "N/A"
        p_fmt = f"{p:.4f}" if (p is not None and p >= 0.001) else f"{p:.2e}" if p else "N/A"
        out.write(f"| {a} | {b} | {met} | {w_str} | {p_fmt} | {delta:+.3f} | {mag} | {sig} |\n")

print("Done part 2")
