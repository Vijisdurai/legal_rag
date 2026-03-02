import json

with open("data/charts/per_query_stats.json") as f:
    d = json.load(f)

with open("tables.md", "w", encoding="utf-8") as out:
    out.write("### 95% Bootstrap Confidence Intervals (Recall@5, MRR, nDCG@5)\n\n")
    out.write("| System | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) |\n")
    out.write("|--------|-------------------|--------------|-----------------|\n")
    for sys_name, stats in d["systems"].items():
        out.write(f"| {sys_name} | {stats['recall_mean']:.4f} [{stats['recall_ci_lo']:.4f}-{stats['recall_ci_hi']:.4f}] | {stats['mrr_mean']:.4f} [{stats['mrr_ci_lo']:.4f}-{stats['mrr_ci_hi']:.4f}] | {stats['ndcg_mean']:.4f} [{stats['ndcg_ci_lo']:.4f}-{stats['ndcg_ci_hi']:.4f}] |\n")

print("Done")
