import json, sys
from evaluation.compute_stats import print_ci_table, print_wilcoxon_table

with open("data/charts/per_query_stats.json") as f:
    d = json.load(f)

# redirect stdout to tables.md
import builtins
with open("tables.md", "w", encoding="utf-8") as out:
    def safe_print(*args, **kwargs):
        out.write(" ".join(map(str, args)) + "\n")
    builtins.print = safe_print
    
    print_ci_table(d["systems"])
    print_wilcoxon_table(d["systems"])
