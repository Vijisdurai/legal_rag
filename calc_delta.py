import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from evaluation.compute_stats import cliffs_delta

try:
    with open(os.path.join(os.path.dirname(__file__), "data/eval_results.json"), "r") as f:
        data = json.load(f)
    
    qr = data["results"]["Vector-Only"]["metrics"]["Recall@5_list"]
    no_qr = data["results"]["Vector-Only"]["original_metrics"]["Recall@5_list"]
    
    delta, mag = cliffs_delta(qr, no_qr)
    print(f"Effect Size (Cliff's Delta) for Query Rewriting: {delta:+.3f} ({mag})")
except Exception as e:
    print(f"Error: {e}")
