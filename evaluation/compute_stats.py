"""
evaluation/compute_stats.py
Computes per-query Wilcoxon W, p-values, 95% bootstrap CIs, and Cliff's delta
for all 5 systems on the 132-query benchmark (K=5).

Run from legal_rag/:
    python -m evaluation.compute_stats
"""

import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy import stats as scipy_stats

# ── Load everything via the corrected load_all() ─────────────────────────────
from evaluation.metrics import load_all
from retrieval.baseline import vector_search
from retrieval.bm25_baseline import bm25_search
from retrieval.tfidf_baseline import tfidf_search
from retrieval.hybrid import hybrid_search
from retrieval.mmr import mmr_rerank

TOP_K   = 5
SEED    = 42
N_BOOT  = 1000
LAMBDA  = 0.9

# ── Metric helpers ────────────────────────────────────────────────────────────
def _section(r):
    return str(r.get("section_number", r.get("id", "")))

def recall_k(results, gts, k=TOP_K):
    gt = gts[0] if gts else ""
    return 1.0 if gt in [_section(r) for r in results[:k]] else 0.0

def mrr(results, gts, k=TOP_K):
    gt = gts[0] if gts else ""
    for i, r in enumerate(results[:k], 1):
        if _section(r) == gt:
            return 1.0 / i
    return 0.0

def ndcg_k(results, gts, k=TOP_K):
    gt = gts[0] if gts else ""
    for i, r in enumerate(results[:k], 1):
        if _section(r) == gt:
            return 1.0 / math.log2(i + 1)
    return 0.0

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci(scores, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

# ── Cliff's delta ─────────────────────────────────────────────────────────────
def cliffs_delta(a, b):
    a, b = np.array(a, float), np.array(b, float)
    n = len(a) * len(b)
    greater = float((a[:, None] > b[None, :]).sum())
    less    = float((a[:, None] < b[None, :]).sum())
    delta   = (greater - less) / n
    mag = ("negligible" if abs(delta) < 0.147 else
           "small"      if abs(delta) < 0.330 else
           "medium"     if abs(delta) < 0.474 else "large")
    return delta, mag

# ── Wilcoxon ──────────────────────────────────────────────────────────────────
def wilcoxon_safe(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if np.all(a - b == 0):
        return None, 1.0
    try:
        r = scipy_stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return None, 1.0

# ── Per-query evaluation ──────────────────────────────────────────────────────
def run_eval():
    print("Loading corpus, indices and queries …")
    clauses, faiss_index, embeddings, model, bm25, tfidf_vec, tfidf_mat = load_all()
    
    # load queries.json
    q_path = os.path.join(os.path.dirname(__file__), "..", "data", "queries.json")
    with open(q_path) as f:
        queries = json.load(f)
    
    print(f"Evaluating {len(queries)} queries across 5 systems …")
    
    SYSTEMS = ["BM25", "TF-IDF", "Vector", "Hybrid_noMMR", "Hybrid_MMR"]
    scores  = {s: {"recall": [], "mrr": [], "ndcg": []} for s in SYSTEMS}

    for i, q in enumerate(queries, 1):
        text = q.get("query", q.get("question", ""))
        gts = [str(x) for x in q.get("relevant_sections", [])]
        if not gts:
            gts = [str(q.get("ground_truth_section", q.get("ground_truth", "")))]
        
        def record(sys_key, results):
            scores[sys_key]["recall"].append(recall_k(results, gts))
            scores[sys_key]["mrr"].append(mrr(results, gts))
            scores[sys_key]["ndcg"].append(ndcg_k(results, gts))

        record("BM25", bm25_search(text, clauses, bm25, top_k=TOP_K))
        record("TF-IDF", tfidf_search(text, clauses, tfidf_vec, tfidf_mat, top_k=TOP_K))
        record("Vector", vector_search(text, clauses, faiss_index, model, top_k=TOP_K))
        record("Hybrid_noMMR", hybrid_search(
            text, clauses, faiss_index, embeddings, model, bm25,
            top_k=TOP_K, vector_weight=0.6, bm25_weight=0.4))
        cands = hybrid_search(
            text, clauses, faiss_index, embeddings, model, bm25,
            top_k=50, vector_weight=0.6, bm25_weight=0.4)
        reranked = mmr_rerank(cands, lam=LAMBDA, top_k=TOP_K)
        record("Hybrid_MMR", reranked)

        if i % 20 == 0:
            print(f"  Progress: {i}/{len(queries)}")

    return scores, len(queries)

# ── Tables ────────────────────────────────────────────────────────────────────
SYS_LABELS = {
    "BM25":         "BM25-Only",
    "TF-IDF":       "TF-IDF",
    "Vector":       "Vector-Only",
    "Hybrid_noMMR": "Hybrid (no MMR)",
    "Hybrid_MMR":   "Hybrid+MMR",
}

def print_ci_table(scores):
    print("\n### 95% Bootstrap Confidence Intervals (Recall@5, MRR, nDCG@5)\n")
    print("| System | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) |")
    print("|--------|-------------------|--------------|-----------------|")
    for sk, label in SYS_LABELS.items():
        d = scores[sk]
        if not d["recall"]: continue
        rm, rl, rh = bootstrap_ci(d["recall"])
        mm, ml, mh = bootstrap_ci(d["mrr"])
        nm, nl, nh = bootstrap_ci(d["ndcg"])
        print(f"| {label} | {rm:.4f} [{rl:.4f}-{rh:.4f}] "
              f"| {mm:.4f} [{ml:.4f}-{mh:.4f}] "
              f"| {nm:.4f} [{nl:.4f}-{nh:.4f}] |")

def print_wilcoxon_table(scores):
    print("\n### Wilcoxon Signed-Rank Tests + Cliff's Delta\n")
    print("| System A | System B | Metric | W | p-value | Cliff's delta | Magnitude | Sig? |")
    print("|----------|----------|--------|---|---------|---------------|-----------|------|")
    comps = [
        ("Vector", "BM25",         "Recall@5"),
        ("Vector", "Hybrid_MMR",   "Recall@5"),
        ("Vector", "Hybrid_noMMR", "Recall@5"),
        ("BM25",   "Hybrid_MMR",   "Recall@5"),
        ("Hybrid_noMMR","Hybrid_MMR","Recall@5"),
        ("Vector", "BM25",         "MRR"),
        ("Vector", "Hybrid_MMR",   "MRR"),
    ]
    metric_key = {"Recall@5": "recall", "MRR": "mrr"}
    for a, b, met in comps:
        key = metric_key[met]
        av, bv = scores[a][key], scores[b][key]
        if len(av) != len(bv): continue
        W, p = wilcoxon_safe(av, bv)
        delta, mag = cliffs_delta(av, bv)
        sig = "Y p<0.05" if (p is not None and p < 0.05) else "N"
        w_str = f"{W:.1f}" if W is not None else "N/A"
        p_fmt = f"{p:.4f}" if (p is not None and p >= 0.001) else f"{p:.2e}" if p else "N/A"
        print(f"| {SYS_LABELS[a]} | {SYS_LABELS[b]} | {met} "
              f"| {w_str} | {p_fmt} | {delta:+.3f} | {mag} | {sig} |")

def save_results(scores, n_queries):
    out = {"n_queries": n_queries, "systems": {}}
    for sk, label in SYS_LABELS.items():
        d = scores[sk]
        if not d["recall"]: continue
        rm, rl, rh = bootstrap_ci(d["recall"])
        mm, ml, mh = bootstrap_ci(d["mrr"])
        nm, nl, nh = bootstrap_ci(d["ndcg"])
        out["systems"][label] = {
            "recall_mean": rm, "recall_ci_lo": rl, "recall_ci_hi": rh,
            "mrr_mean":    mm, "mrr_ci_lo":   ml, "mrr_ci_hi":   mh,
            "ndcg_mean":   nm, "ndcg_ci_lo":  nl, "ndcg_ci_hi":  nh,
            "per_query_recall": d["recall"],
            "per_query_mrr":    d["mrr"],
            "per_query_ndcg":   d["ndcg"],
        }
    path = os.path.join(os.path.dirname(__file__), "..", "data", "charts", "per_query_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Y] Saved to {path}")
    return out

if __name__ == "__main__":
    scores, n = run_eval()
    print(f"\n[Y] Done: {n} queries × 5 systems")
    print_ci_table(scores)
    print_wilcoxon_table(scores)
    save_results(scores, n)
