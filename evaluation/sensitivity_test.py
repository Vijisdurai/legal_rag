"""
evaluation/sensitivity_test.py
──────────────────────────────
Multi-factor sensitivity analysis using real queries and metrics.

Tests:
  1. Top-K sensitivity:      K = 3, 5, 10
  2. MMR Lambda sensitivity: λ = 0.7, 0.8, 0.9, 1.0
  3. Per-category breakdown: Exact / Paraphrase / Conceptual

Outputs console tables, charts, and sensitivity_results.json
"""

import os, sys, json, time, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

from evaluation.metrics import (
    recall_at_k, precision_at_k, ndcg_at_k, mrr as mrr_fn, ap_at_k,
    hit_at_k, wilcoxon_test, bootstrap_confidence_interval, load_all
)
from tabulate import tabulate

QUERIES_PATH = os.path.join(BASE_DIR, 'data', 'queries.json')
CHARTS_DIR   = os.path.join(BASE_DIR, 'data', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

SYSTEMS_5 = ["BM25-Only", "TF-IDF", "Vector-Only", "Hybrid (no MMR)", "Hybrid + MMR"]


def safe_retrieve(system, query, components, k=5, lam=0.9):
    """Safe retrieval that handles index mismatches gracefully."""
    clauses, index, embeddings, model, bm25, vectorizer, tfidf_matrix = components
    n_clauses = len(clauses)

    try:
        t0 = time.time()
        if system == "BM25-Only":
            tokens = query.lower().split()
            scores = np.array(bm25.get_scores(tokens), dtype='float32')
            top_indices = np.argsort(scores)[::-1]
            results = []
            rank = 1
            for idx in top_indices:
                if idx >= n_clauses:
                    continue
                c = clauses[idx]
                results.append({"section_number": c["section_number"]})
                rank += 1
                if rank > k:
                    break
        elif system == "TF-IDF":
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1]
            results = []
            rank = 1
            for idx in top_indices:
                if idx >= n_clauses:
                    continue
                c = clauses[idx]
                results.append({"section_number": c["section_number"]})
                rank += 1
                if rank > k:
                    break
        elif system == "Vector-Only":
            from retrieval.baseline import vector_search
            results = vector_search(query, clauses, index, model, top_k=k)
        elif system == "Hybrid (no MMR)":
            from retrieval.hybrid import hybrid_search
            results = hybrid_search(query, clauses, index, embeddings, model, bm25, top_k=k)
        elif system == "Hybrid + MMR":
            from retrieval.hybrid import hybrid_search
            from retrieval.mmr import mmr_rerank
            candidates = hybrid_search(query, clauses, index, embeddings, model, bm25, top_k=50)
            results = mmr_rerank(candidates, lam=lam, top_k=k)
        else:
            results = []

        elapsed = (time.time() - t0) * 1000
        sections = [r["section_number"] for r in results]
        return sections, elapsed
    except Exception as e:
        return [], 0.0


def compute_metrics(sections, relevant, k=5):
    return {
        "r_k":    recall_at_k(sections, relevant, k),
        "p_k":    precision_at_k(sections, relevant, k),
        "ndcg_k": ndcg_at_k(sections, relevant, k),
        "mrr":    mrr_fn(sections, relevant),
        "hit":    hit_at_k(sections, relevant, k),
    }


# ── 1. TOP-K SENSITIVITY ─────────────────────────────────────────────────
def test_topk_sensitivity(queries, components):
    print("\n" + "=" * 80)
    print("TEST 1: TOP-K SENSITIVITY (K = 3, 5, 10)")
    print("=" * 80)

    ks = [3, 5, 10]
    results = {}
    for k in ks:
        results[k] = {}
        for s in SYSTEMS_5:
            r_vals, mrr_vals, ndcg_vals = [], [], []
            for q in queries:
                secs, _ = safe_retrieve(s, q["query"], components, k=k)
                m = compute_metrics(secs, q["relevant_sections"], k)
                r_vals.append(m["r_k"])
                mrr_vals.append(m["mrr"])
                ndcg_vals.append(m["ndcg_k"])
            results[k][s] = {
                "recall": float(np.mean(r_vals)), "mrr": float(np.mean(mrr_vals)),
                "ndcg": float(np.mean(ndcg_vals)),
                "recall_ci": [float(x) for x in bootstrap_confidence_interval(r_vals)],
            }
            sys.stdout.write(f"\r  K={k}, {s:25s} ✓")
            sys.stdout.flush()
        print(f"\n  K={k} complete")

    # Print table
    headers = ["System", "K=3 R@K", "K=3 MRR", "K=5 R@K", "K=5 MRR", "K=10 R@K", "K=10 MRR"]
    rows = []
    for s in SYSTEMS_5:
        row = [s]
        for k in ks:
            r = results[k][s]
            row.extend([f"{r['recall']:.4f}", f"{r['mrr']:.4f}"])
        rows.append(row)
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # Key findings
    print("\n  KEY FINDINGS:")
    for s in SYSTEMS_5:
        r3 = results[3][s]["recall"]
        r10 = results[10][s]["recall"]
        m3 = results[3][s]["mrr"]
        m10 = results[10][s]["mrr"]
        gain = ((r10 - r3) / r3 * 100) if r3 > 0 else 0
        mdrop = ((m10 - m3) / m3 * 100) if m3 > 0 else 0
        print(f"    {s:25s}: Recall K=3→10: {gain:+.1f}%  |  MRR K=3→10: {mdrop:+.1f}%")

    # Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
    colors = {"BM25-Only":"#9CB3C9","TF-IDF":"#B5C8A8","Vector-Only":"#4299E1",
              "Hybrid (no MMR)":"#FB7185","Hybrid + MMR":"#1B2A4A"}
    for s in SYSTEMS_5:
        r_vals = [results[k][s]["recall"] for k in ks]
        m_vals = [results[k][s]["mrr"] for k in ks]
        lw = 2.5 if s == "Vector-Only" else 1.5
        ax1.plot(ks, r_vals, 'o-', label=s, color=colors[s], lw=lw, ms=5)
        ax2.plot(ks, m_vals, 's--', label=s, color=colors[s], lw=lw, ms=5)
    ax1.set_xlabel("K"); ax1.set_ylabel("Recall@K"); ax1.set_title("Recall@K by System")
    ax2.set_xlabel("K"); ax2.set_ylabel("MRR"); ax2.set_title("MRR by K")
    ax1.legend(fontsize=7); ax2.legend(fontsize=7)
    ax1.set_xticks(ks); ax2.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "topk_sensitivity.png"), dpi=150)
    plt.close()
    print(f"  [chart] Saved topk_sensitivity.png")
    return results


# ── 2. MMR LAMBDA SENSITIVITY ──────────────────────────────────────────────
def test_lambda_sensitivity(queries, components):
    print("\n" + "=" * 80)
    print("TEST 2: MMR LAMBDA SENSITIVITY (λ = 0.7, 0.8, 0.9, 1.0)")
    print("=" * 80)

    lambdas = [0.7, 0.8, 0.9, 1.0]
    results = {}
    for lam_val in lambdas:
        r_vals, mrr_vals, ndcg_vals = [], [], []
        for q in queries:
            secs, _ = safe_retrieve("Hybrid + MMR", q["query"], components, k=5, lam=lam_val)
            m = compute_metrics(secs, q["relevant_sections"])
            r_vals.append(m["r_k"])
            mrr_vals.append(m["mrr"])
            ndcg_vals.append(m["ndcg_k"])
        results[lam_val] = {
            "recall": float(np.mean(r_vals)), "mrr": float(np.mean(mrr_vals)),
            "ndcg": float(np.mean(ndcg_vals)),
            "recall_scores": r_vals.copy(),
            "recall_ci": [float(x) for x in bootstrap_confidence_interval(r_vals)],
        }
        print(f"  λ={lam_val} done ✓ (R@5={np.mean(r_vals):.4f})")

    # Print table
    headers = ["Lambda", "Recall@5", "95% CI", "MRR", "nDCG@5"]
    rows = []
    for lam_val in lambdas:
        r = results[lam_val]
        ci = r["recall_ci"]
        rows.append([f"λ={lam_val}", f"{r['recall']:.4f}", f"[{ci[0]:.3f}–{ci[1]:.3f}]",
                      f"{r['mrr']:.4f}", f"{r['ndcg']:.4f}"])
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # Wilcoxon
    if 0.9 in results and 0.7 in results:
        W, p, sig = wilcoxon_test(results[0.9]["recall_scores"], results[0.7]["recall_scores"])
        print(f"\n  Wilcoxon λ=0.9 vs λ=0.7: W={W:.1f}, p={p:.4f}, {'*** SIGNIFICANT ***' if sig else 'not significant'}")

    # λ=1.0 vs λ=0.9 — critical: λ=1.0 disables MMR
    if 1.0 in results and 0.9 in results:
        W2, p2, sig2 = wilcoxon_test(results[1.0]["recall_scores"], results[0.9]["recall_scores"])
        print(f"  Wilcoxon λ=1.0 vs λ=0.9: W={W2:.1f}, p={p2:.4f}, {'*** SIGNIFICANT ***' if sig2 else 'not significant'}")
        print(f"    → λ=1.0 effectively disables diversification (pure relevance)")

    best_lam = max(results, key=lambda l: results[l]["recall"])
    worst_lam = min(results, key=lambda l: results[l]["recall"])
    print(f"\n  Best λ: {best_lam} (R@5={results[best_lam]['recall']:.4f})")
    print(f"  Worst λ: {worst_lam} (R@5={results[worst_lam]['recall']:.4f})")
    if results[best_lam]["recall"] > 0:
        delta = ((results[worst_lam]["recall"] - results[best_lam]["recall"]) / results[best_lam]["recall"]) * 100
        print(f"  Max sensitivity range: {delta:+.1f}%")

    # Chart
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(lambdas, [results[l]["recall"] for l in lambdas], 'o-', label='Recall@5', color='#4299E1', lw=2.2, ms=7)
    ax.plot(lambdas, [results[l]["mrr"] for l in lambdas], 's--', label='MRR', color='#1B2A4A', lw=2.2, ms=7)
    ax.plot(lambdas, [results[l]["ndcg"] for l in lambdas], '^:', label='nDCG@5', color='#9CB3C9', lw=1.8, ms=6)
    ax.set_xlabel("MMR Lambda (λ)"); ax.set_ylabel("Score")
    ax.set_title("Hybrid+MMR: Sensitivity to Lambda"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "lambda_sensitivity.png"), dpi=150)
    plt.close()
    print(f"  [chart] Saved lambda_sensitivity.png")
    return results


# ── 3. PER-CATEGORY BREAKDOWN ──────────────────────────────────────────────
def test_category_breakdown(queries, components):
    print("\n" + "=" * 80)
    print("TEST 3: PER-CATEGORY PERFORMANCE (Exact / Paraphrase / Conceptual)")
    print("=" * 80)

    cats = ["exact", "paraphrase", "conceptual"]
    results = {s: {c: {"r5": [], "mrr": []} for c in cats} for s in SYSTEMS_5}

    for i, q in enumerate(queries, 1):
        cat = q.get("category", "exact")
        for s in SYSTEMS_5:
            secs, _ = safe_retrieve(s, q["query"], components, k=5)
            m = compute_metrics(secs, q["relevant_sections"])
            if cat in results[s]:
                results[s][cat]["r5"].append(m["r_k"])
                results[s][cat]["mrr"].append(m["mrr"])
        if i % 30 == 0:
            sys.stdout.write(f"\r  [{i}/{len(queries)}] queries processed")
            sys.stdout.flush()

    headers = ["System", "Exact R@5", "Para R@5", "Conc R@5", "Exact MRR", "Para MRR", "Conc MRR"]
    rows = []
    for s in SYSTEMS_5:
        row = [s]
        for c in cats:
            vals = results[s][c]["r5"]
            row.append(f"{np.mean(vals):.4f}" if vals else "N/A")
        for c in cats:
            vals = results[s][c]["mrr"]
            row.append(f"{np.mean(vals):.4f}" if vals else "N/A")
        rows.append(row)
    print("\n\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # Key findings
    print("\n  BEST SYSTEM PER CATEGORY (Recall@5):")
    for c in cats:
        best = max(SYSTEMS_5, key=lambda s: np.mean(results[s][c]["r5"]) if results[s][c]["r5"] else 0)
        val = np.mean(results[best][c]["r5"]) if results[best][c]["r5"] else 0
        print(f"    {c.capitalize():15s}: {best:25s} R@5 = {val:.4f}")

    # Sensitivity: which system drops most from Exact to Conceptual
    print("\n  EXACT → CONCEPTUAL DROP (% decline in Recall@5):")
    for s in SYSTEMS_5:
        e = np.mean(results[s]["exact"]["r5"]) if results[s]["exact"]["r5"] else 0
        c_val = np.mean(results[s]["conceptual"]["r5"]) if results[s]["conceptual"]["r5"] else 0
        drop = ((c_val - e) / e * 100) if e > 0 else 0
        print(f"    {s:25s}: Exact={e:.4f} → Conc={c_val:.4f} ({drop:+.1f}%)")

    # Chart: grouped bars per category
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
    colors = {"BM25-Only":"#9CB3C9","TF-IDF":"#B5C8A8","Vector-Only":"#4299E1",
              "Hybrid (no MMR)":"#FB7185","Hybrid + MMR":"#1B2A4A"}
    for ax, metric_key, title in [(axes[0], "r5", "Recall@5 by Category"),
                                   (axes[1], "mrr", "MRR by Category")]:
        x = np.arange(len(cats))
        w = 0.15
        for si, s in enumerate(SYSTEMS_5):
            vals = [np.mean(results[s][c][metric_key]) if results[s][c][metric_key] else 0 for c in cats]
            off = (si - len(SYSTEMS_5)/2) * w + w/2
            ax.bar(x + off, vals, w, label=s if ax == axes[0] else "", color=colors[s], edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cats], fontsize=9)
        ax.set_title(title)
    axes[0].legend(fontsize=6, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "category_sensitivity.png"), dpi=150)
    plt.close()
    print(f"\n  [chart] Saved category_sensitivity.png")

    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(QUERIES_PATH, encoding='utf-8') as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries")
    components = load_all()

    topk_res   = test_topk_sensitivity(queries, components)
    lambda_res = test_lambda_sensitivity(queries, components)
    cat_res    = test_category_breakdown(queries, components)

    # Save all results to JSON
    output = {
        "topk": {str(k): {s: {"recall": v["recall"], "mrr": v["mrr"], "ndcg": v["ndcg"]}
                 for s, v in sv.items()} for k, sv in topk_res.items()},
        "lambda": {str(l): {"recall": v["recall"], "mrr": v["mrr"], "ndcg": v["ndcg"]}
                   for l, v in lambda_res.items()},
        "category": {s: {c: {"r5": float(np.mean(v["r5"])) if v["r5"] else 0,
                              "mrr": float(np.mean(v["mrr"])) if v["mrr"] else 0}
                         for c, v in cv.items()} for s, cv in cat_res.items()},
    }
    out_path = os.path.join(CHARTS_DIR, "sensitivity_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[saved] All results -> {out_path}")
    print("\n✅ Sensitivity analysis complete.")
