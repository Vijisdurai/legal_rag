"""
evaluation/metrics.py
---------------------
Full ablation evaluation: 6 retrieval systems × 132 queries.

Metrics:
  - Precision@5, Recall@5 (Hit Rate@5)
  - NDCG@5  (primary metric — graded relevance)
  - MAP@10
  - MRR
  - Wilcoxon signed-rank test (all pairwise system comparisons)
  - 95% bootstrap confidence intervals on Recall@5
  - Query Rewriting impact table (Recall@5 with/without LLM rewriting)

Systems evaluated:
  1. BM25-Only
  2. TF-IDF
  3. Vector-Only (FAISS)
  4. Hybrid (no MMR)          ← ablation: fusion only, no diversity reranking
  5. Hybrid + MMR
  6. Hybrid + CrossEncoder + MMR  (optional - requires Ollama)

Charts saved to data/charts/:
  1. system_comparison.png     — grouped bar (all systems × all metrics)
  2. category_breakdown.png    — stacked bar (hit rate by query category)
  3. score_distribution.png    — histogram (score distributions)
  4. latency_comparison.png    — bar chart (avg ms per system)
  5. chapter_coverage.png      — pie chart (covered IPC chapters)
  6. score_correlation.png     — scatter (BM25 vs vector scores)
  7. rewriting_impact.png      — bar chart (Recall@5 delta from query rewriting)
"""

import os
import sys
import json
import time
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)
warnings.filterwarnings('ignore')

QUERIES_PATH  = os.path.join(BASE_DIR, 'data', 'queries.json')
CHARTS_DIR    = os.path.join(BASE_DIR, 'data', 'charts')
TOP_K = 5

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOURS = {
    "BM25-Only":              "#F97316",
    "TF-IDF":                 "#FACC15",
    "Vector-Only":            "#60A5FA",
    "Hybrid (no MMR)":        "#FB7185",   # ablation: fusion only
    "Hybrid + MMR":           "#A78BFA",
    "Hybrid + CE + MMR":      "#34D399",
}
SYSTEMS = list(COLOURS.keys())

os.makedirs(CHARTS_DIR, exist_ok=True)


# ── Metric functions ───────────────────────────────────────────────────────────

def hit_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> int:
    return int(any(s in relevant for s in retrieved[:k]))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    hits = sum(1 for s in retrieved[:k] if s in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for s in retrieved[:k] if s in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    for i, s in enumerate(retrieved, 1):
        if s in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Graded NDCG: rel=1 if in relevant set."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, s in enumerate(retrieved[:k])
        if s in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """Average Precision@k."""
    hits, sum_prec = 0, 0.0
    for i, s in enumerate(retrieved[:k], 1):
        if s in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(relevant), k) if relevant else 0.0


def mcnemar_test(hits_a: list[int], hits_b: list[int]) -> tuple[float, bool]:
    """
    McNemar's test for paired binary outcomes.
    Returns (p_value, is_significant) where significant = p < 0.05.
    """
    b = sum(1 for a, bv in zip(hits_a, hits_b) if a == 1 and bv == 0)
    c = sum(1 for a, bv in zip(hits_a, hits_b) if a == 0 and bv == 1)
    n = b + c
    if n == 0:
        return 1.0, False
    # Chi-square with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # p-value approximation (chi-sq df=1)
    from math import erfc, sqrt
    p = erfc(sqrt(chi2 / 2))
    return p, p < 0.05


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> tuple[float, float, bool]:
    """
    Wilcoxon signed-rank test for paired continuous metric scores.
    More appropriate than McNemar for non-binary metrics (MRR, nDCG).
    Returns (W_statistic, p_value, is_significant) where significant = p < 0.05.
    """
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        non_zero = [d for d in diffs if abs(d) > 1e-10]
        if len(non_zero) < 10:
            return 0.0, 1.0, False
        W, p = scipy_wilcoxon(non_zero, alternative='two-sided')
        return float(W), float(p), p < 0.05
    except ImportError:
        # scipy not available — fall back to McNemar on binarised scores
        hits_a = [int(s > 0) for s in scores_a]
        hits_b = [int(s > 0) for s in scores_b]
        p, sig = mcnemar_test(hits_a, hits_b)
        return 0.0, p, sig


def bootstrap_confidence_interval(scores: list[float], n_boot: int = 1000,
                                   ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """
    Bootstrap 95% CI for the mean of a metric score list.
    Returns (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))


# ── Load components (cached at module level) ───────────────────────────────────

def load_all():
    from retrieval.dual_corpus import load_combined_corpus
    from indexing.vector_index import get_or_build_index
    from indexing.bm25_index import get_or_build_bm25
    from retrieval.tfidf_baseline import get_or_build_tfidf

    # Use the combined corpus (IPC + BNS) so that the clause list
    # matches the indices built by rebuild_indices.py (633 entries).
    clauses = load_combined_corpus()
    index, embeddings, model = get_or_build_index(clauses)
    bm25, _ = get_or_build_bm25(clauses)
    vectorizer, tfidf_matrix = get_or_build_tfidf(clauses)
    return clauses, index, embeddings, model, bm25, vectorizer, tfidf_matrix



# ── Run retrieval per system ────────────────────────────────────────────────────

def retrieve(system: str, query: str, components: tuple, k: int = TOP_K) -> tuple[list[str], float]:
    """Returns (section_number_list, elapsed_ms)."""
    clauses, index, embeddings, model, bm25, vectorizer, tfidf_matrix = components

    t0 = time.time()
    if system == "BM25-Only":
        from retrieval.bm25_baseline import bm25_search
        results = bm25_search(query, clauses, bm25, top_k=k)
    elif system == "TF-IDF":
        from retrieval.tfidf_baseline import tfidf_search
        results = tfidf_search(query, clauses, vectorizer, tfidf_matrix, top_k=k)
    elif system == "Vector-Only":
        from retrieval.baseline import vector_search
        results = vector_search(query, clauses, index, model, top_k=k)
    elif system == "Hybrid (no MMR)":
        # Ablation: hybrid fusion score only, no MMR diversity reranking.
        # Isolates whether MMR (not fusion) is responsible for the performance drop.
        from retrieval.hybrid import hybrid_search
        results = hybrid_search(query, clauses, index, embeddings, model, bm25, top_k=k)
    elif system == "Hybrid + MMR":
        from retrieval.hybrid import hybrid_search
        from retrieval.mmr import mmr_rerank
        candidates = hybrid_search(query, clauses, index, embeddings, model, bm25, top_k=50)
        results = mmr_rerank(candidates, top_k=k)
    elif system == "Hybrid + CE + MMR":
        try:
            from retrieval.cross_encoder_rerank import hybrid_cross_mmr_pipeline
            results = hybrid_cross_mmr_pipeline(query, clauses, index, embeddings, model, bm25,
                                                 final_top_k=k)
        except Exception:
            results = []
    else:
        results = []

    elapsed = (time.time() - t0) * 1000
    sections = [r["section_number"] for r in results]
    return sections, elapsed


# ── Main evaluation loop ────────────────────────────────────────────────────────

def run_evaluation(skip_cross_encoder: bool = True):
    """
    Evaluate all systems on queries.json.
    Returns aggregated results dict.
    """
    with open(QUERIES_PATH, encoding='utf-8') as f:
        queries = json.load(f)

    print(f"\nLoaded {len(queries)} evaluation queries.")

    components = load_all()

    # Results storage per system
    results = {s: {
        "p5": [], "r5": [], "ndcg5": [], "ap10": [], "mrr": [],
        "hits": [], "latency": [], "cat_hits": {"exact": [], "paraphrase": [], "conceptual": []}
    } for s in SYSTEMS}

    # SYSTEMS[:5] = BM25, TF-IDF, Vector, Hybrid(no MMR), Hybrid+MMR (skip CE+MMR by default)
    active_systems = SYSTEMS[:5] if skip_cross_encoder else SYSTEMS

    print(f"\nRunning ablation on {len(active_systems)} systems × {len(queries)} queries...\n")

    for i, q in enumerate(queries, 1):
        query   = q["query"]
        rel     = q["relevant_sections"]
        cat     = q.get("category", "exact")

        sys.stdout.write(f"\r  [{i:3}/{len(queries)}] {query[:45]:<45}")
        sys.stdout.flush()

        for sys_name in active_systems:
            sections, ms = retrieve(sys_name, query, components)
            r = results[sys_name]
            r["p5"].append(precision_at_k(sections, rel))
            r["r5"].append(recall_at_k(sections, rel))
            r["ndcg5"].append(ndcg_at_k(sections, rel))
            r["ap10"].append(ap_at_k(sections, rel, k=10))
            r["mrr"].append(mrr(sections, rel))
            r["hits"].append(hit_at_k(sections, rel))
            r["latency"].append(ms)
            if cat in r["cat_hits"]:
                r["cat_hits"][cat].append(hit_at_k(sections, rel))

    print("\n")
    return results, queries


# ── Print table ────────────────────────────────────────────────────────────────

def print_results(results: dict, queries: list):
    n = len(queries)
    # ── Main results table with 95% CI on Recall@5 ─────────────────────────
    headers = ["System", "P@5", "R@5 (95% CI)", "NDCG@5", "MAP@10", "MRR", "Latency(ms)"]
    rows = []
    active = {s: r for s, r in results.items() if r["p5"]}
    for s, r in active.items():
        lo, hi = bootstrap_confidence_interval(r["r5"])
        rows.append([
            s,
            f'{sum(r["p5"])/n:.4f}',
            f'{sum(r["r5"])/n:.4f} [{lo:.3f}–{hi:.3f}]',
            f'{sum(r["ndcg5"])/n:.4f}',
            f'{sum(r["ap10"])/n:.4f}',
            f'{sum(r["mrr"])/n:.4f}',
            f'{sum(r["latency"])/n:.1f}',
        ])

    print("=" * 90)
    print(f"ABLATION RESULTS  ({n} queries, top-{TOP_K}, RANDOM_SEED=42)")
    print("=" * 90)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # ── Wilcoxon signed-rank tests: all key pairwise comparisons ────────────
    pairs = [
        ("Vector-Only",   "BM25-Only",      "Recall@5", "r5"),
        ("Vector-Only",   "Hybrid + MMR",   "Recall@5", "r5"),
        ("Vector-Only",   "Hybrid (no MMR)","Recall@5", "r5"),
        ("Hybrid (no MMR)","Hybrid + MMR",  "Recall@5", "r5"),
        ("BM25-Only",     "Hybrid + MMR",   "Recall@5", "r5"),
        ("Vector-Only",   "BM25-Only",      "MRR",      "mrr"),
        ("Vector-Only",   "Hybrid + MMR",   "MRR",      "mrr"),
    ]
    sig_rows = []
    for sys_a, sys_b, metric_name, key in pairs:
        ra = active.get(sys_a, {}).get(key, [])
        rb = active.get(sys_b, {}).get(key, [])
        if ra and rb:
            W, p, sig = wilcoxon_test(ra, rb)
            sig_rows.append([
                sys_a, sys_b, metric_name,
                f"{W:.1f}", f"{p:.4f}",
                "✓ p<0.05" if sig else "✗ n.s."
            ])
    if sig_rows:
        print("\n" + "=" * 90)
        print("STATISTICAL SIGNIFICANCE — Wilcoxon Signed-Rank Test (two-sided, α=0.05)")
        print("=" * 90)
        sig_headers = ["System A", "System B", "Metric", "W", "p-value", "Significant?"]
        print(tabulate(sig_rows, headers=sig_headers, tablefmt="grid"))


# ── Generate charts ─────────────────────────────────────────────────────────────

def save_charts(results: dict, queries: list):
    n = len(queries)
    active = {s: r for s, r in results.items() if r["p5"]}

    # ── 1. System comparison grouped bar ───────────────────────────────────────
    metrics = ["p5", "r5", "ndcg5", "ap10", "mrr"]
    labels  = ["P@5", "R@5", "NDCG@5", "MAP@10", "MRR"]
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    for i, (s, r) in enumerate(active.items()):
        vals = [sum(r[m]) / n for m in metrics]
        offset = (i - len(active) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=s,
                      color=COLOURS[s], alpha=0.9, edgecolor='white', linewidth=0.4)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha='center', va='bottom',
                    fontsize=6.5, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='white', fontsize=11)
    ax.set_ylabel("Score", color='white', fontsize=11)
    ax.set_title("System Ablation — All Metrics", color='white', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1a1040', labelcolor='white', framealpha=0.7)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_tick_params(labelcolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'system_comparison.png'), dpi=150)
    plt.close()
    print(f"[charts] Saved system_comparison.png")

    # ── 2. Category breakdown stacked bar ──────────────────────────────────────
    categories = ["exact", "paraphrase", "conceptual"]
    cat_colors = ["#60A5FA", "#A78BFA", "#34D399"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    sys_names = list(active.keys())
    x = np.arange(len(sys_names))

    bottoms = np.zeros(len(sys_names))
    for cat, col in zip(categories, cat_colors):
        vals = []
        for s in sys_names:
            ch = active[s]["cat_hits"].get(cat, [])
            vals.append(sum(ch) / len(ch) if ch else 0.0)
        ax.bar(x, vals, 0.5, bottom=bottoms, label=cat.capitalize(),
               color=col, alpha=0.85, edgecolor='white', linewidth=0.3)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(sys_names, color='white', fontsize=9, rotation=12)
    ax.set_ylabel("Hit Rate", color='white')
    ax.set_title("Hit Rate by Query Category", color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1a1040', labelcolor='white', framealpha=0.7)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.set_tick_params(labelcolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'category_breakdown.png'), dpi=150)
    plt.close()
    print("[charts] Saved category_breakdown.png")

    # ── 3. Score distribution histogram ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(active), figsize=(4 * len(active), 4),
                              facecolor="#0f0c29")
    if len(active) == 1:
        axes = [axes]
    for ax, (s, r) in zip(axes, active.items()):
        ax.set_facecolor("#1a1040")
        ax.hist(r["ndcg5"], bins=10, color=COLOURS[s], edgecolor='white',
                linewidth=0.5, alpha=0.85)
        ax.set_title(s, color='white', fontsize=8)
        ax.set_xlabel("NDCG@5", color='white', fontsize=8)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#444')
    fig.suptitle("NDCG@5 Score Distribution per System", color='white',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'score_distribution.png'), dpi=150)
    plt.close()
    print("[charts] Saved score_distribution.png")

    # ── 4. Latency bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    sys_names = list(active.keys())
    latencies = [sum(active[s]["latency"]) / n for s in sys_names]
    bars = ax.bar(sys_names, latencies,
                  color=[COLOURS[s] for s in sys_names],
                  edgecolor='white', linewidth=0.5, alpha=0.9)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}ms", ha='center', va='bottom', color='white', fontsize=9)
    ax.set_ylabel("Avg Query Time (ms)", color='white')
    ax.set_title("Retrieval Latency per System", color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    plt.xticks(rotation=12, color='white', fontsize=9)
    ax.yaxis.set_tick_params(labelcolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'latency_comparison.png'), dpi=150)
    plt.close()
    print("[charts] Saved latency_comparison.png")

    # ── 5. Chapter coverage pie chart ─────────────────────────────────────────
    from indexing.vector_index import load_clauses
    clauses = load_clauses()
    chapter_map = {c["section_number"]: c.get("chapter", 0) for c in clauses}
    all_chapters = set(chapter_map.values()) - {0}

    # For hybrid+MMR: which chapters had at least 1 hit?
    hyb = active.get("Hybrid + MMR", None)
    if hyb:
        covered_chapters = set()
        for q in queries:
            for sec in q["relevant_sections"]:
                if sec in chapter_map and chapter_map[sec] in all_chapters:
                    # Count as covered if hybrid retrieved it
                    covered_chapters.add(chapter_map[sec])
        # Actually check hits
        covered = sum(1 for h in hyb["hits"] if h == 1)
        missed  = len(queries) - covered
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0f0c29")
        ax.set_facecolor("#1a1040")
        wedge_vals  = [covered, missed]
        wedge_cols  = ["#A78BFA", "#EF4444"]
        wedge_labels = [f"Retrieved ({covered})", f"Missed ({missed})"]
        ax.pie(wedge_vals, labels=wedge_labels, colors=wedge_cols,
               autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 12},
               startangle=90, wedgeprops=dict(edgecolor='white', linewidth=0.5))
        ax.set_title("Hybrid+MMR: Query Hit Coverage", color='white',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, 'chapter_coverage.png'), dpi=150)
        plt.close()
        print("[charts] Saved chapter_coverage.png")

    # ── 6. BM25 vs Vector score scatter ────────────────────────────────────────
    from indexing.vector_index import get_or_build_index
    from indexing.bm25_index import get_or_build_bm25
    from retrieval.hybrid import hybrid_search

    clauses2 = load_clauses()
    index2, emb2, model2 = get_or_build_index(clauses2)
    bm25_2, _ = get_or_build_bm25(clauses2)

    # Collect top-20 candidates for first 20 queries
    bm25_scores, vec_scores = [], []
    for q in queries[:20]:
        cands = hybrid_search(q["query"], clauses2, index2, emb2, model2, bm25_2, top_k=20)
        for c in cands:
            bm25_scores.append(c.get("bm25_score", 0))
            vec_scores.append(c.get("vector_score", 0))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    ax.scatter(bm25_scores, vec_scores, alpha=0.4, s=20,
               color="#A78BFA", edgecolors="white", linewidths=0.2)
    ax.set_xlabel("BM25 Score (normalised)", color='white', fontsize=11)
    ax.set_ylabel("Vector Score (normalised)", color='white', fontsize=11)
    ax.set_title("BM25 vs Vector Score Correlation\n(top-20 candidates, first 20 queries)",
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.yaxis.set_tick_params(labelcolor='white')
    ax.xaxis.set_tick_params(labelcolor='white')

    # Add correlation coefficient
    if len(bm25_scores) > 1:
        corr = np.corrcoef(bm25_scores, vec_scores)[0, 1]
        ax.text(0.05, 0.93, f"r = {corr:.3f}", transform=ax.transAxes,
                color='#FACC15', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'score_correlation.png'), dpi=150)
    plt.close()
    print("[charts] Saved score_correlation.png")

    # ── Combined evaluation_results.png (original chart for app sidebar) ───────
    old_chart = os.path.join(BASE_DIR, 'data', 'evaluation_results.png')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    metric_keys = ["r5", "ndcg5", "mrr"]
    metric_names = ["Recall@5", "NDCG@5", "MRR"]
    x = np.arange(len(metric_names))
    width = 0.15
    for i, (s, r) in enumerate(active.items()):
        vals = [sum(r[m]) / n for m in metric_keys]
        offset = (i - len(active) / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=s,
               color=COLOURS[s], alpha=0.9, edgecolor='white', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color='white', fontsize=12)
    ax.set_ylabel("Score", color='white')
    ax.set_title(f"IPC Legal Retrieval — Ablation Study ({n} queries)",
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#1a1040', labelcolor='white', framealpha=0.7)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_tick_params(labelcolor='white')
    plt.tight_layout()
    plt.savefig(old_chart, dpi=150)
    plt.close()
    print(f"[charts] Updated evaluation_results.png")

    print(f"\n[charts] All charts saved -> {CHARTS_DIR}/")


# ── Query Rewriting Impact ──────────────────────────────────────────────────────

def run_rewriting_impact() -> None:
    """
    Measure the Recall@5 delta for key systems with vs without LLM query rewriting.
    Prints a formatted table and saves rewriting_impact.png.

    Systems evaluated: BM25-Only, Vector-Only, Hybrid + MMR
    Rewriting uses query_rewriter.py (requires Ollama to be running).
    """
    with open(QUERIES_PATH, encoding='utf-8') as f:
        queries = json.load(f)

    components = load_all()
    n = len(queries)

    target_systems = ["BM25-Only", "Vector-Only", "Hybrid + MMR"]

    # baseline (no rewriting)
    base = {s: [] for s in target_systems}
    for q in queries:
        for s in target_systems:
            secs, _ = retrieve(s, q["query"], components)
            base[s].append(recall_at_k(secs, q["relevant_sections"]))

    # with LLM query rewriting
    import importlib
    try:
        qr_mod = importlib.import_module("retrieval.query_rewriter")
        rewrite_fn = getattr(qr_mod, "rewrite_query", None)
        if rewrite_fn is None:
            print("[rewriting] query_rewriter.rewrite_query not found — skipping rewriting eval")
            return
    except ImportError:
        print("[rewriting] retrieval.query_rewriter not found — skipping rewriting eval")
        return

    rewrite = {s: [] for s in target_systems}
    print("\n[rewriting] Running rewriting impact evaluation...")
    for i, q in enumerate(queries, 1):
        sys.stdout.write(f"\r  [{i:3}/{n}]")
        sys.stdout.flush()
        try:
            expanded = rewrite_fn(q["query"])
            fused_q = q["query"] + " " + expanded if expanded else q["query"]
        except Exception:
            fused_q = q["query"]
        for s in target_systems:
            secs, _ = retrieve(s, fused_q, components)
            rewrite[s].append(recall_at_k(secs, q["relevant_sections"]))

    print("\n")

    # Print table
    rw_headers = ["System", "R@5 (No Rewrite)", "R@5 (With Rewrite)", "Δ Recall@5", "Δ%"]
    rw_rows = []
    for s in target_systems:
        b = sum(base[s]) / n
        r = sum(rewrite[s]) / n
        delta = r - b
        pct = (delta / b * 100) if b > 0 else 0.0
        rw_rows.append([s, f"{b:.4f}", f"{r:.4f}",
                        f"{delta:+.4f}", f"{pct:+.1f}%"])

    print("\n" + "=" * 65)
    print("QUERY REWRITING IMPACT — Recall@5 (with vs without LLM rewriting)")
    print("=" * 65)
    print(tabulate(rw_rows, headers=rw_headers, tablefmt="grid"))

    # Chart
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0c29")
    ax.set_facecolor("#1a1040")
    x = np.arange(len(target_systems))
    width = 0.32
    bars_base = ax.bar(x - width/2, [sum(base[s])/n for s in target_systems],
                       width, label="No Rewriting", color="#60A5FA", alpha=0.9,
                       edgecolor='white', linewidth=0.4)
    bars_rw = ax.bar(x + width/2, [sum(rewrite[s])/n for s in target_systems],
                     width, label="With LLM Rewriting", color="#34D399", alpha=0.9,
                     edgecolor='white', linewidth=0.4)
    for bar in list(bars_base) + list(bars_rw):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha='center', va='bottom', fontsize=8, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(target_systems, color='white', fontsize=10)
    ax.set_ylabel("Recall@5", color='white', fontsize=11)
    ax.set_title("Query Rewriting Impact on Recall@5\n(LLM vocabulary expansion vs raw query)",
                 color='white', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1a1040', labelcolor='white', framealpha=0.7)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_tick_params(labelcolor='white')
    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, 'rewriting_impact.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[charts] Saved rewriting_impact.png")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results, queries = run_evaluation(skip_cross_encoder=True)
    print_results(results, queries)
    print("\nGenerating charts...")
    save_charts(results, queries)
    print("\n[Optional] Running query rewriting impact evaluation...")
    print("  (Requires Ollama to be running. Skip with Ctrl+C if not available.)")
    try:
        run_rewriting_impact()
    except KeyboardInterrupt:
        print("\n[skipped] Rewriting impact evaluation cancelled.")
    print("\nDone.")
