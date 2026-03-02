"""
regenerate_paper_charts.py
Regenerates all paper figures (assets/paper_figures/) with the corrected
real measured data from the 633-clause 5-system evaluation.

Run from: legal_rag/
  python regenerate_paper_charts.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Output dirs ──────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Real measured data (sensitivity_results.json @K=5) ─────────────────────
SYSTEMS   = ["BM25-Only", "TF-IDF", "Vector-Only", "Hybrid\n(no MMR)", "Hybrid+MMR"]
SYS_SHORT = ["BM25",      "TF-IDF", "Vector",       "Hybrid",           "H+MMR"]
COLORS    = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#F5A623"]
HATCHES   = ["", "//", "", "\\\\", "xx"]

# From sensitivity_results.json (K=5 results)
RECALL5  = [0.654, 0.624, 0.743, 0.609, 0.568]
MRR      = [0.548, 0.522, 0.630, 0.502, 0.484]
NDCG5    = [0.544, 0.521, 0.624, 0.501, 0.472]
PREC5    = [0.1789, 0.1718, 0.2071, 0.1682, 0.1544]
MAP10    = [0.4912, 0.4703, 0.5698, 0.4528, 0.4268]
LATENCY  = [1.1, 2.0, 11.8, 13.5, 16.6]

# Per-category (from section 9.4.6)
EXACT_R   = [0.833, 0.785, 0.904, 0.801, 0.748]
PARA_R    = [0.325, 0.333, 0.450, 0.250, 0.233]
CONC_R    = [0.384, 0.379, 0.497, 0.324, 0.299]

# Top-K data
K_VALUES  = [3, 5, 10]
TOPK_DATA = {
    "BM25-Only":       {"recall": [0.575, 0.654, 0.740], "mrr": [0.528, 0.548, 0.560]},
    "TF-IDF":          {"recall": [0.535, 0.624, 0.731], "mrr": [0.501, 0.522, 0.538]},
    "Vector-Only":     {"recall": [0.643, 0.743, 0.818], "mrr": [0.610, 0.630, 0.638]},
    "Hybrid (no MMR)": {"recall": [0.515, 0.609, 0.675], "mrr": [0.479, 0.502, 0.514]},
    "Hybrid+MMR":      {"recall": [0.449, 0.568, 0.647], "mrr": [0.453, 0.484, 0.493]},
}

STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D26",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#B0B0B0",
    "ytick.color":      "#B0B0B0",
    "text.color":       "#E0E0E0",
    "grid.color":       "#2A2D3D",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "DejaVu Sans",
}

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Aggregate Performance — grouped bar chart (4 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig_aggregate_perf():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(18, 6))
        fig.suptitle("Aggregate Retrieval Performance — 132 Queries × 5 Systems",
                     fontsize=16, fontweight="bold", color="#FFFFFF", y=1.01)

        datasets  = [RECALL5, MRR, NDCG5, PREC5]
        titles    = ["Recall@5", "MRR", "nDCG@5", "P@5"]
        ylims     = [(0, 0.95), (0, 0.80), (0, 0.80), (0, 0.28)]

        for ax, data, title, ylim in zip(axes, datasets, titles, ylims):
            bars = ax.bar(SYS_SHORT, data, color=COLORS, width=0.65, edgecolor="#111",
                          linewidth=0.8)
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", color="#FFFFFF")
            best = max(data)
            for bar, val in zip(bars, data):
                if val == best:
                    bar.set_edgecolor("#FFD700")
                    bar.set_linewidth(2.5)
            ax.set_title(title, fontsize=13, fontweight="bold", color="#FFFFFF", pad=8)
            ax.set_ylim(ylim)
            ax.tick_params(axis="x", labelsize=9, rotation=0)
            ax.grid(axis="y")
            ax.set_axisbelow(True)

        note = ("Vector-Only (gold border) leads on all metrics.  "
                "Hybrid+MMR ranks last despite highest latency.  "
                "133-clause combined corpus (IPC 575 + BNS 58)")
        fig.text(0.5, -0.04, note, ha="center", fontsize=10, color="#A0A0A0", style="italic")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig_aggregate_perf.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close()
        print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Latency vs Accuracy scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_latency_tradeoff():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Accuracy vs. Latency Tradeoff — 5-System Comparison",
                     fontsize=14, fontweight="bold", color="#FFFFFF", pad=12)

        for i, (sys, lat, r5) in enumerate(zip(SYS_SHORT, LATENCY, RECALL5)):
            ax.scatter(lat, r5, s=280, color=COLORS[i], zorder=5,
                       edgecolors="#FFFFFF", linewidths=1.2)
            dx = 0.4 if sys not in ("H+MMR",) else -0.6
            ax.annotate(f"  {sys}\n  R@5={r5:.3f}", (lat, r5),
                        fontsize=10, color="#E0E0E0",
                        xytext=(dx, 0.008), textcoords="offset points",
                        annotation_clip=False)

        ax.axvline(2.0,  color="#666", lw=1, ls=":")
        ax.axhline(0.654, color="#4878CF", lw=1, ls="--", alpha=0.5,
                   label="BM25 Recall@5 (0.654) — deployment floor")

        ax.set_xlabel("Retrieval Latency (ms)", fontsize=12)
        ax.set_ylabel("Recall@5", fontsize=12)
        ax.set_xlim(-0.5, 20)
        ax.set_ylim(0.45, 0.82)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True)

        note = ("Ideal system: top-right corner.  "
                "Vector-Only (11.8ms, 0.743) is the practical optimum.  "
                "Hybrid+MMR is slowest AND least accurate at N=633.")
        fig.text(0.5, -0.04, note, ha="center", fontsize=10,
                 color="#A0A0A0", style="italic")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig_latency_tradeoff.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close()
        print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Query difficulty tiers — grouped bars
# ─────────────────────────────────────────────────────────────────────────────
def fig_difficulty_tiers():
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Recall@5 by Query Difficulty Tier — All 5 Systems",
                     fontsize=15, fontweight="bold", color="#FFFFFF", y=1.02)

        tier_data = [
            ("Exact Terminology (n=44)",   EXACT_R),
            ("Paraphrase (n=44)",           PARA_R),
            ("Conceptual (n=44)",           CONC_R),
        ]

        for ax, (title, data) in zip(axes, tier_data):
            bars = ax.bar(SYS_SHORT, data, color=COLORS, width=0.65,
                          edgecolor="#111", linewidth=0.8)
            best = max(data)
            for bar, val in zip(bars, data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", color="#FFFFFF")
                if val == best:
                    bar.set_edgecolor("#FFD700")
                    bar.set_linewidth(2.5)
            ax.set_title(title, fontsize=11, fontweight="bold", color="#FFFFFF", pad=8)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.set_ylabel("Recall@5", fontsize=11)

        note = ("Paraphrase is the hardest tier for all systems.  "
                "Hybrid systems perform worst even on Exact queries due to fusion miscalibration.")
        fig.text(0.5, -0.04, note, ha="center", fontsize=10,
                 color="#A0A0A0", style="italic")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig_difficulty_tiers.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close()
        print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Corpus quality impact (IPC-only experiment — shows real improvement)
# ─────────────────────────────────────────────────────────────────────────────
def fig_corpus_quality():
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Corpus Quality Impact on Vector-Only Retrieval\n"
                     "(IPC-Only Experiment: PDF-extracted vs. Curated JSON)",
                     fontsize=13, fontweight="bold", color="#FFFFFF", pad=12)

        categories = ["PDF-extracted\n(455 sections,\n87 chars avg)",
                      "Curated JSON\n(575 sections,\n312 chars avg)"]
        recall_vals = [0.625, 0.764]
        mrr_vals    = [0.556, 0.705]
        ndcg_vals   = [0.521, 0.683]
        x = np.arange(len(categories))
        width = 0.25

        b1 = ax.bar(x - width, recall_vals, width, label="Recall@5",  color="#4878CF")
        b2 = ax.bar(x,         mrr_vals,    width, label="MRR",        color="#D65F5F")
        b3 = ax.bar(x + width, ndcg_vals,   width, label="nDCG@5",     color="#6ACC65")

        for bars in [b1, b2, b3]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color="#FFFFFF")

        # Annotate improvement
        ax.annotate("", xy=(1 - width, 0.764), xytext=(0 - width, 0.625),
                    arrowprops=dict(arrowstyle="->", color="#FFD700", lw=2))
        ax.text(0.5, 0.72, "+22.2% Recall@5\n+26.8% MRR\n+31.1% nDCG@5",
                ha="center", fontsize=11, color="#FFD700", fontweight="bold",
                transform=ax.transData)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=12)
        ax.set_ylim(0, 0.95)
        ax.legend(fontsize=11)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

        note = ("Data quality improvement exceeds ANY algorithmic gain tested.  "
                "No code change — just better extracted text per clause.")
        fig.text(0.5, -0.04, note, ha="center", fontsize=10,
                 color="#A0A0A0", style="italic")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig_corpus_quality.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close()
        print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 (bonus): Top-K sensitivity line chart — all 5 systems
# ─────────────────────────────────────────────────────────────────────────────
def fig_topk_sensitivity():
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Top-K Sensitivity — Recall@K and MRR across K ∈ {3, 5, 10}",
                     fontsize=14, fontweight="bold", color="#FFFFFF", y=1.02)

        for i, (sys, color) in enumerate(zip(list(TOPK_DATA.keys()), COLORS)):
            ax1.plot(K_VALUES, TOPK_DATA[sys]["recall"], "o-", color=color,
                     label=sys, lw=2, ms=8)
            ax2.plot(K_VALUES, TOPK_DATA[sys]["mrr"], "s--", color=color,
                     label=sys, lw=2, ms=8)

        for ax, title, ylabel in zip([ax1, ax2],
                                     ["Recall@K", "MRR"],
                                     ["Recall", "MRR"]):
            ax.set_title(title, fontsize=12, fontweight="bold", color="#FFFFFF")
            ax.set_xlabel("K (Top-K retrieved)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(K_VALUES)
            ax.set_ylim(0.35, 0.90)
            ax.legend(fontsize=9, loc="upper left")
            ax.grid(True)
            ax.axvline(5, color="#FFD700", lw=1.5, ls=":", alpha=0.8,
                       label="K=5 (optimal)")

        plt.tight_layout()
        path = os.path.join(OUT_DIR, "fig_topk_sensitivity.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["figure.facecolor"])
        plt.close()
        print(f"[✓] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Regenerating all paper figures with corrected 633-clause 5-system data...\n")
    fig_aggregate_perf()
    fig_latency_tradeoff()
    fig_difficulty_tiers()
    fig_corpus_quality()
    fig_topk_sensitivity()
    print("\n✅ All figures saved to assets/paper_figures/")
    print("   fig_aggregate_perf.png  — Sec 8.2")
    print("   fig_latency_tradeoff.png — Sec 8.2")
    print("   fig_difficulty_tiers.png — Sec 9.1")
    print("   fig_corpus_quality.png  — Sec 9.2")
    print("   fig_topk_sensitivity.png — Sec 9.4.2 (new)")
