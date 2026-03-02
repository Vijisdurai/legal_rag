"""
Generate comprehensive diagrams and charts for Legal RAG Research
Infrastructure, Architecture, and Key Findings Visualization
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os

# Create output directory
os.makedirs('output', exist_ok=True)

# Color scheme
COLORS = {
    'primary': '#1B2A4A',
    'secondary': '#2C5282',
    'accent': '#4299E1',
    'success': '#276749',
    'warning': '#744210',
    'light': '#F7F8FA',
    'border': '#E2E8F0'
}

# Research data from paper
EVAL_DATA = {
    "System": ["BM25-Only", "TF-IDF", "Vector-Only", "Hybrid+MMR"],
    "P@5": [0.1894, 0.1864, 0.2258, 0.1470],
    "R@5": [0.6890, 0.6504, 0.7637, 0.5685],
    "NDCG@5": [0.5891, 0.5619, 0.6826, 0.4611],
    "MAP@10": [0.5331, 0.5099, 0.6237, 0.4108],
    "MRR": [0.5975, 0.5665, 0.7053, 0.4619],
    "Latency(ms)": [1.1, 2.0, 11.8, 16.6],
}

CATEGORY_DATA = {
    "Exact": {"R@5": [0.818, 0.773, 0.727, 0.636], "MRR": [0.742, 0.698, 0.681, 0.592]},
    "Paraphrase": {"R@5": [0.636, 0.614, 0.773, 0.568], "MRR": [0.551, 0.533, 0.718, 0.435]},
    "Conceptual": {"R@5": [0.614, 0.568, 0.750, 0.500], "MRR": [0.499, 0.468, 0.717, 0.409]},
}

CORPUS_QUALITY = [
    {"version": "PDF-extracted", "clauses": 455, "avg_len": 87, "r5": 0.625, "mrr": 0.556},
    {"version": "Clean JSON", "clauses": 575, "avg_len": 312, "r5": 0.764, "mrr": 0.705},
]


def create_system_architecture():
    """Generate comprehensive system architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Hybrid-MMR Legal RAG System Architecture', 
            ha='center', fontsize=20, fontweight='bold', color=COLORS['primary'])
    ax.text(8, 11, 'Indian Penal Code (IPC 1860) + Bharatiya Nyaya Sanhita (BNS 2023)',
            ha='center', fontsize=12, color=COLORS['secondary'])
    
    # Layer 1: Data Sources
    y_start = 9.5
    ax.text(8, y_start + 0.5, 'DATA LAYER', ha='center', fontsize=11, 
            fontweight='bold', color=COLORS['primary'])
    
    boxes = [
        {'x': 1, 'text': 'IPC PDF\n(1860)\n511 sections', 'color': '#EBF4FF'},
        {'x': 4, 'text': 'BNS Gazette\n(2023)\n358 sections', 'color': '#EBF4FF'},
        {'x': 7, 'text': 'IPC-BNS\nMapping\n130+ pairs', 'color': '#F0FFF4'},
        {'x': 10, 'text': 'Evaluation\nQueries\n132 queries', 'color': '#FFFBEB'},
        {'x': 13, 'text': 'Ground Truth\nAnnotations\n3 categories', 'color': '#FFF5F5'}
    ]
    
    for box in boxes:
        rect = FancyBboxPatch((box['x'], y_start-0.8), 2, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor=box['color'], 
                              edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1, y_start-0.3, box['text'], ha='center', va='center',
                fontsize=9, fontweight='600')

    
    # Layer 2: Preprocessing
    y_start = 7.5
    ax.text(8, y_start + 0.5, 'PREPROCESSING LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    preproc_boxes = [
        {'x': 1.5, 'text': 'PDF Extraction\npdfplumber\nTesseract OCR', 'color': '#F7F8FA'},
        {'x': 5, 'text': 'Text Cleaning\nSegmentation\n575 IPC clauses', 'color': '#F7F8FA'},
        {'x': 8.5, 'text': 'BNS Curation\n58 key sections\nManual QA', 'color': '#F7F8FA'},
        {'x': 12, 'text': 'Corpus Merge\n633 total clauses\nMetadata tagging', 'color': '#F7F8FA'}
    ]
    
    for box in preproc_boxes:
        rect = FancyBboxPatch((box['x'], y_start-0.8), 2.5, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=box['color'],
                              edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.25, y_start-0.3, box['text'], ha='center', va='center',
                fontsize=8, fontweight='500')
    
    # Layer 3: Indexing
    y_start = 5.5
    ax.text(8, y_start + 0.5, 'INDEXING LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    index_boxes = [
        {'x': 1, 'text': 'FAISS Vector Index\nall-MiniLM-L6-v2\n384-dim embeddings\nIndexFlatIP', 'color': '#EBF8FF'},
        {'x': 5.5, 'text': 'BM25 Index\nBM25Okapi\nTokenized corpus\nrank_bm25', 'color': '#F0FFF4'},
        {'x': 10, 'text': 'TF-IDF Index\nscikit-learn\nUnigrams+Bigrams\nCosine similarity', 'color': '#FFFBEB'}
    ]
    
    for box in index_boxes:
        rect = FancyBboxPatch((box['x'], y_start-0.9), 3.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=box['color'],
                              edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.75, y_start-0.3, box['text'], ha='center', va='center',
                fontsize=8, fontweight='500', linespacing=1.5)

    
    # Layer 4: Retrieval Pipeline
    y_start = 3
    ax.text(8, y_start + 0.5, 'RETRIEVAL PIPELINE', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    # Query input
    query_box = FancyBboxPatch((6.5, y_start-0.3), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFFBEB',
                               edgecolor=COLORS['secondary'], linewidth=3)
    ax.add_patch(query_box)
    ax.text(8, y_start, 'Natural Language Query', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Retrieval methods
    retrieval_y = y_start - 1.5
    methods = [
        {'x': 0.5, 'text': 'BM25\nSparse\n1.1ms', 'color': '#F0FFF4'},
        {'x': 3.5, 'text': 'TF-IDF\nStatistical\n2.0ms', 'color': '#FFFBEB'},
        {'x': 6.5, 'text': 'Dense Vector\nSemantic\n11.8ms', 'color': '#EBF8FF'},
        {'x': 9.5, 'text': 'Hybrid Fusion\n0.6×Vec + 0.4×BM25\n', 'color': '#EBF4FF'},
        {'x': 12.5, 'text': 'MMR Rerank\nλ=0.9\nDiversity', 'color': '#F7FAFC'}
    ]
    
    for method in methods:
        rect = FancyBboxPatch((method['x'], retrieval_y), 2.5, 0.9,
                              boxstyle="round,pad=0.1",
                              facecolor=method['color'],
                              edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(method['x']+1.25, retrieval_y+0.45, method['text'], 
                ha='center', va='center', fontsize=8, fontweight='500')
    
    # Layer 5: Generation
    y_start = 0.8
    ax.text(8, y_start + 0.3, 'GENERATION LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    gen_box = FancyBboxPatch((4, y_start-0.5), 8, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#EBF4FF',
                             edgecolor=COLORS['secondary'], linewidth=3)
    ax.add_patch(gen_box)
    ax.text(8, y_start-0.1, 'LLM Answer Generation (GPT-OSS 120B via Ollama)\nGrounded in Retrieved Sections • Citation-Based • Hallucination Guard',
            ha='center', va='center', fontsize=9, fontweight='600')
    
    plt.tight_layout()
    plt.savefig('output/01_system_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 01_system_architecture.png")



def create_retrieval_pipeline_flow():
    """Detailed retrieval pipeline flowchart"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Hybrid-MMR Retrieval Pipeline Flow', 
            ha='center', fontsize=18, fontweight='bold', color=COLORS['primary'])
    
    # Step 1: Query Input
    y = 8.5
    rect = FancyBboxPatch((5, y), 4, 0.7, boxstyle="round,pad=0.1",
                          facecolor='#FFFBEB', edgecolor=COLORS['secondary'], linewidth=3)
    ax.add_patch(rect)
    ax.text(7, y+0.35, 'User Query (Natural Language)', ha='center', va='center',
            fontsize=11, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(7, y), xytext=(7, y-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    # Step 2: Parallel Retrieval
    y = 6.5
    ax.text(7, y+1, 'Parallel Retrieval (4 Methods)', ha='center', fontsize=10,
            fontweight='bold', color=COLORS['primary'])
    
    methods = [
        {'x': 0.5, 'name': 'BM25', 'desc': 'Token matching\nTF-IDF weighting', 'time': '1.1ms'},
        {'x': 3.5, 'name': 'TF-IDF', 'desc': 'Statistical\nCosine similarity', 'time': '2.0ms'},
        {'x': 6.5, 'name': 'Dense Vector', 'desc': 'Semantic embedding\nFAISS search', 'time': '11.8ms'},
        {'x': 9.5, 'name': 'Hybrid', 'desc': 'Fusion\n0.6×Vec+0.4×BM25', 'time': '—'}
    ]
    
    for m in methods:
        rect = FancyBboxPatch((m['x'], y), 2.5, 1.2, boxstyle="round,pad=0.1",
                              facecolor='#EBF8FF', edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(m['x']+1.25, y+0.85, m['name'], ha='center', fontsize=10, fontweight='bold')
        ax.text(m['x']+1.25, y+0.45, m['desc'], ha='center', fontsize=8)
        ax.text(m['x']+1.25, y+0.1, m['time'], ha='center', fontsize=7, 
                style='italic', color=COLORS['secondary'])

    
    # Step 3: Score Normalization & Fusion
    y = 4.5
    for m_x in [1.75, 4.75, 7.75]:
        ax.annotate('', xy=(7, y+1), xytext=(m_x, y+2.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['border']))
    
    rect = FancyBboxPatch((4.5, y), 5, 0.9, boxstyle="round,pad=0.1",
                          facecolor='#F0FFF4', edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(rect)
    ax.text(7, y+0.65, 'Score Normalization (Min-Max)', ha='center', fontsize=9, fontweight='bold')
    ax.text(7, y+0.35, 'Hybrid = 0.6 × norm(vector) + 0.4 × norm(BM25)', ha='center', fontsize=8)
    ax.text(7, y+0.05, 'Top-50 candidates selected', ha='center', fontsize=7, style='italic')
    
    # Arrow down
    ax.annotate('', xy=(7, y), xytext=(7, y-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    # Step 4: Corpus Filtering
    y = 3
    rect = FancyBboxPatch((4.5, y), 5, 0.7, boxstyle="round,pad=0.1",
                          facecolor='#FFFBEB', edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(rect)
    ax.text(7, y+0.35, 'Corpus Filtering: IPC / BNS / Both', ha='center', fontsize=9, fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(7, y), xytext=(7, y-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    # Step 5: MMR Reranking
    y = 1.5
    rect = FancyBboxPatch((4, y), 6, 0.9, boxstyle="round,pad=0.1",
                          facecolor='#EBF4FF', edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(rect)
    ax.text(7, y+0.65, 'MMR Reranking (λ=0.9)', ha='center', fontsize=10, fontweight='bold')
    ax.text(7, y+0.35, 'MMR(c) = λ·Relevance(c) − (1−λ)·max Similarity(c, selected)', 
            ha='center', fontsize=8)
    ax.text(7, y+0.05, 'Iterative greedy selection for diversity', ha='center', fontsize=7, style='italic')
    
    # Arrow down
    ax.annotate('', xy=(7, y), xytext=(7, y-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['primary']))
    
    # Step 6: Final Results
    y = 0.3
    rect = FancyBboxPatch((5, y), 4, 0.6, boxstyle="round,pad=0.1",
                          facecolor='#F0FFF4', edgecolor=COLORS['success'], linewidth=3)
    ax.add_patch(rect)
    ax.text(7, y+0.3, 'Top-5 Ranked Legal Clauses', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/02_retrieval_pipeline_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 02_retrieval_pipeline_flow.png")



def create_performance_comparison():
    """Key findings: Performance comparison across systems"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor='white')
    fig.suptitle('Key Research Findings: Retrieval System Performance Comparison\n132 Queries • 4 Systems • IPC + BNS 2023',
                 fontsize=16, fontweight='bold', color=COLORS['primary'], y=0.98)
    
    systems = EVAL_DATA["System"]
    colors = ['#9CB3C9', '#B5C8A8', '#4299E1', '#1B2A4A']
    
    # Chart 1: Recall@5
    ax = axes[0, 0]
    bars = ax.bar(systems, EVAL_DATA["R@5"], color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Recall@5', fontsize=11, fontweight='600')
    ax.set_title('Recall@5 (Higher is Better)', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (bar, val) in enumerate(zip(bars, EVAL_DATA["R@5"])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    # Chart 2: NDCG@5
    ax = axes[0, 1]
    bars = ax.bar(systems, EVAL_DATA["NDCG@5"], color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('NDCG@5', fontsize=11, fontweight='600')
    ax.set_title('NDCG@5 (Ranking Quality)', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (bar, val) in enumerate(zip(bars, EVAL_DATA["NDCG@5"])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    # Chart 3: MRR
    ax = axes[0, 2]
    bars = ax.bar(systems, EVAL_DATA["MRR"], color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('MRR', fontsize=11, fontweight='600')
    ax.set_title('Mean Reciprocal Rank', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (bar, val) in enumerate(zip(bars, EVAL_DATA["MRR"])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)

    
    # Chart 4: Latency
    ax = axes[1, 0]
    bars = ax.bar(systems, EVAL_DATA["Latency(ms)"], color=colors, edgecolor='white', linewidth=2)
    ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='600')
    ax.set_title('Retrieval Latency (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (bar, val) in enumerate(zip(bars, EVAL_DATA["Latency(ms)"])):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}ms',
                ha='center', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    # Chart 5: Accuracy vs Latency Scatter
    ax = axes[1, 1]
    for i, (sys, ndcg, lat) in enumerate(zip(systems, EVAL_DATA["NDCG@5"], EVAL_DATA["Latency(ms)"])):
        ax.scatter(lat, ndcg, s=200, color=colors[i], edgecolors='white', linewidth=2, zorder=5)
        ax.annotate(sys, (lat, ndcg), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='600')
    ax.set_xlabel('Latency (ms)', fontsize=11, fontweight='600')
    ax.set_ylabel('NDCG@5', fontsize=11, fontweight='600')
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Chart 6: % Change vs Vector-Only
    ax = axes[1, 2]
    vec_ndcg = EVAL_DATA["NDCG@5"][2]
    deltas = [(EVAL_DATA["NDCG@5"][i] - vec_ndcg) / vec_ndcg * 100 
              for i in range(len(systems)) if systems[i] != "Vector-Only"]
    sys_names = [s for s in systems if s != "Vector-Only"]
    bar_colors = ['#276749' if d > 0 else '#9B2C2C' for d in deltas]
    
    bars = ax.barh(sys_names, deltas, color=bar_colors, edgecolor='white', linewidth=2)
    ax.axvline(0, color='#CBD5E0', linewidth=2, linestyle='--')
    ax.set_xlabel('% Change vs Vector-Only', fontsize=11, fontweight='600')
    ax.set_title('Relative Performance (NDCG@5)', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, deltas):
        ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=10, fontweight='bold',
                color='#276749' if val >= 0 else '#9B2C2C')
    
    plt.tight_layout()
    plt.savefig('output/03_performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 03_performance_comparison.png")



def create_query_category_analysis():
    """Query category breakdown: Exact vs Paraphrase vs Conceptual"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    fig.suptitle('Query Category Analysis: System Performance by Query Type\n44 Exact • 44 Paraphrase • 44 Conceptual',
                 fontsize=14, fontweight='bold', color=COLORS['primary'], y=0.98)
    
    systems = EVAL_DATA["System"]
    categories = list(CATEGORY_DATA.keys())
    x = np.arange(len(systems))
    width = 0.25
    colors_cat = ['#9CB3C9', '#4299E1', '#1B2A4A']
    
    # Chart 1: Recall@5 by Category
    ax = axes[0]
    for i, cat in enumerate(categories):
        values = CATEGORY_DATA[cat]["R@5"]
        bars = ax.bar(x + i*width, values, width, label=cat, color=colors_cat[i], 
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                    ha='center', fontsize=8, fontweight='600')
    
    ax.set_ylabel('Recall@5', fontsize=11, fontweight='600')
    ax.set_title('Recall@5 by Query Category', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Chart 2: MRR by Category
    ax = axes[1]
    for i, cat in enumerate(categories):
        values = CATEGORY_DATA[cat]["MRR"]
        bars = ax.bar(x + i*width, values, width, label=cat, color=colors_cat[i],
                      edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                    ha='center', fontsize=8, fontweight='600')
    
    ax.set_ylabel('MRR', fontsize=11, fontweight='600')
    ax.set_title('Mean Reciprocal Rank by Query Category', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x + width)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('output/04_query_category_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 04_query_category_analysis.png")



def create_corpus_quality_impact():
    """Central finding: Corpus quality impact"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    fig.suptitle('Central Research Finding: Corpus Quality Dominates Algorithm Choice\nPDF-Extracted vs Clean Curated JSON',
                 fontsize=14, fontweight='bold', color=COLORS['primary'], y=0.98)
    
    versions = [c["version"] for c in CORPUS_QUALITY]
    colors_qual = ['#FED7D7', '#C6F6D5']
    
    # Chart 1: Recall@5 Improvement
    ax = axes[0]
    r5_vals = [c["r5"] for c in CORPUS_QUALITY]
    bars = ax.bar(versions, r5_vals, color=colors_qual, edgecolor=COLORS['border'], linewidth=2)
    ax.set_ylabel('Recall@5', fontsize=11, fontweight='600')
    ax.set_title('Recall@5 Improvement', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, r5_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    improvement = (r5_vals[1] - r5_vals[0]) / r5_vals[0] * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(1, r5_vals[1]), xytext=(1.3, r5_vals[1]),
                fontsize=14, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['success']))
    
    # Chart 2: MRR Improvement
    ax = axes[1]
    mrr_vals = [c["mrr"] for c in CORPUS_QUALITY]
    bars = ax.bar(versions, mrr_vals, color=colors_qual, edgecolor=COLORS['border'], linewidth=2)
    ax.set_ylabel('MRR', fontsize=11, fontweight='600')
    ax.set_title('MRR Improvement', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, mrr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    
    improvement = (mrr_vals[1] - mrr_vals[0]) / mrr_vals[0] * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(1, mrr_vals[1]), xytext=(1.3, mrr_vals[1]),
                fontsize=14, fontweight='bold', color=COLORS['success'],
                arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['success']))
    
    # Chart 3: Corpus Statistics
    ax = axes[2]
    metrics = ['Clauses', 'Avg Length\n(chars)']
    pdf_vals = [CORPUS_QUALITY[0]["clauses"], CORPUS_QUALITY[0]["avg_len"]]
    clean_vals = [CORPUS_QUALITY[1]["clauses"], CORPUS_QUALITY[1]["avg_len"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, pdf_vals, width, label='PDF-extracted', 
                   color=colors_qual[0], edgecolor=COLORS['border'], linewidth=2)
    bars2 = ax.bar(x + width/2, clean_vals, width, label='Clean JSON',
                   color=colors_qual[1], edgecolor=COLORS['border'], linewidth=2)
    
    ax.set_ylabel('Count', fontsize=11, fontweight='600')
    ax.set_title('Corpus Statistics', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 10, f'{int(height)}',
                    ha='center', fontsize=9, fontweight='600')
    
    plt.tight_layout()
    plt.savefig('output/05_corpus_quality_impact.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 05_corpus_quality_impact.png")



def create_infrastructure_diagram():
    """Technical infrastructure and deployment architecture"""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(8, 9.5, 'Technical Infrastructure & Deployment Architecture',
            ha='center', fontsize=18, fontweight='bold', color=COLORS['primary'])
    
    # Layer 1: User Interface
    y = 8.5
    ax.text(8, y+0.5, 'USER INTERFACE LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    ui_boxes = [
        {'x': 2, 'text': 'Streamlit Web UI\n5 Tabs\nReal-time Analytics', 'color': '#EBF8FF'},
        {'x': 6, 'text': 'CLI Interface\nmain.py\nBatch Evaluation', 'color': '#F0FFF4'},
        {'x': 10, 'text': 'REST API\n(Future)\nJSON Responses', 'color': '#FFFBEB'}
    ]
    
    for box in ui_boxes:
        rect = FancyBboxPatch((box['x'], y-0.7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor=box['color'], edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.5, y-0.2, box['text'], ha='center', va='center',
                fontsize=9, fontweight='500')
    
    # Layer 2: Application Layer
    y = 6.5
    ax.text(8, y+0.5, 'APPLICATION LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    app_boxes = [
        {'x': 1, 'text': 'Query Processing\nNormalization\nTokenization', 'color': '#F7F8FA'},
        {'x': 4.5, 'text': 'Retrieval Engine\n4 Methods\nScore Fusion', 'color': '#F7F8FA'},
        {'x': 8, 'text': 'Reranking\nMMR λ=0.9\nDiversity', 'color': '#F7F8FA'},
        {'x': 11.5, 'text': 'Generation\nOllama Client\nPrompt Templates', 'color': '#F7F8FA'}
    ]
    
    for box in app_boxes:
        rect = FancyBboxPatch((box['x'], y-0.7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor=box['color'], edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.5, y-0.2, box['text'], ha='center', va='center',
                fontsize=8, fontweight='500')
    
    # Layer 3: Data Layer
    y = 4.5
    ax.text(8, y+0.5, 'DATA & INDEX LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    data_boxes = [
        {'x': 0.5, 'text': 'FAISS Index\nvector_index.faiss\n633 × 384-dim', 'color': '#EBF8FF'},
        {'x': 4, 'text': 'BM25 Index\nbm25_index.pkl\nTokenized', 'color': '#F0FFF4'},
        {'x': 7.5, 'text': 'TF-IDF Index\ntfidf_index.pkl\nSparse Matrix', 'color': '#FFFBEB'},
        {'x': 11, 'text': 'Corpus JSON\n633 clauses\nIPC + BNS', 'color': '#FFF5F5'}
    ]
    
    for box in data_boxes:
        rect = FancyBboxPatch((box['x'], y-0.7), 3, 1, boxstyle="round,pad=0.1",
                              facecolor=box['color'], edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.5, y-0.2, box['text'], ha='center', va='center',
                fontsize=8, fontweight='500')

    
    # Layer 4: ML Models
    y = 2.5
    ax.text(8, y+0.5, 'MODEL LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    model_boxes = [
        {'x': 2, 'text': 'SentenceTransformers\nall-MiniLM-L6-v2\n384-dim embeddings', 'color': '#EBF8FF'},
        {'x': 7, 'text': 'Ollama Server\nGPT-OSS 120B\nLocal/Cloud', 'color': '#EBF4FF'},
        {'x': 12, 'text': 'Query Rewriter\nLLM-based\nTerminology Bridge', 'color': '#F0FFF4'}
    ]
    
    for box in model_boxes:
        rect = FancyBboxPatch((box['x'], y-0.7), 3.5, 1, boxstyle="round,pad=0.1",
                              facecolor=box['color'], edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.75, y-0.2, box['text'], ha='center', va='center',
                fontsize=8, fontweight='500')
    
    # Layer 5: Infrastructure
    y = 0.8
    ax.text(8, y+0.5, 'INFRASTRUCTURE LAYER', ha='center', fontsize=11,
            fontweight='bold', color=COLORS['primary'])
    
    infra_boxes = [
        {'x': 1, 'text': 'Python 3.13\nNumPy\nScikit-learn', 'color': '#F7F8FA'},
        {'x': 4.5, 'text': 'FAISS-CPU\nrank_bm25\nMatplotlib', 'color': '#F7F8FA'},
        {'x': 8, 'text': 'Streamlit\nWeb Server\nPort 8501', 'color': '#F7F8FA'},
        {'x': 11.5, 'text': 'File System\nJSON/PKL/NPY\nLocal Storage', 'color': '#F7F8FA'}
    ]
    
    for box in infra_boxes:
        rect = FancyBboxPatch((box['x'], y-0.6), 3, 0.8, boxstyle="round,pad=0.1",
                              facecolor=box['color'], edgecolor=COLORS['border'], linewidth=2)
        ax.add_patch(rect)
        ax.text(box['x']+1.5, y-0.2, box['text'], ha='center', va='center',
                fontsize=7, fontweight='500')
    
    plt.tight_layout()
    plt.savefig('output/06_infrastructure_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 06_infrastructure_diagram.png")


def create_key_findings_summary():
    """Summary infographic of key research findings"""
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Key Research Findings Summary',
            ha='center', fontsize=20, fontweight='bold', color=COLORS['primary'])
    ax.text(7, 9, 'Hybrid-MMR Legal RAG System • 132 Queries • 633 Clauses • IPC + BNS 2023',
            ha='center', fontsize=11, color=COLORS['secondary'])
    
    # Finding 1: Best Performance
    y = 7.8
    rect = FancyBboxPatch((0.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#EBF8FF', edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(rect)
    ax.text(3.5, y+0.9, '🏆 FINDING 1: Best Overall Performance', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(3.5, y+0.5, 'Vector-Only achieves highest Recall@5: 0.764', ha='center', fontsize=10)
    ax.text(3.5, y+0.2, 'Dense embeddings excel at paraphrase & conceptual queries', ha='center', fontsize=9)

    
    # Finding 2: BM25 for Exact Queries
    rect = FancyBboxPatch((7.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#F0FFF4', edgecolor=COLORS['success'], linewidth=3)
    ax.add_patch(rect)
    ax.text(10.5, y+0.9, '⚡ FINDING 2: BM25 Speed & Precision', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(10.5, y+0.5, 'BM25 achieves 1.1ms latency (15× faster than Hybrid)', ha='center', fontsize=10)
    ax.text(10.5, y+0.2, 'Best MRR (0.598) for exact legal terminology queries', ha='center', fontsize=9)
    
    # Finding 3: Corpus Quality (CENTRAL)
    y = 6
    rect = FancyBboxPatch((1, y), 12, 1.5, boxstyle="round,pad=0.15",
                          facecolor='#FFFBEB', edgecolor=COLORS['warning'], linewidth=4)
    ax.add_patch(rect)
    ax.text(7, y+1.15, '⭐ CENTRAL FINDING: Corpus Quality Dominates Algorithm Choice', ha='center',
            fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.text(7, y+0.75, 'Clean JSON corpus: +22.2% Recall@5, +26.9% MRR vs PDF-extracted', ha='center',
            fontsize=11, fontweight='600', color=COLORS['warning'])
    ax.text(7, y+0.45, 'Data quality improvement exceeds any algorithmic optimization tested', ha='center', fontsize=10)
    ax.text(7, y+0.15, 'PDF: 455 clauses, 87 chars avg → JSON: 575 clauses, 312 chars avg', ha='center', fontsize=9)
    
    # Finding 4: Query Category Insights
    y = 4.2
    rect = FancyBboxPatch((0.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#EBF4FF', edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(rect)
    ax.text(3.5, y+0.9, '📊 FINDING 4: Query-Specific Strengths', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(3.5, y+0.5, 'Exact: BM25 wins (R@5=0.818)', ha='center', fontsize=10)
    ax.text(3.5, y+0.2, 'Paraphrase/Conceptual: Vector wins (R@5=0.773, 0.750)', ha='center', fontsize=9)
    
    # Finding 5: Dual Corpus
    rect = FancyBboxPatch((7.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#F0FFF4', edgecolor=COLORS['success'], linewidth=3)
    ax.add_patch(rect)
    ax.text(10.5, y+0.9, '🔗 FINDING 5: Dual-Corpus System', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(10.5, y+0.5, 'First system spanning IPC (1860) + BNS (2023)', ha='center', fontsize=10)
    ax.text(10.5, y+0.2, '130+ section cross-references with provenance tagging', ha='center', fontsize=9)
    
    # Finding 6: LLM Query Rewriting
    y = 2.4
    rect = FancyBboxPatch((0.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#FFF5F5', edgecolor='#E53E3E', linewidth=3)
    ax.add_patch(rect)
    ax.text(3.5, y+0.9, '🔄 FINDING 6: Query Rewriting Impact', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(3.5, y+0.5, 'GPT-OSS 120B bridges vocabulary gap', ha='center', fontsize=10)
    ax.text(3.5, y+0.2, 'Informal → Legal terminology before retrieval', ha='center', fontsize=9)
    
    # Finding 7: MMR Tuning
    rect = FancyBboxPatch((7.5, y), 6, 1.2, boxstyle="round,pad=0.15",
                          facecolor='#EBF4FF', edgecolor=COLORS['accent'], linewidth=3)
    ax.add_patch(rect)
    ax.text(10.5, y+0.9, '🎯 FINDING 7: MMR Parameter Tuning', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(10.5, y+0.5, 'λ=0.9 optimal (relevance-focused)', ha='center', fontsize=10)
    ax.text(10.5, y+0.2, 'Corpus size (633) requires mild diversity penalty', ha='center', fontsize=9)
    
    # Statistics Box
    y = 0.5
    stats_text = (
        'System Statistics: 633 clauses • 132 evaluation queries • 4 retrieval systems\n'
        'Best Recall@5: 0.764 (Vector) • Best MRR: 0.598 (BM25) • Fastest: 1.1ms (BM25)\n'
        'Technology Stack: Python 3.13 • FAISS • SentenceTransformers • Ollama • Streamlit'
    )
    ax.text(7, y, stats_text, ha='center', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F8FA', edgecolor=COLORS['border'], linewidth=2))
    
    plt.tight_layout()
    plt.savefig('output/07_key_findings_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Generated: 07_key_findings_summary.png")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Generating Legal RAG Research Diagrams & Charts")
    print("="*60 + "\n")
    
    create_system_architecture()
    create_retrieval_pipeline_flow()
    create_performance_comparison()
    create_query_category_analysis()
    create_corpus_quality_impact()
    create_infrastructure_diagram()
    create_key_findings_summary()
    
    print("\n" + "="*60)
    print("✅ All diagrams generated successfully!")
    print("📁 Output directory: ./output/")
    print("="*60 + "\n")
