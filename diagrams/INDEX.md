# Legal RAG Research Diagrams - Quick Reference Index

## 📁 Generated Files

All diagrams are in `output/` directory at 300 DPI resolution.

### Architecture & Infrastructure

| File | Title | Purpose | Key Content |
|---|---|---|---|
| `01_system_architecture.png` | System Architecture | End-to-end system overview | 5 layers: Data → Preprocessing → Indexing → Retrieval → Generation |
| `02_retrieval_pipeline_flow.png` | Retrieval Pipeline Flow | Detailed retrieval flowchart | 6-step pipeline with timing and scoring formulas |
| `06_infrastructure_diagram.png` | Infrastructure Diagram | Technical stack & deployment | 5 layers: UI → Application → Data → Model → Infrastructure |

### Performance & Findings

| File | Title | Purpose | Key Content |
|---|---|---|---|
| `03_performance_comparison.png` | Performance Comparison | 4-system metric comparison | 6 charts: R@5, NDCG@5, MRR, Latency, Accuracy vs Speed, % Change |
| `04_query_category_analysis.png` | Query Category Analysis | Performance by query type | Exact vs Paraphrase vs Conceptual (44 queries each) |
| `05_corpus_quality_impact.png` | Corpus Quality Impact | Central research finding | +22% R@5, +27% MRR from data cleaning |
| `07_key_findings_summary.png` | Key Findings Summary | One-page research overview | All 7 major findings + statistics |

---

## 🎯 Quick Selection Guide

**Need to understand the system?**
→ Start with `01_system_architecture.png`

**Implementing the retrieval pipeline?**
→ Use `02_retrieval_pipeline_flow.png`

**Choosing a retrieval method?**
→ Check `03_performance_comparison.png` and `04_query_category_analysis.png`

**Justifying data curation investment?**
→ Show `05_corpus_quality_impact.png`

**Planning deployment?**
→ Reference `06_infrastructure_diagram.png`

**Presenting to stakeholders?**
→ Use `07_key_findings_summary.png`

---

## 📊 Key Metrics at a Glance

### Best Overall Performance
- **System:** Vector-Only
- **Recall@5:** 0.764
- **NDCG@5:** 0.683
- **MRR:** 0.705

### Fastest System
- **System:** BM25-Only
- **Latency:** 1.1ms (15× faster than Hybrid+MMR)
- **MRR:** 0.598 (competitive accuracy)

### Central Finding
- **Corpus Quality Impact:** +22.2% Recall@5, +26.9% MRR
- **Improvement:** PDF-extracted (455 clauses, 87 chars) → Clean JSON (575 clauses, 312 chars)

### Query-Specific Winners
- **Exact Terminology:** BM25-Only (R@5=0.818)
- **Paraphrase:** Vector-Only (R@5=0.773)
- **Conceptual:** Vector-Only (R@5=0.750)

---

## 🔍 Diagram Details

### 01_system_architecture.png
**Dimensions:** 16×12 inches (4800×3600 px @ 300 DPI)  
**Layers:**
1. Data Layer: IPC PDF, BNS Gazette, IPC-BNS mapping, queries, ground truth
2. Preprocessing: PDF extraction, cleaning, segmentation, curation, merge
3. Indexing: FAISS (384-dim), BM25 (tokenized), TF-IDF (unigrams+bigrams)
4. Retrieval: BM25 (1.1ms), TF-IDF (2.0ms), Vector (11.8ms), Hybrid, MMR
5. Generation: GPT-OSS 120B, grounded, citation-based

### 02_retrieval_pipeline_flow.png
**Dimensions:** 14×10 inches (4200×3000 px @ 300 DPI)  
**Steps:**
1. User Query (Natural Language)
2. Parallel Retrieval (4 methods)
3. Score Normalization (Min-Max) → Hybrid = 0.6×Vec + 0.4×BM25
4. Corpus Filtering (IPC/BNS/Both)
5. MMR Reranking (λ=0.9)
6. Top-5 Results

### 03_performance_comparison.png
**Dimensions:** 16×10 inches (4800×3000 px @ 300 DPI)  
**Charts:**
1. Recall@5 bar chart (Vector wins: 0.764)
2. NDCG@5 bar chart (Vector wins: 0.683)
3. MRR bar chart (Vector wins: 0.705)
4. Latency bar chart (BM25 wins: 1.1ms)
5. Accuracy vs Speed scatter plot
6. % Change vs Vector-Only horizontal bar chart

### 04_query_category_analysis.png
**Dimensions:** 14×6 inches (4200×1800 px @ 300 DPI)  
**Charts:**
1. Recall@5 by Category (3 categories × 4 systems)
2. MRR by Category (3 categories × 4 systems)

**Categories:**
- Exact: BM25 wins (R@5=0.818, MRR=0.742)
- Paraphrase: Vector wins (R@5=0.773, MRR=0.718)
- Conceptual: Vector wins (R@5=0.750, MRR=0.717)

### 05_corpus_quality_impact.png
**Dimensions:** 15×5 inches (4500×1500 px @ 300 DPI)  
**Charts:**
1. Recall@5 Improvement (0.625 → 0.764, +22.2%)
2. MRR Improvement (0.556 → 0.705, +26.9%)
3. Corpus Statistics (clauses, avg length)

### 06_infrastructure_diagram.png
**Dimensions:** 16×10 inches (4800×3000 px @ 300 DPI)  
**Layers:**
1. User Interface: Streamlit (5 tabs), CLI, REST API (future)
2. Application: Query processing, retrieval engine, reranking, generation
3. Data & Index: FAISS, BM25, TF-IDF, Corpus JSON
4. Model: SentenceTransformers, Ollama, Query Rewriter
5. Infrastructure: Python 3.13, NumPy, scikit-learn, FAISS-CPU, Streamlit

### 07_key_findings_summary.png
**Dimensions:** 14×10 inches (4200×3000 px @ 300 DPI)  
**Findings:**
1. Best Overall Performance (Vector-Only: R@5=0.764)
2. BM25 Speed & Precision (1.1ms, MRR=0.598)
3. CENTRAL: Corpus Quality Dominates (+22% R@5, +27% MRR)
4. Query-Specific Strengths (Exact: BM25, Paraphrase/Conceptual: Vector)
5. Dual-Corpus System (IPC + BNS, 130+ mappings)
6. Query Rewriting Impact (GPT-OSS 120B vocabulary bridge)
7. MMR Parameter Tuning (λ=0.9 optimal)

**Statistics Box:**
- 633 clauses • 132 queries • 4 systems
- Best R@5: 0.764 (Vector) • Best MRR: 0.598 (BM25) • Fastest: 1.1ms (BM25)
- Stack: Python 3.13 • FAISS • SentenceTransformers • Ollama • Streamlit

---

## 📝 Usage Examples

### For Research Papers
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=\textwidth]{03_performance_comparison.png}
  \caption{Performance comparison across 4 retrieval systems on 132 legal queries.
           Vector-Only achieves best Recall@5 (0.764) while BM25-Only offers 
           15× speed advantage (1.1ms latency).}
  \label{fig:performance}
\end{figure}
```

### For Presentations
- **Title Slide:** Use `07_key_findings_summary.png`
- **System Overview:** Use `01_system_architecture.png`
- **Results:** Use `03_performance_comparison.png`
- **Key Finding:** Use `05_corpus_quality_impact.png`

### For Documentation
- **README.md:** Embed `01_system_architecture.png` for system overview
- **Technical Docs:** Reference `02_retrieval_pipeline_flow.png` for implementation
- **Deployment Guide:** Use `06_infrastructure_diagram.png` for stack details

---

## 🔄 Regenerating Diagrams

```bash
cd legal_rag/diagrams
python generate_diagrams.py
```

All diagrams will be regenerated in `output/` directory.

**Requirements:**
- Python 3.13+
- matplotlib
- numpy

**Customization:**
Edit `generate_diagrams.py` to modify:
- Colors (COLORS dict)
- Data (EVAL_DATA, CATEGORY_DATA, CORPUS_QUALITY)
- Layout (figure sizes, positions)

---

## 📚 Related Documents

- **README.md** — Detailed diagram documentation
- **DIAGRAMS_AND_FINDINGS_SUMMARY.md** — Complete research summary
- **../RESEARCH_PAPER_DRAFT.md** — Full research paper
- **generate_diagrams.py** — Diagram generation script

---

## 📞 Support

For questions or issues:
1. Check README.md for detailed explanations
2. Review DIAGRAMS_AND_FINDINGS_SUMMARY.md for context
3. Open GitHub issue for technical problems
4. Email [contact] for research inquiries

---

**Last Updated:** February 28, 2026  
**Version:** 1.0  
**Total Diagrams:** 7  
**Total Resolution:** 300 DPI  
**Total File Size:** ~15 MB
