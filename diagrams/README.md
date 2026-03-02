# Legal RAG Research: Infrastructure, Architecture & Key Findings

## Overview

This directory contains comprehensive visualizations of the **Hybrid-MMR Legal RAG System** research, covering system architecture, infrastructure design, and key empirical findings from the 132-query evaluation study.

## Generated Diagrams

### 1. System Architecture (`01_system_architecture.png`)

**Purpose:** End-to-end system architecture showing all layers from data sources to answer generation.

**Key Components:**
- **Data Layer:** IPC PDF (511 sections), BNS Gazette (358 sections), IPC-BNS mapping (130+ pairs), evaluation queries (132), ground truth annotations
- **Preprocessing Layer:** PDF extraction (pdfplumber, Tesseract OCR), text cleaning, segmentation (575 IPC clauses), BNS curation (58 sections), corpus merge (633 total)
- **Indexing Layer:** 
  - FAISS Vector Index (all-MiniLM-L6-v2, 384-dim, IndexFlatIP)
  - BM25 Index (BM25Okapi, tokenized corpus)
  - TF-IDF Index (scikit-learn, unigrams+bigrams)
- **Retrieval Pipeline:** BM25 (1.1ms), TF-IDF (2.0ms), Dense Vector (11.8ms), Hybrid Fusion (0.6×Vec + 0.4×BM25), MMR Rerank (λ=0.9)
- **Generation Layer:** LLM Answer Generation (GPT-OSS 120B via Ollama), grounded in retrieved sections, citation-based, hallucination guard

**Use Case:** Understanding the complete system flow from raw legal documents to grounded AI answers.

---

### 2. Retrieval Pipeline Flow (`02_retrieval_pipeline_flow.png`)

**Purpose:** Detailed flowchart of the hybrid retrieval pipeline with timing and scoring details.

**Pipeline Steps:**
1. **User Query** (Natural Language)
2. **Parallel Retrieval** (4 methods run simultaneously)
   - BM25: Token matching, TF-IDF weighting (1.1ms)
   - TF-IDF: Statistical, cosine similarity (2.0ms)
   - Dense Vector: Semantic embedding, FAISS search (11.8ms)
   - Hybrid: Fusion 0.6×Vec + 0.4×BM25
3. **Score Normalization** (Min-Max) → Hybrid = 0.6 × norm(vector) + 0.4 × norm(BM25) → Top-50 candidates
4. **Corpus Filtering:** IPC / BNS / Both
5. **MMR Reranking** (λ=0.9): MMR(c) = λ·Relevance(c) − (1−λ)·max Similarity(c, selected)
6. **Top-5 Ranked Legal Clauses**

**Use Case:** Technical implementation reference for developers building similar legal retrieval systems.

---

### 3. Performance Comparison (`03_performance_comparison.png`)

**Purpose:** Comprehensive 6-chart comparison of all 4 retrieval systems across key metrics.

**Charts:**
1. **Recall@5:** Vector-Only wins (0.764), BM25 second (0.689)
2. **NDCG@5:** Vector-Only wins (0.683), BM25 second (0.589)
3. **MRR:** Vector-Only wins (0.705), BM25 second (0.598)
4. **Latency:** BM25 fastest (1.1ms), Hybrid+MMR slowest (16.6ms)
5. **Accuracy vs Speed Scatter:** Shows trade-off between NDCG@5 and latency
6. **% Change vs Vector-Only:** BM25 (-13.7%), TF-IDF (-17.7%), Hybrid+MMR (-32.4%)

**Key Insight:** Vector-Only achieves best aggregate performance, but BM25 offers 15× speed advantage with competitive accuracy.

**Use Case:** System selection guidance based on accuracy vs latency requirements.

---

### 4. Query Category Analysis (`04_query_category_analysis.png`)

**Purpose:** Performance breakdown by query difficulty: Exact vs Paraphrase vs Conceptual (44 queries each).

**Key Findings:**
- **Exact Terminology Queries:** BM25 wins (R@5=0.818, MRR=0.742) — perfect token matching
- **Paraphrase Queries:** Vector-Only wins (R@5=0.773, MRR=0.718) — semantic similarity bridges vocabulary gap
- **Conceptual Queries:** Vector-Only wins (R@5=0.750, MRR=0.717) — embedding space captures abstract legal concepts

**Use Case:** Choosing retrieval method based on expected query type distribution.

---

### 5. Corpus Quality Impact (`05_corpus_quality_impact.png`)

**Purpose:** Demonstrates the central research finding — corpus quality dominates algorithm choice.

**Comparison:**
- **PDF-Extracted:** 455 clauses, 87 chars avg → R@5=0.625, MRR=0.556
- **Clean JSON:** 575 clauses, 312 chars avg → R@5=0.764, MRR=0.705

**Improvements:**
- **Recall@5:** +22.2% (0.625 → 0.764)
- **MRR:** +26.9% (0.556 → 0.705)

**Key Insight:** Data quality improvement exceeds any algorithmic optimization tested. Invest in corpus curation before algorithm tuning.

**Use Case:** Justifying investment in data cleaning and curation for legal NLP projects.

---

### 6. Infrastructure Diagram (`06_infrastructure_diagram.png`)

**Purpose:** Technical stack and deployment architecture.

**Layers:**
1. **User Interface:** Streamlit Web UI (5 tabs, real-time analytics), CLI Interface (main.py, batch evaluation), REST API (future)
2. **Application Layer:** Query processing, retrieval engine (4 methods, score fusion), reranking (MMR λ=0.9), generation (Ollama client)
3. **Data & Index Layer:** FAISS index (vector_index.faiss, 633×384-dim), BM25 index (bm25_index.pkl), TF-IDF index (tfidf_index.pkl), Corpus JSON (633 clauses, IPC+BNS)
4. **Model Layer:** SentenceTransformers (all-MiniLM-L6-v2, 384-dim), Ollama Server (GPT-OSS 120B), Query Rewriter (LLM-based)
5. **Infrastructure Layer:** Python 3.13, NumPy, scikit-learn, FAISS-CPU, rank_bm25, Matplotlib, Streamlit, File System (JSON/PKL/NPY)

**Use Case:** Deployment planning and technology stack reference.

---

### 7. Key Findings Summary (`07_key_findings_summary.png`)

**Purpose:** One-page infographic summarizing all 7 major research findings.

**Findings:**
1. **🏆 Best Overall Performance:** Vector-Only achieves highest Recall@5 (0.764), excels at paraphrase & conceptual queries
2. **⚡ BM25 Speed & Precision:** 1.1ms latency (15× faster than Hybrid), best MRR (0.598) for exact legal terminology
3. **⭐ CENTRAL FINDING:** Corpus quality dominates algorithm choice — Clean JSON: +22.2% R@5, +26.9% MRR vs PDF-extracted
4. **📊 Query-Specific Strengths:** Exact: BM25 wins (R@5=0.818) | Paraphrase/Conceptual: Vector wins (R@5=0.773, 0.750)
5. **🔗 Dual-Corpus System:** First system spanning IPC (1860) + BNS (2023), 130+ section cross-references with provenance tagging
6. **🔄 Query Rewriting Impact:** GPT-OSS 120B bridges vocabulary gap, informal → legal terminology before retrieval
7. **🎯 MMR Parameter Tuning:** λ=0.9 optimal (relevance-focused), corpus size (633) requires mild diversity penalty

**Statistics:**
- 633 clauses • 132 evaluation queries • 4 retrieval systems
- Best Recall@5: 0.764 (Vector) • Best MRR: 0.598 (BM25) • Fastest: 1.1ms (BM25)
- Technology Stack: Python 3.13 • FAISS • SentenceTransformers • Ollama • Streamlit

**Use Case:** Executive summary for stakeholders, conference presentation slide.

---

## How to Generate Diagrams

```bash
cd legal_rag/diagrams
python generate_diagrams.py
```

All diagrams will be saved to `./output/` directory as high-resolution PNG files (300 DPI).

---

## Research Context

This visualization suite accompanies the research paper:

**"Hybrid Retrieval-Augmented Generation for Indian Legal Clause Search: Combining BM25, Dense Embeddings, and MMR Reranking Across IPC and BNS 2023"**

- **Corpus:** 633 legal clauses (575 IPC + 58 BNS)
- **Evaluation:** 132 annotated queries (44 exact, 44 paraphrase, 44 conceptual)
- **Systems Compared:** BM25-Only, TF-IDF, Vector-Only, Hybrid+MMR
- **Metrics:** Precision@5, Recall@5, NDCG@5, MAP@10, MRR, Latency

---

## Key Contributions

1. **Dual-corpus hybrid retrieval** (IPC + BNS 2023) — first documented system spanning both frameworks
2. **Corpus quality as primary lever** — quantified +22% Recall improvement from clean JSON vs PDF
3. **Correct post-filter architecture** for multi-corpus FAISS
4. **130+ IPC–BNS cross-reference map** — first structured machine-readable mapping
5. **4-system ablation on 132 legal queries** — largest structured ablation study on Indian clause-level retrieval
6. **Grounded LLM answer generation** — retrieve-then-generate pipeline with hallucination guard
7. **Research-grade UI** with real-time analytics — 4 layout modes, 4 retrieval systems, live charts
8. **LLM-based query rewriting** — first application of LLM query expansion to Indian statutory retrieval

---

## Citation

If you use these diagrams or findings in your research, please cite:

```bibtex
@article{legal_rag_2026,
  title={Hybrid Retrieval-Augmented Generation for Indian Legal Clause Search},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2026}
}
```

---

## License

Research diagrams and documentation: CC BY 4.0  
Code: MIT License

---

## Contact

For questions about the research or diagrams:
- GitHub Issues: [Repository URL]
- Email: [Contact Email]

---

**Last Updated:** February 2026  
**Version:** 1.0
