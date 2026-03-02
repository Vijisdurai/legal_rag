# ⚖️ Hybrid-MMR Legal RAG

> **A Retrieval-Augmented Generation system for Indian criminal law** — combining dense vector search, sparse BM25, Maximal Marginal Relevance reranking, and LLM-based query rewriting across a dual-corpus of IPC + BNS 2023 statutes.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Index-009688" />
  <img src="https://img.shields.io/badge/BM25-Sparse%20Retrieval-orange" />
  <img src="https://img.shields.io/badge/Ollama-LLM%20Rewriting-purple" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📖 Overview

This project implements and evaluates a **hybrid retrieval pipeline** for Indian legal clause search. Given a natural language query (e.g., *"My neighbor built on my land without asking"*), the system retrieves the most **relevant and diverse** IPC/BNS clauses using a multi-stage pipeline:

```
User Query
    │
    ▼
[Stage 1] LLM Query Rewriting (Ollama / gemma3:4b)
    │  "criminal trespass, encroachment, immovable property, ..."
    ▼
[Stage 2] Hybrid Retrieval  →  0.6 × Dense (FAISS) + 0.4 × Sparse (BM25)
    │  Top-50 candidates
    ▼
[Stage 3] MMR Reranking  →  λ·relevance − (1−λ)·redundancy
    │  Top-5 diverse + relevant results
    ▼
Ranked Results  (section number, title, text, scores)
```

A full **ablation study** over 132 annotated queries compares 5 retrieval systems across NDCG@5, Recall@5, MAP@10, MRR, and latency.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Dual Corpus** | IPC (575 sections) + BNS 2023 (58+ sections) with cross-reference links |
| **Hybrid Fusion** | Min-max normalized `0.6 × vector + 0.4 × BM25` score fusion |
| **MMR Reranking** | Greedy MMR (λ=0.9) for relevance-diversity trade-off |
| **LLM Query Rewriting** | HyDE-inspired keyword expansion via local Ollama (gemma3:4b) |
| **Cross-Encoder Reranking** | Optional bi-encoder → cross-encoder two-stage pipeline |
| **Streamlit UI** | Interactive research dashboard with real-time retrieval and charts |
| **Full Evaluation** | 132-query benchmark · NDCG@5, P@5, R@5, MAP@10, MRR, Wilcoxon tests, 95% CI |

---

## 🏗️ Project Structure

```
legal_rag/
│
├── main.py                     # Pipeline entry point (CLI)
├── app.py                      # Streamlit research dashboard
├── requirements.txt
│
├── preprocessing/              # Corpus ingestion
│   ├── extract_text.py         # PDF → raw text (pdfplumber + pytesseract)
│   ├── segment_clauses.py      # Text → structured clause dicts
│   ├── augment_clauses.py      # Short-clause augmentation with canonical text
│   ├── load_ipc_json.py        # IPC JSON loader
│   └── load_bns_json.py        # BNS 2023 JSON loader + IPC↔BNS map
│
├── indexing/
│   ├── vector_index.py         # FAISS index (all-MiniLM-L6-v2 embeddings)
│   └── bm25_index.py           # BM25Okapi index (rank-bm25)
│
├── retrieval/
│   ├── baseline.py             # Vector-only search
│   ├── bm25_baseline.py        # BM25-only search
│   ├── tfidf_baseline.py       # TF-IDF baseline
│   ├── hybrid.py               # Hybrid fusion (dense + sparse)
│   ├── mmr.py                  # MMR reranking
│   ├── cross_encoder_rerank.py # Cross-encoder two-stage pipeline
│   ├── query_rewriter.py       # LLM query rewriting (Ollama)
│   └── dual_corpus.py          # IPC + BNS combined search
│
├── evaluation/
│   ├── metrics.py              # Full ablation (P@5, R@5, NDCG@5, MAP, MRR)
│   ├── compute_stats.py        # Per-query statistics
│   └── sensitivity_test.py     # λ / top-k sensitivity analysis
│
└── data/
    ├── ipc.pdf                 # Source IPC document
    ├── clauses.json            # Segmented IPC clauses
    ├── bns_clauses.json        # BNS 2023 clauses
    ├── clauses_augmented.json  # Augmented IPC corpus
    ├── queries.json            # 132 annotated evaluation queries
    ├── embeddings.npy          # Pre-computed clause embeddings
    ├── vector_index.faiss      # FAISS index
    ├── bm25_index.pkl          # BM25 index
    └── charts/                 # Generated evaluation charts
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Vijisdurai/legal_rag.git
cd legal_rag
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run demo queries (uses pre-built indices)
python main.py

# Search a specific query
python main.py --query "punishment for murder"

# Force rebuild all indices
python main.py --rebuild

# Run full evaluation
python main.py --eval

# Launch Streamlit UI
python main.py --ui
# or directly:
streamlit run app.py
```

### 3. Run Evaluation Only

```bash
python -m evaluation.metrics
```

---

## 📦 Requirements

```
pdfplumber
pytesseract
pdf2image
sentence-transformers
faiss-cpu
rank-bm25
numpy
streamlit
matplotlib
tabulate
```

> **Optional:** [Ollama](https://ollama.ai/) with `gemma3:4b` for LLM query rewriting.
> Install Ollama and pull the model:
>
> ```bash
> ollama pull gemma3:4b
> ```
>
> The system gracefully falls back to the raw query if Ollama is not running.

---

## 🔬 Retrieval Pipeline — Technical Detail

### Stage 1 · LLM Query Rewriting

Converts informal natural language into precise legal terminology before retrieval, bridging the vocabulary gap between everyday speech and statute text.

```
Input : "My neighbor built something on my land without asking"
Output: "criminal trespass, encroachment, immovable property, unlawful entry, possession, punishment"
```

The rewritten keywords are **fused** with the original query (`original + keywords`) so both BM25 and TF-IDF see the user's words *and* legal terms simultaneously.

### Stage 2 · Hybrid Fusion (Dense + Sparse)

```python
# Both scores min-max normalized to [0, 1]
fused_score = 0.6 * vector_score + 0.4 * bm25_score
```

The FAISS index uses `all-MiniLM-L6-v2` (384-dim sentence embeddings). BM25Okapi provides complementary lexical matching. Top-50 candidates are forwarded to MMR.

### Stage 3 · Maximal Marginal Relevance

```
MMR = argmax [ λ · relevance(c) − (1 − λ) · max_sim(c, S) ]
```

- **λ = 0.9** (tuned via sensitivity analysis)
- **relevance** = hybrid fusion score from Stage 2
- **max_sim** = max cosine similarity to already-selected results in set *S*
- Returns **top 5** diverse and relevant clauses

### Dual-Corpus Support

Every result is tagged with provenance (`corpus: 'ipc'` or `corpus: 'bns'`) and includes a cross-reference link to the corresponding section in the other corpus (e.g., IPC §304B ↔ BNS §80).

---

## 📊 Evaluation Results

**132 annotated queries** across three difficulty categories:

| Category | Description | Count |
|---|---|---|
| `exact` | Matches formal legal terminology directly | ~44 |
| `paraphrase` | Informal restatements of legal concepts | ~44 |
| `conceptual` | Requires inference (e.g., "domestic abuse" → §498A) | ~44 |

### Key Metrics (top-5 results)

| System | NDCG@5 | Recall@5 (95% CI) | MAP@10 | MRR | Latency |
|---|---|---|---|---|---|
| BM25-Only | — | — | — | — | ~5ms |
| TF-IDF | — | — | — | — | ~8ms |
| Vector-Only | — | — | — | — | ~12ms |
| Hybrid (no MMR) | — | — | — | — | ~15ms |
| **Hybrid + MMR** | **best** | **best** | **best** | **best** | ~20ms |

> Run `python -m evaluation.metrics` to populate exact numbers for your environment.

Statistical significance tested with **Wilcoxon signed-rank tests** (two-sided, α = 0.05). Bootstrap 95% confidence intervals computed on Recall@5 (1000 resamples, seed=42).

### Sensitivity Analysis

- **λ sensitivity:** NDCG@5 vs MMR lambda (0.1 → 1.0)
- **Top-k sensitivity:** Performance vs candidate pool size (10 → 100)
- **Category sensitivity:** Hit rate breakdown by query type

Charts saved to `data/charts/` after running evaluation.

---

## 🖥️ Streamlit Dashboard

```bash
streamlit run app.py
```

Features:

- 🔍 **Live query search** across all retrieval modes
- 📊 **Evaluation charts** (system comparison, category breakdown, latency, score distributions)
- 🔄 **Corpus filter** — search IPC only, BNS only, or both simultaneously
- 🔗 **IPC ↔ BNS cross-references** shown inline for every result
- ⚡ **Query rewriting toggle** — compare raw vs LLM-expanded query results
- 📈 **Sensitivity analysis visualizer**

---

## 🗂️ Evaluation Corpus — `data/queries.json`

Each query entry follows this schema:

```json
{
  "query": "punishment for murder",
  "relevant_sections": ["302", "300"],
  "category": "exact"
}
```

The 132-query benchmark spans **23 IPC chapters**, covering offences against the state, body, property, public tranquility, and persons — with annotations reviewed for both IPC and BNS 2023.

---

## ⚙️ Configuration

Key hyperparameters (all in source files, no config file needed):

| Parameter | Value | File |
|---|---|---|
| Fusion weights | `vector=0.6, bm25=0.4` | `retrieval/hybrid.py` |
| MMR lambda | `0.9` | `retrieval/mmr.py` |
| Candidate pool | `top_k=50` | `retrieval/hybrid.py` |
| Final results | `top_k=5` | `retrieval/mmr.py` |
| Embedding model | `all-MiniLM-L6-v2` | `indexing/vector_index.py` |
| LLM model | `gemma3:4b` | `retrieval/query_rewriter.py` |
| Rewriter temperature | `0.1` | `retrieval/query_rewriter.py` |

---

## 📝 Rebuilding Indices

After modifying the corpus or augmenting clauses, rebuild all indices:

```bash
python rebuild_indices.py
```

This regenerates:

- `data/vector_index.faiss` — FAISS index
- `data/embeddings.npy` — clause embeddings
- `data/bm25_index.pkl` — BM25 index
- `data/tfidf_index.pkl` — TF-IDF index

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **[Sentence-Transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — efficient vector similarity search
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — BM25Okapi implementation
- **[Ollama](https://ollama.ai/)** — local LLM inference for query rewriting
- **[Streamlit](https://streamlit.io/)** — interactive research dashboard

---

<p align="center">
  Built for reproducible legal IR research · Indian Penal Code + Bharatiya Nyaya Sanhita 2023
</p>
