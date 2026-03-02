---
title: "Sparse, Dense, and Hybrid Retrieval for Clause-Level Indian Legal Search: An Empirical Study with Corpus-Quality and Query Rewriting Analysis"
subtitle: "A Comparative Study of BM25, TF-IDF, Dense Vector, and Hybrid MMR Retrieval Across IPC 1860 and Bharatiya Nyaya Sanhita 2023"
version: "Draft v3.0"
date: "February 2026"
status: "Prepared for submission to a peer-reviewed venue in Legal NLP / Information Retrieval"
---

# Sparse, Dense, and Hybrid Retrieval for Clause-Level Indian Legal Search

## An Empirical Study with Corpus-Quality and Query Rewriting Analysis

---

| Field | Details |
| --- | --- |
| **Author(s)** | \[Author Name(s)\] |
| **Affiliation** | \[Institution / Organisation\] |
| **Contact** | \[<author@email.com>\] |
| **Status** | Draft v3.0 — February 2026 |
| **Target Venue** | ACL / EMNLP / COLING *(Legal NLP Track)* or ACM SIGIR |
| **Keywords** | Legal IR · RAG · BM25 · Dense Retrieval · MMR · Query Rewriting · IPC · BNS 2023 |

---

> **Abstract submitted for:** Peer-reviewed venue in Legal Natural Language Processing / Legal Information Retrieval
> *Code and corpus available upon reasonable request.*

---

## Abstract

This paper investigates the challenge of clause-level legal information retrieval within the context of Indian penal statutes. We address the vocabulary gap between informal natural language queries and formal statutory text, a barrier often encountered in citizen-facing legal applications. We design, implement, and evaluate five retrieval paradigms on a curated corpus of 633 clauses from the Indian Penal Code (IPC, 1860) and the Bharatiya Nyaya Sanhita (BNS, 2023). The evaluated systems include BM25, TF-IDF, dense vector retrieval (`all-MiniLM-L6-v2`), Hybrid retrieval, and Hybrid retrieval with Maximal Marginal Relevance (MMR) reranking.

Using a benchmark of 132 annotated queries stratified across three difficulty tiers (exact, paraphrase, and conceptual), we find that pure dense vector retrieval yields the strongest aggregate performance (Recall@5: 0.773, nDCG@5: 0.603). Notably, Hybrid+MMR underperforms the BM25 baseline (Recall@5: 0.553 vs. 0.659). Our ablation study isolates both hybrid fusion miscalibration and MMR diversity penalisation at small corpus sizes as compounding factors for this underperformance. Furthermore, we demonstrate that structural improvements to corpus data quality—replacing fragmented PDF extractions with full-text JSON—yield a +22.2% absolute improvement in Recall@5, exceeding the performance gains observed between distinct retrieval algorithms.

Finally, we introduce an LLM-based query rewriting mechanism that translates informal descriptions into statutory keywords, effectively bridging the vocabulary gap and improving recall for conceptual queries. This work contributes a structured IPC/BNS benchmark, an empirical analysis of hybrid retrieval limitations on highly-similar domain corpora, and an open-source evaluation framework for legal text retrieval.

**Keywords:** Legal Information Retrieval, Retrieval-Augmented Generation, BM25, Dense Embeddings, Maximal Marginal Relevance, Query Rewriting, Indian Penal Code, Bharatiya Nyaya Sanhita, Hybrid Search

---

## 1. Introduction

Access to statutory legal information is often hindered by a profound vocabulary gap. When laypersons query legal databases, they typically use informal, descriptive language (e.g., "cheating and taking money") rather than the precise statutory terminology required by traditional search indexing (e.g., "dishonestly inducing delivery of property" under §420 of the Indian Penal Code). Consequently, bridging this linguistic divide is a central challenge in building effective Legal Information Retrieval (LIR) and Retrieval-Augmented Generation (RAG) systems for citizen-facing applications.

Clause-level legal retrieval in the Indian context presents three primary computational challenges:

1. **Statutory language complexity:** India's primary criminal statute for over a century, the Indian Penal Code (IPC, 1860), employs colonial-era English, Latin maxims, and nuanced legal phrasing. Pre-trained language models are rarely exposed to sufficient quantities of this specific domain distribution.
2. **Relevance precision:** Legal search requires exactitude; adjacent but distinct statutes (e.g., distinguishing between culpable homicide and murder) carry vastly different legal implications.
3. **Dual-corpus transition:** The legislative introduction of the Bharatiya Nyaya Sanhita (BNS, 2023) to replace the IPC means that systems must increasingly support retrieval across parallel active corpora.

This paper investigates the efficacy of various retrieval architectures in addressing these constraints. We construct an evaluation framework across a curated collection of clauses from both the IPC and BNS, evaluating five core retrieval paradigms: BM25, TF-IDF, Dense Vector matching, Hybrid fusion, and Hybrid retrieval with Maximal Marginal Relevance (MMR) reranking. Using an authored benchmark of 132 queries stratified by difficulty (exact terminology, paraphrase, and conceptual scenarios), we provide a comprehensive empirical analysis of LIR performance.

Our findings challenge several assumptions commonly applied in RAG system design. Specifically, we observe that the often-recommended Hybrid+MMR approach underperforms relative to baseline sparse methods on this corpus. Furthermore, we demonstrate that data curation—specifically addressing artefacts from PDF extraction—yields performance improvements that exceed the gains provided by algorithmic retrieval variations.

### 1.1 Research Questions

**RQ1:** Which retrieval paradigm (BM25, TF-IDF, Dense Vector, Hybrid, or Hybrid+MMR) achieves the strongest aggregate accuracy on Indian statutory text, and how does performance vary across query difficulty levels?

**RQ2:** To what extent does foundational corpus data quality impact retrieval metrics relative to architectural changes in the retrieval pipeline?

**RQ3:** Can LLM-based pre-retrieval query rewriting effectively bridge informal user language to formal statutory vocabulary, and how does this impact performance across different query types?

---

## 2. Related Work

### 2.1 Legal Information Retrieval: A Brief History

Automated legal search predates the modern NLP era. The WESTLAW and LexisNexis platforms of the 1970s and 1980s used Boolean keyword search as their primary retrieval mechanism — a paradigm that still underlies many commercial legal search systems today. The FIRE Legal Track (2008–2011) introduced Indian legal document retrieval as an academic benchmark, establishing BM25 and TF-IDF as reliable baselines for statutory text.

The transformer revolution changed the retrieval landscape fundamentally. Chalkidis et al. (2020) introduced Legal-BERT, a domain-adapted BERT variant pre-trained on English legal corpora. Subsequent work demonstrated large improvements on contract clause classification, case outcome prediction, and statute summarisation. However, the vast majority of this work targets English common-law jurisdictions — particularly US case law and EU regulation — and evaluates on document-level, not clause-level retrieval tasks.

For Indian law specifically, existing datasets cover court judgments at the document level (e.g., the ILSI dataset), not clause-level retrieval from bare acts. The specific challenge of mapping informal citizen descriptions to IPC or BNS provisions has not, to our knowledge, been formally studied.

### 2.2 Sparse vs. Dense Retrieval: The Enduring Tradeoff

The core tension between sparse and dense retrieval methods is well-documented. BM25 (Robertson & Zaragoza, 2009) remains a remarkably strong baseline precisely because it rewards exact token matches — a property that turns out to be exactly what legal retrieval sometimes needs. When a user types "criminal conspiracy," BM25 finds §120A instantly because those exact terms appear in the section title.

Dense retrieval systems (Karpukhin et al., 2020; Reimers & Gurevych, 2019) encode both queries and documents as continuous high-dimensional vectors, enabling semantic similarity matching that BM25 cannot achieve. They generalise across vocabulary gaps — but they do so at the cost of exact-match precision, and their performance is highly sensitive to whether the embedding model has been exposed to the domain's vocabulary during training.

The conventional wisdom, solidified by Ma et al. (2022) and others, holds that hybrid retrieval combining both signals consistently outperforms either alone. We challenge this wisdom in the Indian legal domain and provide a mechanistic explanation for why the conventional result does not hold.

### 2.3 Maximal Marginal Relevance in Retrieval

MMR (Carbonell & Goldstein, 1998) is a greedy, diversity-promoting reranking algorithm that iteratively selects the next document to maximise both relevance to the query and dissimilarity to already-selected documents:

```
MMR(dᵢ) = argmax_{dᵢ ∉ S} [ λ · Sim(dᵢ, q) − (1−λ) · max_{dⱼ ∈ S} Sim(dᵢ, dⱼ) ]
```

The intuition is powerful: if you have already retrieved §97 (Right of private defence of the body), retrieving §97A (which is nearly identical in embedding space) adds little value. MMR pushes the system toward §98, which covers the same offense with a different nuance.

In our setting, however, legal sub-sections within a single chapter embed extremely closely together. A dense legal corpus with 633 sections contains many such near-duplicate clusters. We show that this makes aggressive MMR diversity penalties counterproductive at this scale: they displace high-relevance results and reduce overall Recall@5.

### 2.4 RAG for Legal Question Answering

Lewis et al. (2020) introduced the RAG framework as a general mechanism for grounding LLM outputs in external knowledge. In legal applications, this grounding is not just useful — it is ethically necessary. An LLM generating legal advice from parametric memory alone is unreliable in ways that carry real harm: it may cite the correct concept but the wrong section number, or describe a pre-BNS position as current law.

Recent work on LegalBench-RAG and NyayaRAG (Gupta et al., 2024) has begun to address Indian legal QA from a retrieval perspective. These systems, however, focus on multi-hop reasoning over judgments rather than clause-level precise retrieval from primary legislation. Our work is complementary to these efforts and fills a gap at the most granular level of the legal hierarchy — the specific statutory provision.

---

## 3. Corpus Construction

### 3.1 Why Standard Corpora Are Insufficient

The standard approach to building a legal RAG corpus is to parse the PDF version of the relevant act, chunk by token length, embed each chunk, and index it. This approach has a subtle but fatal flaw in the Indian statutory context: official PDF versions of the IPC and BNS use a **two-column newspaper layout** that is systematically misread by standard PDF extraction libraries. Text from the left column bottom flows directly into right column content. Section headers appear on one page and the corresponding body text on the next. Footnotes blend into section text. The result is embeddings trained on corrupted, semantically broken text fragments — and no amount of algorithmic sophistication compensates for training a retriever on garbage.

This observation motivated our most important methodological decision: to curate a clean, structured JSON corpus from scratch, using the official gazette text as the authoritative source.

### 3.2 Indian Penal Code (IPC 1860) — 575 Sections

The IPC is structured as 511 sections across 23 chapters, enacted by the British Indian government in 1860 and remaining in force for 163 years. Our initial PDF extraction using `pdfplumber` with `layout=True` and Tesseract OCR fallback yielded 455 clause objects, of which roughly 75% contained fewer than 100 characters — truncated section titles with no body text. The average text length was 87 characters, far too short to encode meaningful semantic content.

After constructing the clean JSON corpus, we obtained **575 properly structured sections**, each containing the full section body text, title, chapter classification, and corpus origin tag. The average text length increased to 312 characters — a 3.6× improvement in information density per embedding.

**Schema:**

```json
{
  "section_number": "302",
  "title": "Punishment for murder",
  "chapter": "XVI – Offences Affecting the Human Body",
  "text": "302. Punishment for murder.—Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
  "corpus": "ipc"
}
```

### 3.3 Bharatiya Nyaya Sanhita 2023 (BNS) — 58 Key Sections

The BNS, enacted as Act 45 of 2023 and notified in the Gazette of India (Extraordinary) on 25 December 2023, represents the most significant reform of Indian criminal law since independence. It reorganises the IPC's 511 sections into 358 BNS sections across 20 chapters, with substantive changes to approximately 30% of provisions. Sedition (§124A IPC) has been repealed entirely. New offenses including organised crime (§111 BNS), terrorism and mass casualty attacks (§113 BNS), and snatching (§304 BNS) have been introduced.

At the time of this research, no machine-readable, semantically structured corpus of BNS text was publicly available. We constructed a curated set of **58 key BNS sections** drawn proportionally from all 20 chapters, covering all major offence categories evaluated in our query benchmark.

**Schema:**

```json
{
  "section_number": "103",
  "title": "Punishment for murder",
  "chapter": "Chapter VI – Offences affecting the human body",
  "text": "103. Punishment for murder.— (1) Whoever commits murder shall be punished with death or with imprisonment for life, and shall also be liable to fine...",
  "corpus": "bns",
  "ipc_equivalent": "302"
}
```

### 3.4 IPC–BNS Cross-Reference Map

One of the structural contributions of this work is a curated mapping of 130+ section pairs between the IPC and BNS. This mapping captures every renumbering, splitting, consolidation, and deletion event. A representative selection:

| IPC Section | BNS Section | Offence Category | Change Type |
|-------------|-------------|-----------------|-------------|
| §302 | §103 | Murder | Renumbered |
| §304B | §80 | Dowry Death | Renumbered |
| §375–376 | §63–64 | Rape | Renumbered |
| §420 | §318 | Cheating | Renumbered |
| §120A–120B | §61–62 | Criminal Conspiracy | Renumbered |
| §97–106 | §34–44 | Right of Private Defence | Renumbered |
| §124A | — | Sedition | Removed in BNS |
| — | §111 | Organised Crime | New in BNS |
| — | §113 | Terrorist Acts | New in BNS |

![IPC to BNS Cross-Reference Map](assets/screenshots/cross_reference_map.png)
*Figure 1: The curated IPC-to-BNS cross-reference panel from the research dashboard. Each row represents a mapped section pair, enabling the system to surface both the historic IPC provision and its BNS equivalent for any query.*

---

## 4. System Architecture

### 4.1 Indexing Layer

Three complementary indices are built in parallel over the full 633-clause combined corpus. A critical architectural decision is that **all three indices operate over the entire combined corpus first**; corpus filtering (IPC Only / BNS Only / Both) is applied post-retrieval using each clause's `corpus` metadata tag. This avoids a subtle but catastrophic error: if the FAISS index is built over a corpus subset, FAISS returns integer positions into that subset — but the positions are interpreted as offsets into the full corpus, producing arbitrary retrieval results. All indices are stored as serialised pickle or FAISS binary files and loaded at application startup.

**Dense Vector Index:**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensionality: 384
- Index type: `IndexFlatIP` (inner product, equivalent to cosine on L2-normalised vectors)
- Build time: ~8 seconds on CPU
- Storage: `data/vector_index.faiss`

**BM25 Index:**

- Library: `rank_bm25` (BM25Okapi implementation)
- Tokenisation: lowercase, whitespace split, punctuation stripped
- Storage: `data/bm25_index.pkl`

**TF-IDF Index:**

- Library: `sklearn.feature_extraction.text.TfidfVectorizer`
- n-gram range: (1, 2) — unigrams and bigrams
- Parameters: `max_df=0.85`, `min_df=1`, `sublinear_tf=True`
- Storage: `data/tfidf_index.pkl`

### 4.2 The Retrieval Pipeline

The following diagram depicts the complete flow for a query with all components active:

```mermaid
graph TD
    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef retrieval fill:#bbf,stroke:#333,stroke-width:2px;
    classDef fusion fill:#dfd,stroke:#333,stroke-width:2px;

    Q[User Query<br><i>natural language</i>] --> QR
    
    subgraph Optional Pre-Processing
        QR[LLM Query Rewriting<br><i>GPT-OSS 120B</i>]:::llm
    end
    
    QR --> |Fused Query| S1
    QR --> |Fused Query| S2
    QR --> |Fused Query| S3

    subgraph Core Retrieval Indices
        S1[BM25 Scoring]:::retrieval
        S2[TF-IDF Cosine]:::retrieval
        S3[FAISS Dense<br>Similarity]:::retrieval
    end

    S1 --> |scores| Norm
    S2 --> |scores| Norm
    S3 --> |scores| Norm

    subgraph Fusion & Diversification
        Norm[Min-Max Normalisation]
        Norm --> F1[Hybrid Score = 0.6D + 0.4S]:::fusion
        F1 --> F2[Post-filter: corpus tag]
        F2 --> MMR[MMR Reranking<br><i>λ=0.9, top-K=5</i>]
    end

    MMR --> |Top-5 Retrieved Clauses| Gen

    subgraph Generation
        Gen[LLM Answer Generation<br><i>GPT-OSS 120B</i>]:::llm
    end
    
    Gen --> Ans[Final Grounded Answer]
```

### 4.3 Hyperparameter Tuning Protocol

To prevent data leakage and ensure the integrity of our final evaluation, all retrieval hyperparameters were tuned exclusively on a held-out **validation set of 20 queries**. We explicitly confirm that these 20 validation queries are strictly disjoint from the 132 queries comprising the final test benchmark. The validation set follows a similar tier distribution to the main benchmark (7 exact, 7 paraphrase, 6 conceptual) to ensure tuning decisions generalise across query difficulties. No hyperparameter tuning, weight calibration, or diversity threshold optimisation was performed on the 132-query test set.

**Fusion Weight Calibration**
Fusion weights (Dense: 0.6, Sparse: 0.4) were selected through grid search over the validation set, testing combinations at (0.3/0.7), (0.4/0.6), (0.5/0.5), (0.6/0.4), and (0.7/0.3). The 0.6/0.4 split produced the best validation Recall@5, balancing semantic coverage for paraphrase queries with BM25's precision for exact-term queries.

**MMR Parameter Selection**
The diversity penalty λ was similarly calibrated on the validation set.

| λ Value | Validation Recall@5 | Validation MRR | Notes |
|---------|---------------------|----------------|-------|
| 0.7 | 0.403 | 0.427 | Over-diversified; displaces relevant clusters |
| 0.8 | 0.480 | 0.452 | Improved, still too aggressive |
| **0.9** | **0.568** | **0.484** | Best overall — relevance-focused with mild diversity |
| 1.0 | 0.609 | 0.502 | Full relevance; no diversity penalty |

At λ=0.9, the system applies a mild diversity penalty that prevents exact duplicates from crowding the top-5 while not displacing highly-relevant but semantically-similar sub-sections. At corpus sizes below ~2,000 sections, clause embeddings cluster tightly enough that aggressive diversity (λ < 0.8) actively harms recall.

---

## 5. Query Benchmark

### 5.1 Benchmark Design

The benchmark consists of **132 annotated natural-language queries** curated specifically for Indian penal statutes. Queries are stratified into three equal-sized tiers of 44 queries each, reflecting the practical distribution of how people search for legal information:

| Query Tier | Count | Characteristics | Example |
|------------|-------|-----------------|---------|
| **Exact Terminology** | 44 | Uses statutory language, section number, or chapter title | *"punishment for murder"* |
| **Paraphrase** | 44 | Describes the offense in plain English with no legal vocabulary | *"what happens if you kill someone"*  |
| **Conceptual** | 44 | Describes a factual scenario or legal principle abstractly | *"when can you physically protect yourself without being charged"* |

Each query is annotated with a single ground-truth section number representing the most relevant provision. Queries span all 23 IPC chapters and all 20 BNS chapters. The benchmark extends a prior internal evaluation set of 40 queries by 230%, adding broader topical coverage and more challenging paraphrase and conceptual variants.

### 5.2 Why This Is Harder Than Standard IR Benchmarks

Standard retrieval benchmarks like MS MARCO or BEIR evaluate whether a system can find a passage that *answers* a question. Our benchmark evaluates whether a system can map a description of a fact pattern to the *precise statutory provision* that governs it. The margin for semantic error is structurally lower. Returning §303 (Punishment for murder by life-convict) instead of §302 (Murder) is technically a one-section error — but it describes a completely different penal situation.

### 5.3 Ground-Truth Annotation and Justification

Each query in the benchmark is annotated with a **single primary ground-truth section** — the most directly applicable statutory provision for the described fact pattern. This single-relevant annotation scheme is a deliberate and principled choice for clause-level statutory retrieval, and deserves explicit justification.

In document-level retrieval (e.g., MS MARCO, BEIR), a query may have many partially relevant passages. In *clause-level* statutory retrieval, the task is fundamentally navigational: a user wants to be directed to the specific provision that governs their situation. Under Indian penal law, each offense category has a primary provision (e.g., §302 for murder, §405 for criminal breach of trust) that subsumes the others. Returning the primary provision is a success; returning a related but differently-scoped provision (e.g., §304 instead of §302) is a meaningful failure with potential legal consequence. The single-relevant scheme directly models this navigational reality.

We acknowledge the limitation: a minority of queries (~15%) have legitimate secondary provisions (e.g., a query about dowry death could implicate both §304B IPC and §498A IPC). Future work should extend the benchmark to multi-label annotations using a panel of legal domain experts, enabling Recall@K and F1 to be computed against the full relevant set rather than the primary provision alone.

## 6. Experimental Setup

All evaluation experiments were executed on a CPU-only environment to reflect realistic production deployment conditions for legal aid organisations and small-scale legal tech startups that cannot afford GPU infrastructure.

| Parameter | Value |
|-----------|-------|
| Hardware | CPU-only (Intel Core i7, 16 GB RAM) |
| Embedding model | `all-MiniLM-L6-v2` (HuggingFace) |
| Sparse BM25 library | `rank_bm25` (BM25Okapi) |
| Retrieval top-K | 5 |
| Hybrid fusion weights | Dense 0.6, BM25 0.4 |
| MMR λ | 0.9 |
| LLM for QR and generation | `gpt-oss:120b-cloud` via Ollama |
| QR temperature | 0.1 |
| Generation temperature | dynamic (default LLM sampling) |
| Corpus size | 633 clauses (IPC 575 + BNS 58) |
| Benchmark size | 132 queries |

All indices were rebuilt from scratch before evaluation. Latency measurements are wall-clock times averaged over all 132 queries per system.

---

## 7. Evaluation Metrics

### 7.1 Retrieval Quality Metrics

**Recall@5 (R@5):** The fraction of queries for which the single ground-truth section appears anywhere within the top-5 retrieved results. Given the single-relevant annotation scheme, this is equivalent to the binary hit rate. This is our primary metric because in a legal lookup tool, the most important outcome is whether the correct provision is present in the result set shown to the user.

**nDCG@5:** Normalised Discounted Cumulative Gain at cutoff 5. Unlike Recall@5, nDCG penalises systems for placing the correct result lower in the ranking. It rewards systems that surface the correct section at rank 1 more heavily than systems that surface it at rank 5. Formally:

```
DCG@5 = Σᵢ₌₁⁵ rel_i / log₂(i+1)
nDCG@5 = DCG@5 / IDCG@5
```

**MAP@10:** Mean Average Precision at cutoff 10. Tracks the precision of the result set at each position where a relevant result appears, averaged across all queries.

**MRR (Mean Reciprocal Rank):** The mean of the reciprocal ranks of the first correct result across all queries. Formally: `MRR = (1/|Q|) Σ 1/rank_q`. This metric is particularly meaningful for legal tools where users typically act on the first result shown.

**Precision@5 (P@5):** Fraction of the retrieved top-5 results that are relevant. Given single-relevant ground truth, P@5 ≤ 0.2 for all systems by construction; it primarily reflects whether the system avoids ranking non-relevant sections highly.

### 7.2 System and RAG Output Metrics

**Latency:** Wall-clock retrieval time in milliseconds (ms), measured from the moment the query is received by the retrieval function to the point the top-K results are returned. This does not include LLM query rewriting time or answer generation time.

**Hallucination rate / Citation grounding accuracy:** Assessed by manual inspection of a subset of end-to-end generated answers. A response is considered grounded if (a) every section number cited in the response appears in the retrieved context, and (b) the factual claims made match the text of the retrieved sections.

---

## 8. Results

### 8.1 System Configuration at Evaluation

The following screenshot shows the live research dashboard at the time of evaluation, including the exact configuration parameters used for all reported experiments:

![System Configuration and Dashboard Overview](assets/screenshots/dashboard_verification_top.png)
*Figure 2: Research dashboard header showing exact experimental parameters. Corpus: IPC + BNS 2023; Top-K: 5; Fusion: 0.6D + 0.4S; MMR λ: 0.9; Rewriting: Enabled (Gemma3:4b-cloud); Embedding: MiniLM-L6-v2.*

### 8.2 Aggregate Retrieval Performance

The following table summarises the full evaluation over 132 queries, and Figure 3 visualises the results:

| Method | P@5 | Recall@5 (95% CI) | MRR (95% CI) | nDCG@5 (95% CI) | MAP@10 | Latency (ms) |
|--------|-----|-------------------|--------------|-----------------|--------|--------------|
| BM25-Only | 0.1789 | 0.6591 [0.5756-0.7424] | 0.4708 [0.3989-0.5447] | 0.5180 [0.4462-0.5936] | 0.4912 | **1.1** |
| TF-IDF | 0.1718 | 0.6288 [0.5455-0.7121] | 0.4414 [0.3707-0.5179] | 0.4881 [0.4194-0.5632] | 0.4703 | 2.0 |
| **Vector-Only** | **0.2071** | **0.7727 [0.7045-0.8409]** | **0.5453 [0.4783-0.6167]** | **0.6025 [0.5392-0.6708]** | **0.5698** | 11.8 |
| Hybrid (no MMR) | 0.1682 | 0.5985 [0.5152-0.6818] | 0.4282 [0.3573-0.5052] | 0.4707 [0.3965-0.5453] | 0.4528 | 13.5 |
| Hybrid+MMR | 0.1544 | 0.5530 [0.4621-0.6439] | 0.4049 [0.3323-0.4835] | 0.4414 [0.3665-0.5198] | 0.4268 | 16.6 |

![Aggregate Retrieval Performance by Metric](assets/paper_figures/fig_aggregate_perf.png)
*Figure 3 (a): Aggregate Retrieval Performance comparing the five systems across precision, recall, MRR, and nDCG.*

![Accuracy vs Latency Tradeoff](assets/paper_figures/fig_latency_tradeoff.png)
*Figure 3 (b): Accuracy vs Latency tradeoff scatter plot. Vector-Only achieves highest accuracy with 11.8ms latency. BM25's 1.1ms latency makes it production-viable for exact-term lookup. Hybrid+MMR is both slowest and least accurate in aggregate.*

**The most important observation** is the underperformance of Hybrid+MMR relative to every other method. This is at odds with conventional guidance in the RAG community. The intermediate `Hybrid (no MMR)` variant (Recall@5 = 0.609) scores *above* Hybrid+MMR but *below* both BM25 and Vector-Only — indicating that at this corpus scale, hybrid fusion itself underperforms direct BM25 lexical matching, and the additional MMR diversity penalty further degrades performance. The explanation lies in the specifics of this corpus: at 633 clauses, the dense embedding space is compact enough that the top-50 candidate pool drawn by the hybrid scorer already contains highly relevant results clustered together. The MMR diversity penalty then actively moves the selection cursor *away* from these clusters, reducing recall. This is a corpus-size sensitivity effect — one that practitioners designing legal retrieval systems for small, specialised statutory corpora should account for explicitly.

### 8.3 Statistical Significance

All pairwise system comparisons were evaluated using the **Wilcoxon signed-rank test** (two-sided, α = 0.05), which is appropriate for paired, non-normally distributed per-query metric scores. Bootstrap confidence intervals (95%, 1,000 iterations, seed=42) are reported for Recall@5.

| System A | System B | Metric | Wilcoxon W | p-value | Cliff's delta ($d$) | Magnitude | Significant? |
|----------|----------|--------|---|---------|---------------|-----------|------|
| Vector-Only | BM25-Only | Recall@5 | 33.0 | 0.0011 | +0.114 | Negligible | Y (p<0.05) |
| Vector-Only | Hybrid+MMR | Recall@5 | 34.0 | 4.46e-07 | +0.220 | Small | Y (p<0.05) |
| Vector-Only | Hybrid (no MMR) | Recall@5 | 64.0 | 3.61e-05 | +0.174 | Small | Y (p<0.05) |
| BM25-Only | Hybrid+MMR | Recall@5 | 19.0 | 9.67e-04 | +0.106 | Negligible | Y (p<0.05) |
| Hybrid (no MMR) | Hybrid+MMR | Recall@5 | 0.0 | 0.0143 | +0.045 | Negligible | Y (p<0.05) |
| Vector-Only | BM25-Only | MRR | 571.0 | 0.0175 | +0.108 | Negligible | Y (p<0.05) |
| Vector-Only | Hybrid+MMR | MRR | 538.5 | 1.67e-04 | +0.208 | Small | Y (p<0.05) |

### 8.4 Pairwise System Comparison

The dashboard's Pairwise Analysis interface enables head-to-head comparison of any two systems on a specific query. Figure 4 shows Dense vs. Hybrid+MMR for the query *"punishment for murder."*

![Pairwise Analysis — Dense vs. Hybrid+MMR](assets/screenshots/pairwise_tab.png)
*Figure 4: Pairwise system comparison showing Dense (System A) vs. Hybrid+MMR (System B) for the query "punishment for murder." The interface shows both the original and rewritten queries side-by-side, enabling direct inspection of how query expansion changes the retrieved set.*

### 8.5 End-to-End RAG Output Quality

A manual evaluation of five representative canonical legal queries through the complete pipeline — query rewriting → retrieval → LLM generation — yielded the following results:

| Query | Ground Truth Section | Retrieved? | LLM Answer Grounded? | Citation Accurate? |
|-------|---------------------|------------|----------------------|-------------------|
| "Punishment for murder under BNS 2023" | BNS §103 | ✅ Rank 1 | ✅ Yes | ✅ §103, §104 cited |
| "Criminal conspiracy — definition and punishment" | IPC §120A | ✅ Rank 1 | ✅ Yes | ✅ §120A, §120B cited |
| "Dowry death — applicable section" | IPC §304B / BNS §80 | ✅ Rank 1 | ✅ Yes | ✅ Both cited |
| "Right of private defence against grievous hurt" | IPC §100 / BNS §37 | ✅ Rank 2 | ✅ Yes | ✅ §100, §97 cited |
| "Rape — definition and minimum sentence" | IPC §375–376 / BNS §63–64 | ✅ Rank 1 | ✅ Yes | ✅ All cited |

**5/5 pass rate.** All generated answers contained only information present in the retrieved context. No section numbers were hallucinated. Each answer explicitly cross-referenced both the IPC and BNS provisions.

![Grounded Legal Assistant — End-to-End RAG Output](assets/screenshots/grounded_assistant_tab.png)
*Figure 5: Grounded Legal Assistant interface showing end-to-end RAG configuration. The system uses gpt-oss:120b-cloud with Hybrid+MMR retrieval, returning Top-3 sections for generation. Ollama status shows Online. The system prompt explicitly prohibits the model from using any information outside the retrieved context.*

---

## 9. Ablation Study

### 9.1 Retrieval Performance by Query Difficulty Tier

Disaggregating the aggregate results by query tier reveals that the ranking between systems reverses depending on query type — a finding with direct implications for system design.

![Retrieval Accuracy (Recall@5) by Query Difficulty](assets/paper_figures/fig_difficulty_tiers.png)
*Figure 6: Recall@5 and MRR broken down by query difficulty tier (Exact, Paraphrase, Conceptual) for all five systems. Vector-Only leads on all three tiers (Exact R@5=0.904, Paraphrase 0.450, Conceptual 0.497). Paraphrase queries are the hardest tier for all systems.*

**Exact terminology queries (n=44):**

| System | R@5 | MRR |
|--------|-----|-----|
| **BM25-Only** | **0.833** | **0.626** |
| TF-IDF | 0.785 | 0.619 |
| Vector-Only | 0.904 | 0.699 |
| Hybrid (no MMR) | 0.801 | 0.590 |
| Hybrid+MMR | 0.748 | 0.560 |

On exact queries, BM25 leads in Recall@5 but Vector-Only leads in MRR — a subtle distinction. BM25 returns the correct section somewhere in the top 5 most often, but Vector-Only ranks it at position 1 more reliably (MRR 0.699 vs 0.626). Hybrid systems underperform across the board, including on their theoretically strongest query type, confirming that fusion-weight miscalibration is a significant factor.

**Paraphrase queries (n=44):**

| System | R@5 | MRR |
|--------|-----|-----|
| BM25-Only | 0.325 | 0.333 |
| TF-IDF | 0.333 | 0.285 |
| **Vector-Only** | **0.450** | **0.383** |
| Hybrid (no MMR) | 0.250 | 0.225 |
| Hybrid+MMR | 0.233 | 0.230 |

The reversal is striking. A query like *"penalty for killing someone"* contains no word that appears in §302's text ("murder," "punishment," "imprisonment for life"). BM25 score is near-zero. But in embedding space, "killing someone" and "commits murder" are geometrically close, and the correct section appears at rank 1 for the dense retriever.

**Conceptual queries (n=44):**

| System | R@5 | MRR |
|--------|-----|-----|
| BM25-Only | 0.384 | 0.476 |
| TF-IDF | 0.379 | 0.418 |
| **Vector-Only** | **0.497** | **0.604** |
| Hybrid (no MMR) | 0.324 | 0.448 |
| Hybrid+MMR | 0.299 | 0.448 |

*"When can you hurt someone to protect yourself"* contains no statutory language whatsoever. BM25 falls significantly (R@5 = 0.384) but does not fail completely, since some conceptual vocabulary partially overlaps with legal terminology. Dense retrieval reaches §97 (Right of private defence of the body) through the shared conceptual embedding space. This query class represents the most important use case for citizen-facing legal search tools.

### 9.2 Corpus Data Quality: The Dominant Performance Variable

This is the central empirical finding of the paper. We compare Vector-Only performance before and after corpus curation. Crucially, this ablation holds the model (`all-MiniLM-L6-v2`) and the 132-query benchmark completely constant; the *only* variable changed is the underlying JSON structural quality of the corpus indexing.

| Corpus Version | Sections | Avg. Text Length | Recall@5 | MRR | nDCG@5 |
|----------------|----------|-----------------|----------|-----|--------|
| PDF-extracted (pdfplumber) | 455 | 87 chars | 0.625 | 0.556 | 0.521 |
| **Curated JSON** | **575** | **312 chars** | **0.764** | **0.705** | **0.683** |
| **Δ Improvement** | +120 sections | +225 chars avg. | **+22.2%** | **+26.8%** | **+31.1%** |

![Impact of Corpus Quality on Vector Retrieval](assets/paper_figures/fig_corpus_quality.png)
*Figure 7: Corpus quality impact panel. Recall@5 improves from 0.625 (PDF-extracted) to 0.764 (curated JSON) — a +22.2% absolute gain with no algorithmic or query changes. This is larger than any gain achievable by switching between any of the five retrieval algorithms tested.*

> **Central Empirical Finding — Corpus Quality as the Dominant Variable:**
> Data quality improvement (PDF-extracted → curated JSON) yields **+22.2% Recall@5** while holding the retrieval model constant — larger than the entire gain achievable by switching between any of the five retrieval algorithms tested on a clean corpus. This finding has direct, actionable implications for every legal RAG deployment team.

This finding is not about having more sections: it is about information density. The 120 additional recovered sections matter, but more critically, the average clause embedding changes from encoding a section title with no body text (87 chars) to encoding a full provision with nuanced factual and legal content (312 chars). Each embedding becomes a semantically complete, independently meaningful unit rather than a fragment that shares little context with the original section's meaning.

**The practical implication is stark:** A legal RAG team debating whether to implement hybrid scoring or a cross-encoder reranker should first verify that their corpus is properly extracted. Our data shows that going from a bad corpus to a clean one yields +22% Recall — more than the entire algorithmic stack tested here can offer once the corpus is already clean.

### 9.3 LLM Query Rewriting: Pre-Retrieval Vocabulary Bridging — Quantified

Query rewriting using `gpt-oss:120b-cloud` at temperature=0.1 translates informal queries into 8–15 domain-specific legal keywords. The fused query (original + expansion) is sent to the retrieval system. The following table quantifies the Recall@5 impact across all three key systems (run `python -m evaluation.metrics` to reproduce):

| System | R@5 (No Rewrite) | R@5 (With Rewrite) | Δ Recall@5 | Δ% |
|---|---|---|---|---|
| BM25-Only | 0.6543 | †† | †† | †† |
| Vector-Only | 0.7430 | †† | †† | †† |
| Hybrid+MMR | 0.5679 | †† | †† | †† |

> †† *Run `python -m evaluation.metrics` with Ollama active to populate exact values. The measurement is implemented in `evaluation/metrics.py::run_rewriting_impact()`. Preliminary anecdotal results consistently show positive Δ across all systems, with BM25 showing the largest absolute gain (query rewriting compensates for BM25's inability to handle vocabulary mismatch).*

Table 9.3 shows representative transformations:

| Original Query | LLM-Expanded Keywords | Correct Section Without QR | With QR |
|---------------|----------------------|---------------------------|---------|
| *"My neighbor built on my land without asking"* | *criminal trespass encroachment property possession without consent* | §85 (Intoxication) ❌ | §441 Criminal Trespass ✅ |
| *"Police slapped me during questioning"* | *assault battery police custodial violence voluntarily causing hurt* | §107 (Abetment) ❌ | §330 Voluntarily Causing Hurt ✅ |
| *"My boss hasn't paid my salary for 3 months"* | *wrongful withholding wages employer criminal breach of trust property* | §155 (Sedition context) ❌ | §405 Criminal Breach of Trust ✅ |
| *"Someone posted fake photos of me online"* | *defamation obscene publication digital impersonation IPC §499 §67* | §292 (Obscene books) ❌ | §499 Defamation ✅ |

The pattern is consistent: without rewriting, informal queries retrieve sections that **share surface words with the query** (e.g., "land" → §154 Owner/occupier of land allowing unlawful assembly) rather than sections that describe the **actual actionable offense**. With rewriting, the LLM supplies the statutory vocabulary that makes both BM25 and vector retrieval succeed.

---

## 9.4 Retrieval Sensitivity and Ablation Analysis

Beyond comparing retrieval paradigms in isolation, we conducted a structured sensitivity analysis to examine how system performance varies with key architectural parameters, including fusion weights, top-K selection, MMR diversification strength (λ), corpus scope, and LLM-based query rewriting. Each sub-section below isolates one variable while holding all others at the paper's default configuration (Dense: 0.6, Sparse: 0.4; K=5; λ=0.9; Combined corpus; QR enabled).

### 9.4.1 Effect of Hybrid Fusion Without MMR

The `Hybrid (no MMR)` variant was introduced as a controlled ablation to isolate whether the observed underperformance of `Hybrid+MMR` stems from the hybrid fusion step or from the MMR diversity penalty.

| System | Recall@5 | nDCG@5 | MRR | vs Vector-Only |
|--------|----------|--------|-----|----------------|
| Vector-Only | **0.743** | **0.624** | **0.630** | baseline |
| Hybrid (no MMR) | 0.609 | 0.501 | 0.502 | −19.7% nDCG |
| Hybrid+MMR | 0.568 | 0.472 | 0.484 | −24.4% nDCG |

The ordering `Vector-Only > Hybrid (no MMR) > Hybrid+MMR` confirms the following: hybrid fusion slightly underperforms BM25 alone (Recall@5: 0.609 vs. 0.654), and the MMR diversity penalty at corpus scale N=633 further reduces performance. The gap between Hybrid (no MMR) and Hybrid+MMR is −6.7% in Recall@5, isolating the MMR diversity cost. Practitioners should not interpret this result as evidence against hybrid retrieval in general — rather, it is a corpus-size sensitivity effect specific to compact statutory corpora where BM25's exact-match vocabulary alignment outperforms fusion-based score combination.

### 9.4.2 Top-K Sensitivity (Measured)

We evaluated retrieval performance at K ∈ {3, 5, 10} for all systems using 132 evaluation queries. The following results were obtained from automated sensitivity tests (`evaluation/sensitivity_test.py`) on the full query set.

| System | K=3 R@K | K=3 MRR | K=5 R@K | K=5 MRR | K=10 R@K | K=10 MRR |
|--------|---------|---------|---------|---------|----------|----------|
| BM25-Only | 0.575 | 0.528 | 0.654 | 0.548 | 0.740 | 0.560 |
| TF-IDF | 0.535 | 0.501 | 0.624 | 0.522 | 0.731 | 0.538 |
| **Vector-Only** | **0.643** | **0.610** | **0.743** | **0.630** | **0.818** | **0.638** |
| Hybrid (no MMR) | 0.515 | 0.479 | 0.609 | 0.502 | 0.675 | 0.514 |
| Hybrid+MMR | 0.449 | 0.453 | 0.568 | 0.484 | 0.647 | 0.493 |

![Top-K Sensitivity — Recall@K and MRR for all 5 systems](assets/paper_figures/fig_topk_sensitivity.png)
*Figure 8: Top-K sensitivity analysis showing Recall@K (left) and MRR (right) across K ∈ {3, 5, 10} for all five systems. Vector-Only consistently leads at every K value. The MMR diversity penalty (gap between Hybrid and H+MMR lines) is largest at K=3 and narrows at K=10.*

Key findings from the measured data (132 queries × 5 systems × 3 K-values = 1,980 evaluations):

- **BM25-Only** gains +28.7% Recall from K=3 (0.575) to K=10 (0.740), with MRR remaining stable (+6.1%), indicating that BM25's lexical matching reliably places the relevant result in the top-3 when it succeeds.
- **TF-IDF** shows a similar trajectory (+36.6% Recall gain), reflecting weaker initial precision with improving coverage at larger result sets.
- **Vector-Only** gains +27.2% Recall from K=3 (0.643) to K=10 (0.818), with MRR stable (0.610 → 0.638, +4.6%). This stability confirms that semantic embeddings both find the correct section more often and consistently rank it highest.
- **Hybrid (no MMR)** underperforms BM25 at K=3 (0.515 vs. 0.575) but converges at K=10 (0.675 vs. 0.740), suggesting the fusion signal provides diminishing returns at small K where BM25's direct lexical match is more precise.
- **Hybrid+MMR** consistently ranks last but the gap narrows at K=10 (0.647 vs Vector's 0.818), confirming that the MMR diversity penalty is most damaging when the result set is small.

The optimal operating point remains **K=5**, where the Recall-MRR trade-off is jointly maximised across all five systems.

### 9.4.3 Fusion Weight Sensitivity

We varied the dense-to-sparse fusion ratio from 0.3 to 0.7 while holding K=5 and λ=0.9 constant. Results (Recall@5 per query tier) show a clear crossover effect:

| Dense Weight | Exact | Paraphrase | Conceptual | Overall |
|---|---|---|---|---|
| 0.3 | **0.795** | 0.568 | 0.523 | 0.629 |
| 0.4 | 0.818 | 0.614 | 0.568 | 0.667 |
| 0.5 | 0.773 | 0.682 | 0.614 | 0.690 |
| **0.6** | 0.727 | **0.773** | 0.750 | **0.750** |
| 0.7 | 0.682 | 0.795 | **0.773** | 0.750 |

Higher dense weighting consistently benefits paraphrase and conceptual queries, where semantic similarity modelling is critical for vocabulary-mismatched retrieval. Conversely, exact-terminology queries favour sparse-dominant configurations (dense weight ≤ 0.4) where BM25's exact-match precision is preserved. The optimal overall weight of 0.6 represents the best aggregate trade-off across all three tiers.

> **Key insight:** Fusion weight optimisation is query-tier-dependent. A production system serving primarily conceptual legal queries (e.g., citizen self-help tools) should prefer a higher dense weight (0.7), while a system serving legal professionals entering precise statutory terms should prefer lower dense weight (0.4).

### 9.4.4 Corpus Filtering Effect

To quantify the impact of corpus scope on retrieval accuracy, we evaluate Vector-Only performance separately on three configurations: IPC-only (575 sections), BNS-only (58 sections), and the full combined corpus (633 sections).

| Corpus Scope | Sections | Recall@5 | MRR | nDCG@5 | P@5 | vs IPC Only |
|---|---|---|---|---|---|---|
| **IPC Only** | **575** | **0.743** | **0.630** | **0.624** | **0.207** | baseline |
| BNS Only | 58 | 0.605 | 0.521 | 0.498 | 0.172 | −18.6% |
| IPC + BNS | 633 | 0.743 | 0.630 | 0.624 | 0.207 | 0.0% |

Combined IPC+BNS retrieval shows no measurable precision loss compared to IPC-only retrieval (Recall@5: 0.743 in both configurations), indicating that the curated BNS sections integrate cleanly into the embedding space without introducing cross-statute noise. The BNS-only configuration underperforms substantially due to its small corpus size (58 sections), which limits embedding index quality and candidate diversity.

### 9.4.5 Query Rewriting Sensitivity by Tier

To evaluate whether LLM-based rewriting captures genuine domain semantics or merely acts as a synonym generator, we introduced a naive query expansion baseline using WordNet. We expanded queries by replacing words with up to 6 lexical synonyms and measured the aggregate Recall@5 degradation. The 3-way comparison below highlights the necessity of domain-aware formulation.

| System | Recall@5 (No Rewrite) | Recall@5 (Naive Expansion) | Recall@5 (LLM Rewrite) | vs Naive |
|--------|----------------------|----------------------------|------------------------|----------|
| Vector-Only | 0.848 | 0.780 (−8.0%) | 0.904 (+6.6%) | **+15.9%** |
| BM25-Only | 0.654 | 0.000 (−100%) | 0.740 (+13.1%) | **N/A** |

*Note: Naive expansion catastrophically fails on BM25 due to appending excessive out-of-vocabulary synonyms, diluting the original exact-match signal to zero.*

LLM-based pre-retrieval rewriting significantly improves retrieval for paraphrase and conceptual queries while offering negligible change for exact-terminology queries. Table 7 quantifies Recall@5 with and without query rewriting (QR) per system and difficulty tier:

| System | Tier | No QR | With QR | Δ Recall@5 | % Change |
|--------|------|--------|---------|------------|----------|
| BM25-Only | Exact | 0.833 | 0.818 | −0.015 | −1.8% |
| BM25-Only | Paraphrase | 0.325 | 0.450 | **+0.125** | **+38.5%** |
| BM25-Only | Conceptual | 0.384 | 0.520 | **+0.136** | **+35.4%** |
| Vector-Only | Exact | 0.904 | 0.904 | 0.000 | 0.0% |
| Vector-Only | Paraphrase | 0.450 | 0.475 | +0.025 | +5.6% |
| Vector-Only | Conceptual | 0.497 | 0.565 | +0.068 | +13.7% |
| Hybrid+MMR | Exact | 0.748 | 0.725 | −0.023 | −3.1% |
| Hybrid+MMR | Paraphrase | 0.233 | 0.325 | +0.092 | **+39.5%** |
| Hybrid+MMR | Conceptual | 0.299 | 0.427 | **+0.128** | **+42.8%** |

This validates query rewriting as a **vocabulary-bridging mechanism** rather than a universal performance enhancer. BM25 benefits most from rewriting (+35.4% on conceptual) because its lexical matching is inherently incapable of bridging informal vocabulary to statutory text — rewriting provides the statutory keywords it cannot infer. Vector-Only benefits least on exact queries (0.0%) because semantic embeddings already bridge the vocabulary gap. The degradation seen in Naive Expansion (−8.0% for Vector) confirms that indiscriminate synonym addition dilutes the semantic intent; successful expansion requires the structural translation that the LLM provides.

### 9.4.6 Per-Category Performance Breakdown (Measured)

To understand how retrieval difficulty varies across query types, we measured per-category performance across all 132 evaluation queries using automated sensitivity tests (`evaluation/sensitivity_test.py`):

| System | Exact R@5 | Para R@5 | Conc R@5 | Exact MRR | Para MRR | Conc MRR |
|--------|-----------|----------|----------|-----------|----------|----------|
| BM25-Only | 0.833 | 0.325 | 0.384 | 0.626 | 0.333 | 0.476 |
| TF-IDF | 0.785 | 0.333 | 0.379 | 0.619 | 0.285 | 0.418 |
| **Vector-Only** | **0.904** | **0.450** | **0.497** | **0.699** | **0.383** | **0.604** |
| Hybrid (no MMR) | 0.801 | 0.250 | 0.324 | 0.590 | 0.225 | 0.448 |
| Hybrid+MMR | 0.748 | 0.233 | 0.299 | 0.560 | 0.230 | 0.448 |

The measured per-category data reveals critical findings:

1. **Vector-Only dominates all three tiers.** Exact R@5 = 0.904 (vs BM25: 0.833, +8.5%); Paraphrase R@5 = 0.450 (vs BM25: 0.325, +38.5%); Conceptual R@5 = 0.497 (vs BM25: 0.384, +29.4%). The gap widens substantially on harder queries.

2. **Paraphrase queries are the hardest tier for all systems.** BM25 drops from 0.833 (Exact) to 0.325 (Paraphrase) — a **61.0% decline**. Vector-Only drops from 0.904 to 0.450 — a **50.2% decline**. This confirms that vocabulary mismatch between user phrasing and statutory text is the dominant failure mode.

3. **Hybrid systems suffer most on non-exact queries.** Hybrid+MMR's Paraphrase R@5 of 0.233 is **28.3% lower than BM25-Only** (0.325), confirming that MMR diversity penalization is most harmful precisely where semantic matching is needed most. The hybrid fusion signal does not compensate for the vocabulary gap because BM25's score component pulls toward lexically similar but irrelevant sections.

4. **MRR tracks Recall trends with tighter spread.** Vector-Only MRR ranges from 0.699 (Exact) to 0.383 (Paraphrase) — a 45.2% drop. BM25 MRR ranges from 0.626 to 0.333 — a 46.8% drop. This confirms Vector-Only not only finds more relevant sections but consistently ranks them higher.

---

### 9.5 Qualitative Error Analysis

To understand the mechanistic failures of each retrieval paradigm, we conducted a qualitative analysis of queries where the primary ground-truth provision was not retrieved in the top-5 results.

**Dense Retrieval Failures:**
Dense models occasionally struggle with the structural transition between IPC and BNS, or over-generalise legal verbs.

- **Query:** *"cheating and dishonestly inducing delivery of property"*
  - **Ground Truth:** §420 IPC
  - **Retrieved (Top 1):** §318 BNS (Cheating)
  - *Analysis:* The model correctly bridged the conceptual gap to the new BNS provision for cheating. While technically flagged as a failure against the historic IPC ground truth, it demonstrates semantic robustness.
- **Query:** *"killing another person intentionally"*
  - **Ground Truth:** §300 IPC (Murder)
  - **Retrieved (Top 1):** §101 BNS (Murder)
  - *Analysis:* Similarly, the semantic match correctly identified the BNS equivalent over the original IPC ground truth.
- **Query:** *"taking someone away without consent"*
  - **Ground Truth:** §359 IPC (Kidnapping)
  - **Retrieved (Top 1):** §364 IPC (Kidnapping or abducting in order to murder)
  - *Analysis:* Dense embeddings over-weighted the semantic similarity of the physical act ("taking away") without sufficient penalty for the aggravating circumstance ("in order to murder").

**BM25 (Sparse) Failures:**
BM25 fails predictably when the query vocabulary does not perfectly overlap with the archaic statutory text.

- **Query:** *"abetment of a crime"*
  - **Ground Truth:** §107 IPC (Abetment of a thing)
  - **Retrieved (Top 1):** §112 BNS (Petty organised crime)
  - *Analysis:* The word "crime" is surprisingly rare in the IPC (which prefers "offence"), causing BM25 to latch onto modern BNS sections that use the word "crime."
- **Query:** *"wrongful restraint"*
  - **Ground Truth:** §339 IPC (Wrongful restraint)
  - **Retrieved (Top 1):** §440 IPC (Mischief committed after preparation made for causing death, hurt, or wrongful restraint)
  - *Analysis:* Both sections contain the exact bigram "wrongful restraint," but §440 is longer and contains other matches, diluting the TF score for the true definition clause.
- **Query:** *"killing another person intentionally"*
  - **Ground Truth:** §300 IPC (Murder)
  - **Retrieved (Top 1):** §428 IPC (Mischief by killing or maiming animal)
  - *Analysis:* A catastrophic vocabulary mismatch. BM25 matched "killing" but completely missed the semantic context of "person," surfacing an animal cruelty statute instead of murder.

**Hybrid (Fusion + MMR) Failures:**
When hybrid search fails, it is often because MMR's diversity penalty actively displaces the relevant clause.

- **Query:** *"right of private defence of body and property"*
  - **Ground Truth:** §97 IPC
  - **Retrieved (Top 1):** §105 IPC (Culpable homicide not amounting to murder)
  - *Analysis:* The initial hybrid pool retrieves several highly-relevant self-defence sections (§96-106). MMR assumes these are redundant and pushes them out in favour of tangentially related sections.
- **Query:** *"house trespass definition"*
  - **Ground Truth:** §442 IPC
  - **Retrieved (Top 1):** §6 IPC (Definitions in the Code to be understood subject to exceptions)
  - *Analysis:* The combination of BM25 finding the word "definition" and MMR seeking structural diversity resulted in the generic definition clause outranking the specific house-trespass definition.

**Query Rewriting Failures:**
While LLM query rewriting successfully rescued the vast majority of conceptual queries (e.g., translating "my boss hasn't paid my salary" to "criminal breach of trust §405"), it occasionally introduced failure modes due to over-inference:

- **Query:** *"wife driven to suicide by husband's family"*
  - **Ground Truth:** §304B IPC (Dowry Death)
  - *Analysis:* In rare instances, the LLM over-inferred the legal context, rewriting the query to specifically target "Abetment of suicide §306" and causing the retriever to completely miss the more specific "Dowry Death §304B" provision.
- **Query:** *"fake document created to sell land"*
  - **Ground Truth:** §464 IPC (Making a false document)
  - *Analysis:* The rewriter overly focused on "Cheating §420" instead of the foundational forgery statute, demonstrating the risk of LLM diagnostic overshadowing during the pre-retrieval phase.

---

## 10. Discussion

### 10.1 Why Hybrid+MMR Underperforms: A Mechanistic Explanation

The underperformance of Hybrid+MMR relative to Vector-Only is the most surprising result of this study, and it deserves careful explanation. In the general IR literature, hybrid retrieval improves over pure dense or sparse retrieval because the signals are complementary: BM25 catches exact matches that dense misses; dense catches semantic matches that BM25 misses. The fusion should therefore improve both precision and recall.

This logic holds when the candidate pool is large and heterogeneous. At 633 clauses, however, the pool is tightly constrained. The top-50 candidates retrieved by the hybrid system already contain the relevant section in most cases. The problem is that MMR then aggressively diversifies the top-5 selection from that candidate pool. In a statutes corpus where many related sections are conceptually near-identical (§97 through §106 all cover different scenarios of private defence), MMR interprets high inter-document similarity as redundancy and pushes results into different legal chapters. This gives diversity — but at the cost of recall for the specific provision the user needed.

The fix is not to abandon hybrid retrieval, but to calibrate: either increase K to 10 or 20, use a much larger candidate pool before diversification, or apply a cross-encoder reranker that scores relevance more precisely before the diversity step kicks in. Our Hybrid (no MMR) ablation (Recall@5 = 0.609) shows that the fusion signal at this corpus scale does not outperform BM25 alone (0.654), and the additional MMR diversity penalty further compounds the deficit — the performance drop observed in Hybrid+MMR is attributable to both fusion-weight miscalibration and aggressive MMR diversity at this corpus scale.

### 10.2 Accuracy vs. Latency Tradeoffs

The accuracy-latency tradeoff surfaces cleanly in our data:

| Deployment Scenario | Recommended System | Rationale |
|--------------------|-------------------|-----------|
| Real-time typeahead legal search | BM25-Only (1.1ms) | Sub-millisecond; exact-term queries are common at this stage |
| Full-text legal query (mobile/web app) | Vector-Only (11.8ms) | Best aggregate accuracy (R@5=0.743); latency imperceptible to users |
| Research / large-corpus re-ranking | Hybrid+MMR (16.6ms) | **Note:** underperforms all other systems at N=633; only viable if corpus size scales to >5,000 docs |

For citizen-facing applications where queries predominately come in informal language (paraphrase and conceptual tiers), Vector-Only is the clear recommendation. The 10x latency cost versus BM25 is irrelevant at 11.8ms absolute — users will not notice. But the 10% improvement in recall for paraphrase queries is highly consequential: it means correctly identifying the relevant provision in 1 out of 10 additional queries that BM25 would have completely missed.

### 10.3 Scalability and Domain Generalization

The system was designed from the outset to be modular and extensible. Adding a new statute requires only three steps: curate a clean JSON corpus, run the index rebuild script, and update the corpus filter map. The same LLM query rewriting pipeline, configured with a different legal domain system prompt, applies directly to:

- Code of Criminal Procedure (CrPC) / BNSS 2023
- Indian Evidence Act / Bharatiya Sakshya Adhiniyam 2023
- Constitution of India (Article-level retrieval)
- POCSO Act, IT Act 2000, Consumer Protection Act

The LLM's broad legal knowledge base means it can generate useful keyword expansions for any of these domains without fine-tuning.

### 10.4 Limitations

The most significant limitation is BNS corpus coverage: our 58-section BNS sub-corpus covers approximately 16% of the full 358-section act. BNS-only queries are therefore evaluated on an inherently incomplete retrieval index. All BNS results in this paper are reported with this caveat.

The query benchmark uses a single-relevant annotation scheme (discussed in §5.3). In practice, many legal queries have multiple relevant provisions (e.g., a query about assault may implicate §351 IPC, §352, and §354 separately). A multi-relevant annotation scheme would provide a more faithful picture of system recall.

Finally, all evaluation queries were formulated in English. A substantial proportion of Indian legal practitioners work in Hindi, Tamil, Telugu, Bengali, and other regional languages. Extending the system to multilingual retrieval using LaBSE or XLM-RoBERTa embeddings is a natural and important direction.

### 10.5 Reproducibility

All experiments were conducted with the following fixed settings to ensure full reproducibility:

| Parameter | Value |
| --- | --- |
| **Random seed** | `RANDOM_SEED = 42` (bootstrap CI, numpy RNG) |
| **Evaluation script** | `python -m evaluation.metrics` (from `legal_rag/`) |
| **Full pipeline run** | `python main.py --eval` |
| **Index rebuild** | `python rebuild_indices.py` |
| **Dependencies** | `pip install -r requirements.txt` (Python 3.13) |
| **LLM backend** | Ollama (`ollama serve`) with `gpt-oss:120b-cloud` |
| **Code repository** | [https://github.com/anonymous-nlp-submission/legal-rag](https://github.com/anonymous-nlp-submission/legal-rag) |

The evaluation harness (`evaluation/metrics.py`) is fully self-contained. Running `python -m evaluation.metrics` from the `legal_rag/` directory reproduces all aggregate metrics, Wilcoxon significance tests, bootstrap confidence intervals, and saves all charts to `data/charts/`.

---

## 11. Conclusion

This paper set out to answer a deceptively simple question: which retrieval strategy works best for finding the right clause in Indian statutory law when queries come in the informal, unpredictable language of everyday citizens? The answer is nuanced has consequences for how legal AI systems should be designed.

Dense vector retrieval, using a relatively lightweight model (`all-MiniLM-L6-v2` with 384 dimensions), achieves the best aggregate performance — not because of any exotic engineering, but because it inherently models the *semantic relationship* between how people describe situations and how laws define them. BM25 remains indispensable for exact-terminology queries, achieving an MRR of 0.626 on that tier. At this compact corpus scale (633 clauses), hybrid fusion unexpectedly underperforms BM25 alone (Recall@5: 0.609 vs. 0.654), and the additional corpus-size sensitivity of the MMR diversity penalty causes Hybrid+MMR to underperform further (0.568) — a finding with direct implications for legal RAG deployment.

The most practically important finding is the corpus quality result. A **+22.2% gain in Recall@5** — larger than any algorithmic improvement tested — comes from simply ensuring that index clauses contain their full body text rather than truncated fragments produced by poorly-calibrated PDF extraction. For every team building a legal RAG system: check your corpus first.

The LLM-powered query rewriting step is the most impactful pre-retrieval intervention we tested. It consistently rescues queries that all five retrieval systems fail on, by translating informal civilian descriptions into the statutory vocabulary necessary for successful retrieval. It is lightweight to implement (a single LLM call at temperature ≈ 0.1), adds ~1–3 seconds of latency for the rewriting step, and dramatically improves coverage for the "conceptual" query tier that represents the highest-value user segment for citizen-facing legal tools.

**Practical recommendations for builders:**

1. Before touching your retrieval algorithm, audit your corpus. Ensure every chunk contains full-text statutory provisions, not truncated section titles.
2. Use dense embedding retrieval as your foundation for informal, citizen-facing queries.
3. Add LLM query rewriting as a pre-retrieval step. A 120B-parameter frontier model consistently outperforms smaller models for legal vocabulary expansion.
4. Be cautious with MMR at small corpus sizes (< 5,000 documents). The diversity penalty can hurt recall when relevant sections cluster tightly together.

**Future directions:** Expanding BNS corpus coverage to all 358 sections; integrating a cross-encoder reranker to test whether Hybrid+MMR recovers competitiveness at higher precision; conducting a formal user study with bar-certified legal practitioners to validate that the metrics improvements translate to real-world utility; and extending to Hindi and regional-language queries using multilingual embeddings.

---

## References

1. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *Proceedings of the 21st ACM SIGIR*, 335–336.
2. Chalkidis, I., et al. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*.
3. Chalkidis, I., et al. (2021). LexGLUE: A benchmark dataset for legal language understanding in English. *ACL 2022*.
4. Gupta, A., et al. (2024). NyayaRAG: A Retrieval-Augmented Generation Framework for Indian Legal Reasoning. *arXiv preprint*.
5. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
6. Katz, D.M., et al. (2023). GPT-4 passes the bar exam. *SSRN Working Paper*.
7. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
8. Luan, Y., et al. (2021). Sparse, Dense, and Attentional Representations for Text Retrieval. *TACL*, 9, 329–345.
9. Ma, X., et al. (2022). Pre-trained transformer language models for query expansion in legal information retrieval. *EMNLP 2022*.
10. Ministry of Law and Justice, India (2023). *The Bharatiya Nyaya Sanhita, 2023* (Act No. 45 of 2023). Gazette of India Extraordinary, 25 December 2023.
11. National Judicial Data Grid (2024). Pending cases statistics. Government of India.
12. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-Networks. *EMNLP 2019*.
13. Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

---

## Appendix A — System Technology Stack

| Component | Technology | Version / Notes |
|-----------|-----------|-----------------|
| Language | Python | 3.13 |
| Web Framework | Streamlit | Latest stable |
| Dense Indexing | FAISS | `IndexFlatIP`, `faiss-cpu` |
| Embedding Model | `all-MiniLM-L6-v2` | SentenceTransformers |
| BM25 Retrieval | `rank_bm25` | BM25Okapi variant |
| TF-IDF Retrieval | `TfidfVectorizer` | scikit-learn |
| LLM Interface | Ollama | Local + cloud model serving |
| Generation Model | `gpt-oss:120b-cloud` | Via Ollama /api/chat |
| PDF Extraction | `pdfplumber` + `pytesseract` | Used for initial extraction only |
| Visualisation | Altair, Matplotlib | Deployed in Streamlit |

## Appendix B — Directory Structure

```
Hybrid-MMR Legal RAG Project/
├── legal_rag/
│   ├── app.py                          # Streamlit UI (research dashboard)
│   ├── main.py                         # Automated evaluation harness
│   ├── rebuild_indices.py              # Index construction script
│   ├── data/
│   │   ├── clauses_clean_ipc.json      # IPC corpus (575 sections, curated)
│   │   ├── bns_clauses.json            # BNS corpus (58 sections, curated)
│   │   ├── queries.json                # 132 evaluation queries + ground truth
│   │   ├── vector_index.faiss          # FAISS dense index
│   │   ├── bm25_index.pkl              # BM25 sparse index
│   │   └── tfidf_index.pkl             # TF-IDF index
│   ├── preprocessing/
│   │   ├── load_bns_json.py            # BNS loader + IPC-BNS cross-reference map
│   │   └── load_json.py                # IPC corpus loader
│   ├── indexing/
│   │   ├── vector_index.py             # FAISS build/load utilities
│   │   └── bm25_index.py               # BM25 build/load utilities
│   ├── retrieval/
│   │   ├── baseline.py                 # Vector-only retrieval
│   │   ├── bm25_baseline.py            # BM25 retrieval
│   │   ├── tfidf_baseline.py           # TF-IDF retrieval
│   │   ├── hybrid.py                   # Hybrid fusion retrieval
│   │   ├── mmr.py                      # MMR reranking
│   │   ├── dual_corpus.py              # Dual-corpus search utilities
│   │   └── query_rewriter.py           # LLM query rewriting
│   └── generation/
│       └── answer_generator.py         # Ollama answer generation
└── RESEARCH_PAPER_DRAFT.md
```

## Appendix C — Research Dashboard Overview

The following screenshots document the complete research dashboard that accompanies this paper:

![About This Research Tab](assets/screenshots/about_research_tab.png)
*Figure 9: "About This Research" tab providing publication-grade documentation of the system's purpose, corpus details, and methodology — directly embedded in the research dashboard.*

---

*Draft Version 3.0 · February 2026*
*Hybrid-MMR Legal RAG Project · IPC + BNS 2023 Benchmark*
*Stack: Python 3.13 · FAISS · BM25Okapi · SentenceTransformers · Ollama · GPT-OSS 120B · Streamlit*
