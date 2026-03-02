"""
hybrid.py
---------
Hybrid retrieval combining dense vector (FAISS) + sparse BM25 scores.

Algorithm:
  1. Compute cosine similarity scores for all clauses (vector)
  2. Compute BM25 scores for all clauses
  3. Min-max normalize both score arrays to [0, 1]
  4. Fuse: final_score = 0.6 * vector_score + 0.4 * bm25_score
  5. Return top 20 candidates (for MMR reranking in Stage 5)
"""

import numpy as np
from indexing.vector_index import get_or_build_index, load_clauses
from indexing.bm25_index import get_or_build_bm25, bm25_scores

# Fusion weights (must sum to 1.0)
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Number of candidates to return for MMR reranking
TOP_CANDIDATES = 50

# Administrative/preamble sections to exclude from retrieval
# (Only truly non-substantive: title, jurisdiction, savings clause)
PREAMBLE_SECTIONS = {str(i) for i in range(1, 6)}


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize a score array to [0, 1] using min-max scaling.
    Returns zeros if all scores are equal (avoids division by zero).
    """
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def hybrid_search(query: str,
                  clauses: list[dict],
                  index,
                  embeddings: np.ndarray,
                  model,
                  bm25,
                  top_k: int = TOP_CANDIDATES,
                  vector_weight: float = VECTOR_WEIGHT,
                  bm25_weight: float = BM25_WEIGHT,
                  corpus_filter: str = "both") -> list[dict]:
    """
    Perform hybrid retrieval for a query.

    Args:
        query:          Natural language legal query.
        clauses:        List of clause dicts (same order as index).
        index:          Loaded FAISS index.
        embeddings:     Array of shape (n_clauses, dim) — stored clause embeddings.
        model:          Loaded SentenceTransformer model.
        bm25:           Loaded BM25Okapi object.
        top_k:          Number of candidates to return.
        vector_weight:  Weight for vector score (default 0.6).
        bm25_weight:    Weight for BM25 score (default 0.4).

    Returns:
        List of top_k candidate dicts:
        {
          "rank": int,
          "index": int,               # original clause index (needed by MMR)
          "section_number": str,
          "text": str,
          "snippet": str,
          "vector_score": float,      # normalized [0,1]
          "bm25_score": float,        # normalized [0,1]
          "hybrid_score": float,      # fused score
          "embedding": np.ndarray     # clause embedding (needed by MMR)
        }
    """
    # ── Step 1: Vector scores (cosine sim for all clauses) ──────────────────
    query_vec = model.encode([query], normalize_embeddings=True).astype('float32')
    # Use FAISS to get all scores at once
    raw_vector_scores, _ = index.search(query_vec, len(clauses))
    vector_scores = raw_vector_scores[0]  # shape (n_clauses,)

    # ── Step 2: BM25 scores ─────────────────────────────────────────────────
    raw_bm25_scores = bm25_scores(bm25, query)  # shape (n_bm25_docs,)

    # ── Shape safety: indices may have been built on a different corpus size ─
    n_clauses = len(clauses)
    n_vec     = len(vector_scores)
    n_bm25    = len(raw_bm25_scores)

    if n_vec != n_clauses or n_bm25 != n_clauses:
        # Corpus changed without rebuilding indices — emit a clear warning and
        # align arrays to the actual clause list length.
        import warnings
        warnings.warn(
            f"[hybrid] Index size mismatch: clauses={n_clauses}, "
            f"FAISS={n_vec}, BM25={n_bm25}. "
            "Run `python rebuild_indices.py` to fix. "
            "Falling back to vector-only search.",
            RuntimeWarning, stacklevel=2
        )
        # Safest fallback: use only vector scores, pad BM25 to match
        min_len = min(n_vec, n_clauses)
        vector_scores = vector_scores[:min_len]
        # Create a zero BM25 array so fusion still runs (vector-only effectively)
        raw_bm25_scores = np.zeros(min_len, dtype='float32')
        # Also trim clauses reference used downstream
        clauses = clauses[:min_len]

    # ── Step 3: Normalize both to [0, 1] ────────────────────────────────────
    norm_vector = min_max_normalize(vector_scores)
    norm_bm25 = min_max_normalize(raw_bm25_scores)

    # ── Step 4: Weighted fusion ──────────────────────────────────────────────
    fused = vector_weight * norm_vector + bm25_weight * norm_bm25

    # ── Step 5: Top-k candidates (skip administrative preamble sections) ─────
    all_sorted = np.argsort(fused)[::-1]

    results = []
    rank = 1
    for idx in all_sorted:
        if len(results) >= top_k:
            break
        clause = clauses[idx]
        # Skip preamble/administrative sections (Sections 1-14)
        if clause['section_number'] in PREAMBLE_SECTIONS:
            continue
        # Post-filter by corpus
        if corpus_filter != "both" and clause.get('corpus', 'ipc') != corpus_filter:
            continue
        results.append({
            "rank":           rank,
            "index":          int(idx),
            "section_number": clause['section_number'],
            "title":          clause.get('title', ''),
            "text":           clause['text'],
            "snippet":        clause['text'][:200],
            "vector_score":   float(norm_vector[idx]),
            "bm25_score":     float(norm_bm25[idx]),
            "hybrid_score":   float(fused[idx]),
            "embedding":      embeddings[idx],
            "corpus":         clause.get('corpus', 'ipc'),
            "ipc_equivalent": clause.get('ipc_equivalent'),
        })
        rank += 1

    return results


def run_hybrid(query: str, top_k: int = TOP_CANDIDATES) -> list[dict]:
    """
    Convenience function: load all indices + run hybrid search.
    """
    clauses = load_clauses()
    index, embeddings, model = get_or_build_index(clauses)
    bm25, _ = get_or_build_bm25(clauses)
    return hybrid_search(query, clauses, index, embeddings, model, bm25, top_k)


if __name__ == '__main__':
    test_queries = [
        "punishment for murder",
        "cheating and fraud",
        "right of private defence",
        "rape and sexual assault",
        "criminal conspiracy",
    ]

    clauses = load_clauses()
    index, embeddings, model = get_or_build_index(clauses)
    bm25, _ = get_or_build_bm25(clauses)

    print("=" * 70)
    print("HYBRID RETRIEVAL TEST  (top 5 shown from 20 candidates)")
    print(f"Fusion: {VECTOR_WEIGHT} * vector + {BM25_WEIGHT} * BM25")
    print("=" * 70)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 55)
        results = hybrid_search(
            query, clauses, index, embeddings, model, bm25, top_k=20
        )
        for r in results[:5]:
            print(f"  #{r['rank']:2}  Sec {r['section_number']:>4}  "
                  f"hybrid={r['hybrid_score']:.4f}  "
                  f"[vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f}]  "
                  f"| {r['snippet'][:60]}...")
