"""
retrieval/cross_encoder_rerank.py
----------------------------------
Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.

Implements Stage 2 of the 3-stage pipeline:
  1. Hybrid (BM25 + vector) → top-50 candidates
  2. Cross-encoder → rerank top-50 → top-20          ← THIS FILE
  3. MMR diversity → final top-5

Cross-encoders jointly encode (query, document) pairs and produce
much sharper relevance scores than bi-encoders, at higher compute cost.
We apply them only to the top-50 hybrid candidates for efficiency.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded singleton
_cross_encoder = None


def get_cross_encoder():
    """Load cross-encoder model (lazy, cached)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"[cross_encoder] Loading {CROSS_ENCODER_MODEL}...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
        print("[cross_encoder] Ready.")
    return _cross_encoder


def cross_encoder_rerank(query: str,
                         candidates: list[dict],
                         top_k: int = 20) -> list[dict]:
    """
    Rerank candidates using cross-encoder relevance scores.

    Args:
        query:      User query string.
        candidates: List of candidate dicts (from hybrid_search).
        top_k:      Number to return after reranking.

    Returns:
        Top-k candidates sorted by cross-encoder score (descending),
        each enriched with a 'cross_score' field.
    """
    if not candidates:
        return []

    ce = get_cross_encoder()

    # Build (query, passage) pairs
    pairs = [(query, c['text'][:512]) for c in candidates]
    scores = ce.predict(pairs)

    # Attach scores and sort
    for c, score in zip(candidates, scores):
        c['cross_score'] = float(score)

    reranked = sorted(candidates, key=lambda x: x['cross_score'], reverse=True)
    return reranked[:top_k]


def hybrid_cross_mmr_pipeline(query: str,
                               clauses: list[dict],
                               index,
                               embeddings: np.ndarray,
                               model,
                               bm25,
                               lam: float = 0.9,
                               final_top_k: int = 5) -> list[dict]:
    """
    Full 3-stage pipeline:
      1. Hybrid BM25+Vector → top-50
      2. Cross-encoder rerank → top-20
      3. MMR diversity → top-5

    Args:
        query, clauses, index, embeddings, model, bm25: retrieval components
        lam:           MMR lambda (relevance weight)
        final_top_k:   Final results to return

    Returns:
        List of top-k diversified, cross-encoder-scored results.
    """
    from retrieval.hybrid import hybrid_search
    from retrieval.mmr import mmr_rerank

    # Stage 1
    candidates = hybrid_search(query, clauses, index, embeddings,
                                model, bm25, top_k=50)

    # Stage 2 — cross-encoder rerank
    reranked = cross_encoder_rerank(query, candidates, top_k=20)

    # Override hybrid_score with cross_score for MMR relevance
    for r in reranked:
        r['hybrid_score'] = r['cross_score']

    # Stage 3 — MMR
    final = mmr_rerank(reranked, lam=lam, top_k=final_top_k)
    return final


if __name__ == '__main__':
    from indexing.vector_index import get_or_build_index, load_clauses
    from indexing.bm25_index import get_or_build_bm25

    clauses = load_clauses()
    index, embeddings, model = get_or_build_index(clauses)
    bm25, _ = get_or_build_bm25(clauses)

    queries = [
        "punishment for murder",
        "right of private defence",
        "cruelty by husband or relatives",
    ]
    for q in queries:
        print(f'\nQuery: "{q}"')
        results = hybrid_cross_mmr_pipeline(q, clauses, index, embeddings, model, bm25)
        for r in results:
            print(f"  MMR#{r['mmr_rank']}  Sec {r['section_number']:>4}  "
                  f"cross={r.get('cross_score',0):.4f}  {r['snippet'][:60]}...")
