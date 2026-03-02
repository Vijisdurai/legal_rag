"""
mmr.py
------
Maximal Marginal Relevance (MMR) reranking over hybrid retrieval candidates.

MMR formula (iterative greedy):
    MMR = argmax [ lambda * relevance(c) - (1 - lambda) * max_sim(c, S) ]

Where:
    lambda    = 0.7  (controls relevance vs. diversity trade-off)
    relevance = hybrid_score of candidate (from Stage 4)
    max_sim   = max cosine similarity between candidate and already-selected set S
    S         = set of already-selected results

Returns top 5 diverse + relevant results.
"""

import numpy as np
from retrieval.hybrid import hybrid_search, run_hybrid
from indexing.vector_index import get_or_build_index, load_clauses
from indexing.bm25_index import get_or_build_bm25

# MMR parameters
LAMBDA = 0.9        # relevance weight (1 - LAMBDA = diversity weight)
FINAL_TOP_K = 5     # number of results to return after MMR


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized vectors.
    Since embeddings are already L2-normalized (from SentenceTransformers),
    this reduces to a dot product.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_rerank(candidates: list[dict],
               lam: float = LAMBDA,
               top_k: int = FINAL_TOP_K) -> list[dict]:
    """
    Apply MMR to a list of hybrid retrieval candidates.

    Args:
        candidates: List of candidate dicts from hybrid_search().
                    Each must have: 'hybrid_score', 'embedding' (np.ndarray).
        lam:        Lambda trade-off (higher = more relevance, lower = more diversity).
        top_k:      Number of results to return.

    Returns:
        List of top_k reranked result dicts with added 'mmr_score' and 'mmr_rank'.
    """
    if not candidates:
        return []

    top_k = min(top_k, len(candidates))
    remaining = list(range(len(candidates)))  # indices into candidates
    selected = []                              # indices of selected candidates

    for _ in range(top_k):
        best_idx = None
        best_score = -np.inf

        for i in remaining:
            # Relevance term
            relevance = candidates[i]['hybrid_score']

            # Diversity term: max cosine sim to already-selected set
            if not selected:
                diversity_penalty = 0.0
            else:
                sims = [
                    cosine_similarity(
                        candidates[i]['embedding'],
                        candidates[j]['embedding']
                    )
                    for j in selected
                ]
                diversity_penalty = max(sims)

            # MMR score
            mmr_score = lam * relevance - (1 - lam) * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)

    # Build result list
    results = []
    for mmr_rank, idx in enumerate(selected, start=1):
        c = candidates[idx].copy()
        c.pop('embedding', None)        # don't expose raw embeddings in output
        c['mmr_rank'] = mmr_rank
        c['mmr_score'] = float(
            lam * c['hybrid_score'] - 0.0   # final score stored for display
        )
        results.append(c)

    return results


def run_hybrid_mmr(query: str,
                   candidates_k: int = 20,
                   final_k: int = FINAL_TOP_K,
                   lam: float = LAMBDA) -> list[dict]:
    """
    Full pipeline: hybrid retrieval → MMR reranking.
    """
    candidates = run_hybrid(query, top_k=candidates_k)
    return mmr_rerank(candidates, lam=lam, top_k=final_k)


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
    print(f"HYBRID + MMR RETRIEVAL TEST  (lambda={LAMBDA})")
    print("=" * 70)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 55)

        # Stage 4: hybrid top-20
        candidates = hybrid_search(
            query, clauses, index, embeddings, model, bm25, top_k=20
        )

        # Stage 5: MMR top-5
        results = mmr_rerank(candidates, lam=LAMBDA, top_k=5)

        for r in results:
            print(f"  MMR#{r['mmr_rank']:2}  Sec {r['section_number']:>4}  "
                  f"hybrid={r['hybrid_score']:.4f}  "
                  f"[vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f}]  "
                  f"| {r['snippet'][:55]}...")
