"""
baseline.py
-----------
Vector-only (dense semantic) retrieval baseline using FAISS.

Given a natural language query:
  1. Encode query with the same SentenceTransformer model
  2. Search FAISS index for top-k by cosine similarity
  3. Return ranked results with metadata
"""

import numpy as np
from indexing.vector_index import get_or_build_index, load_clauses


def vector_search(query: str,
                  clauses: list[dict],
                  index,
                  model,
                  top_k: int = 5,
                  corpus_filter: str = "both") -> list[dict]:
    """
    Perform vector-only retrieval for a query.

    IMPORTANT: `clauses` must be the FULL corpus that the FAISS index was built
    on (length must match index.ntotal). FAISS returns absolute indices into this
    list. To restrict results to a corpus subset, use `corpus_filter`.

    Args:
        query:         Natural language legal query.
        clauses:       Full clause list matching the FAISS index order.
        index:         Loaded FAISS index (ntotal == len(clauses)).
        model:         Loaded SentenceTransformer model.
        top_k:         Number of results to return after filtering.
        corpus_filter: 'ipc' | 'bns' | 'both' — filters results by corpus tag.

    Returns:
        List of result dicts with rank, section_number, text, score, snippet,
        corpus, and ipc_equivalent fields.
    """
    # Encode + normalize query
    query_vec = model.encode(
        [query],
        normalize_embeddings=True
    ).astype('float32')

    # Search over-fetch to account for corpus filtering removing results
    fetch_k = min(len(clauses), top_k * 6 if corpus_filter != "both" else top_k * 2)
    scores, indices = index.search(query_vec, fetch_k)
    scores  = scores[0]
    indices = indices[0]

    results = []
    rank    = 1
    for idx, score in zip(indices, scores):
        if idx == -1 or idx >= len(clauses):
            continue
        clause = clauses[idx]

        # Corpus post-filter
        if corpus_filter != "both":
            if clause.get('corpus', 'ipc') != corpus_filter:
                continue

        results.append({
            "rank":           rank,
            "section_number": clause['section_number'],
            "title":          clause.get('title', ''),
            "text":           clause['text'],
            "score":          float(score),
            "snippet":        clause['text'][:200],
            "corpus":         clause.get('corpus', 'ipc'),
            "ipc_equivalent": clause.get('ipc_equivalent'),
        })
        rank += 1
        if rank > top_k:
            break

    return results


def run_baseline(query: str, top_k: int = 5) -> list[dict]:
    """
    Convenience function: load index + run vector search.
    Builds index from scratch if not yet built.
    """
    clauses = load_clauses()
    index, embeddings, model = get_or_build_index(clauses)
    return vector_search(query, clauses, index, model, top_k)


if __name__ == '__main__':
    test_queries = [
        "punishment for murder",
        "cheating and fraud",
        "right of private defence",
        "rape and sexual assault",
        "criminal conspiracy",
    ]

    print("=" * 65)
    print("VECTOR-ONLY BASELINE RETRIEVAL TEST")
    print("=" * 65)

    clauses = load_clauses()
    index, embeddings, model = get_or_build_index(clauses)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)
        results = vector_search(query, clauses, index, model, top_k=5)
        for r in results:
            print(f"  #{r['rank']} Section {r['section_number']:>4}  "
                  f"score={r['score']:.4f}  |  {r['snippet'][:80]}...")
