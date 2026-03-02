"""
retrieval/bm25_baseline.py
--------------------------
BM25-only retrieval baseline. Uses the pre-built BM25Okapi index.
No vector scores — purely keyword matching.
"""

import os
import sys
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

from indexing.bm25_index import get_or_build_bm25, load_clauses


def bm25_search(query: str,
                clauses: list[dict],
                bm25,
                top_k: int = 5,
                corpus_filter: str = "both") -> list[dict]:
    """
    Pure BM25 retrieval with optional corpus post-filtering.
    `clauses` must match the corpus BM25 was built on (same length).
    """
    tokens = query.lower().split()
    scores = np.array(bm25.get_scores(tokens), dtype='float32')

    # Over-fetch to allow for corpus filtering
    fetch_k = min(len(clauses), top_k * 6 if corpus_filter != "both" else top_k)
    top_indices = np.argsort(scores)[::-1][:fetch_k]

    results = []
    rank = 1
    for idx in top_indices:
        c = clauses[idx]
        if corpus_filter != "both" and c.get('corpus', 'ipc') != corpus_filter:
            continue
        results.append({
            "rank":           rank,
            "section_number": c["section_number"],
            "title":          c.get("title", ""),
            "chapter":        c.get("chapter", ""),
            "text":           c["text"],
            "bm25_score":     float(scores[idx]),
            "score":          float(scores[idx]),
            "snippet":        c["text"][:80],
            "corpus":         c.get('corpus', 'ipc'),
            "ipc_equivalent": c.get('ipc_equivalent'),
        })
        rank += 1
        if rank > top_k:
            break
    return results


if __name__ == '__main__':
    from indexing.vector_index import load_clauses as vc_load
    clauses = vc_load()
    bm25, _ = get_or_build_bm25(clauses)

    test_queries = [
        "punishment for murder",
        "right of private defence",
        "criminal conspiracy definition",
    ]
    for q in test_queries:
        print(f'\nQuery: "{q}"')
        for r in bm25_search(q, clauses, bm25, top_k=3):
            print(f"  #{r['rank']} Sec {r['section_number']:>4}  bm25={r['bm25_score']:.4f}  {r['snippet'][:60]}...")
