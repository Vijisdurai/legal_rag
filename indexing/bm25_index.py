"""
bm25_index.py
-------------
Builds a BM25 sparse index over IPC clauses using rank_bm25 (BM25Okapi).

Tokenization: simple whitespace + lowercase (no stemming, no stopwords removal)
to keep the system research-transparent and reproducible.

Usage:
    from indexing.bm25_index import build_bm25, bm25_scores

    bm25, clauses = build_bm25()
    scores = bm25_scores(bm25, "punishment for murder")
"""

import os
import json
import re
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CLAUSES_PATH = os.path.join(BASE_DIR, 'data', 'clauses_augmented.json')
BM25_PATH = os.path.join(BASE_DIR, 'data', 'bm25_index.pkl')


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.
    Strips punctuation and splits on non-alphanumeric characters.
    """
    text = text.lower()
    tokens = re.split(r'[^a-z0-9]+', text)
    return [t for t in tokens if len(t) > 1]  # drop single chars


def load_clauses(path: str = CLAUSES_PATH) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_bm25(clauses: list[dict] = None, save: bool = True) -> tuple:
    """
    Build BM25Okapi index from clause texts.

    Args:
        clauses: list of clause dicts. Loaded from disk if None.
        save:    Whether to pickle the index to disk.

    Returns:
        (bm25, clauses)
    """
    if clauses is None:
        clauses = load_clauses()

    print(f"[bm25_index] Tokenizing {len(clauses)} clauses...")
    tokenized_corpus = [tokenize(c['text']) for c in clauses]

    print("[bm25_index] Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    if save:
        with open(BM25_PATH, 'wb') as f:
            pickle.dump({'bm25': bm25, 'tokenized_corpus': tokenized_corpus}, f)
        print(f"[bm25_index] Saved BM25 index → {BM25_PATH}")

    return bm25, clauses


def load_bm25() -> tuple:
    """
    Load pre-built BM25 index from disk.

    Returns:
        (bm25, clauses)
    """
    if not os.path.exists(BM25_PATH):
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. Run build_bm25() first."
        )
    with open(BM25_PATH, 'rb') as f:
        data = pickle.load(f)
    clauses = load_clauses()
    print(f"[bm25_index] Loaded BM25 index from {BM25_PATH}")
    return data['bm25'], clauses


def get_or_build_bm25(clauses: list[dict] = None) -> tuple:
    """Load BM25 if exists, else build and save."""
    if os.path.exists(BM25_PATH):
        return load_bm25()
    return build_bm25(clauses)


def bm25_scores(bm25: BM25Okapi, query: str) -> np.ndarray:
    """
    Compute BM25 scores for all clauses given a query.

    Args:
        bm25:  Built BM25Okapi object.
        query: Natural language query string.

    Returns:
        numpy array of shape (n_clauses,) with BM25 scores.
    """
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    return np.array(scores, dtype='float32')


def bm25_top_k(bm25: BM25Okapi,
               clauses: list[dict],
               query: str,
               top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k clauses by BM25 score.

    Returns:
        List of result dicts: {rank, section_number, text, score, snippet}
    """
    scores = bm25_scores(bm25, query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        clause = clauses[idx]
        results.append({
            "rank": rank,
            "section_number": clause['section_number'],
            "text": clause['text'],
            "score": float(scores[idx]),
            "snippet": clause['text'][:200]
        })
    return results


if __name__ == '__main__':
    clauses = load_clauses()
    bm25, clauses = build_bm25(clauses)

    test_queries = [
        "punishment for murder",
        "cheating and fraud",
        "right of private defence",
        "rape and sexual assault",
        "criminal conspiracy",
    ]

    print("\n" + "=" * 65)
    print("BM25 RETRIEVAL TEST")
    print("=" * 65)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)
        results = bm25_top_k(bm25, clauses, query, top_k=5)
        for r in results:
            print(f"  #{r['rank']} Section {r['section_number']:>4}  "
                  f"score={r['score']:.4f}  |  {r['snippet'][:80]}...")
