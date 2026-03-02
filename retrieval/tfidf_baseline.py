"""
retrieval/tfidf_baseline.py
---------------------------
TF-IDF + cosine similarity baseline retrieval.
Classic sparse IR baseline for ablation study.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

TFIDF_PATH = os.path.join(BASE_DIR, 'data', 'tfidf_index.pkl')


def build_tfidf(clauses: list[dict], save: bool = True):
    """Build TF-IDF index over clause texts."""
    texts = [c['text'] for c in clauses]
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        lowercase=True,
        ngram_range=(1, 2),     # unigrams + bigrams
        max_df=0.85,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"[tfidf] Built TF-IDF matrix: {tfidf_matrix.shape}")

    if save:
        with open(TFIDF_PATH, 'wb') as f:
            pickle.dump((vectorizer, tfidf_matrix), f)
        print(f"[tfidf] Saved -> {TFIDF_PATH}")

    return vectorizer, tfidf_matrix


def load_tfidf():
    """Load pre-built TF-IDF index."""
    with open(TFIDF_PATH, 'rb') as f:
        return pickle.load(f)


def get_or_build_tfidf(clauses: list[dict]):
    """Load if exists, else build."""
    if os.path.exists(TFIDF_PATH):
        print(f"[tfidf] Loading from {TFIDF_PATH}")
        return load_tfidf()
    return build_tfidf(clauses)


def tfidf_search(query: str,
                 clauses: list[dict],
                 vectorizer: TfidfVectorizer,
                 tfidf_matrix,
                 top_k: int = 5,
                 corpus_filter: str = "both") -> list[dict]:
    """
    TF-IDF cosine similarity retrieval with optional corpus post-filtering.
    `clauses` must match the corpus TF-IDF was built on (same length).
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

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
            "tfidf_score":    float(scores[idx]),
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
    from indexing.vector_index import load_clauses
    clauses = load_clauses()
    vectorizer, tfidf_matrix = build_tfidf(clauses)

    test_queries = [
        "punishment for murder",
        "right of private defence",
        "criminal conspiracy definition",
    ]
    for q in test_queries:
        print(f'\nQuery: "{q}"')
        for r in tfidf_search(q, clauses, vectorizer, tfidf_matrix, top_k=3):
            print(f"  #{r['rank']} Sec {r['section_number']:>4}  tfidf={r['tfidf_score']:.4f}  {r['snippet'][:60]}...")
