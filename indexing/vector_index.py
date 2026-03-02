"""
vector_index.py
---------------
Builds a FAISS dense vector index over IPC clauses using SentenceTransformers.

Model: all-MiniLM-L6-v2 (fast, 384-dim embeddings)
Index: FAISS IndexFlatIP on L2-normalized vectors (= cosine similarity)

Saves:
  data/vector_index.faiss  — FAISS index file
  data/embeddings.npy      — Raw embeddings (for MMR diversity computation)
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths relative to legal_rag/
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CLAUSES_PATH = os.path.join(BASE_DIR, 'data', 'clauses_augmented.json')
INDEX_PATH = os.path.join(BASE_DIR, 'data', 'vector_index.faiss')
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'data', 'embeddings.npy')

MODEL_NAME = 'all-MiniLM-L6-v2'


def load_clauses(path: str = CLAUSES_PATH) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_index(clauses: list[dict],
                model_name: str = MODEL_NAME,
                save: bool = True) -> tuple:
    """
    Encode all clauses and build a FAISS index.

    Args:
        clauses:    List of clause dicts.
        model_name: SentenceTransformer model name.
        save:       Whether to save index + embeddings to disk.

    Returns:
        (index, embeddings, model)
    """
    print(f"[vector_index] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c['text'] for c in clauses]
    print(f"[vector_index] Encoding {len(texts)} clauses...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True  # L2-normalize → IndexFlatIP = cosine similarity
    )
    embeddings = np.array(embeddings, dtype='float32')

    dim = embeddings.shape[1]
    print(f"[vector_index] Building FAISS IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[vector_index] Index size: {index.ntotal} vectors")

    if save:
        faiss.write_index(index, INDEX_PATH)
        np.save(EMBEDDINGS_PATH, embeddings)
        print(f"[vector_index] Saved index → {INDEX_PATH}")
        print(f"[vector_index] Saved embeddings → {EMBEDDINGS_PATH}")

    return index, embeddings, model


def load_index() -> tuple:
    """
    Load pre-built FAISS index and embeddings from disk.

    Returns:
        (index, embeddings, model)
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run build_index() first.")

    print(f"[vector_index] Loading FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)

    print(f"[vector_index] Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"[vector_index] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    return index, embeddings, model


def get_or_build_index(clauses: list[dict] = None) -> tuple:
    """
    Load index if it exists, otherwise build and save it.

    Returns:
        (index, embeddings, model)
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        return load_index()
    if clauses is None:
        clauses = load_clauses()
    return build_index(clauses)


if __name__ == '__main__':
    clauses = load_clauses()
    index, embeddings, model = build_index(clauses)
    print(f"\n✅ Vector index built successfully.")
    print(f"   Clauses: {len(clauses)}, Embedding dim: {embeddings.shape[1]}")
