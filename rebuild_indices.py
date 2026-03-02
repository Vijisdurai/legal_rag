"""
rebuild_indices.py
------------------
Rebuild FAISS, BM25, and TF-IDF indices from the combined IPC+BNS corpus.
Run this whenever the corpus changes (e.g. after adding BNS clauses).
"""
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from retrieval.dual_corpus import load_combined_corpus

# ── Load combined corpus ──────────────────────────────────────────────────────
print("Loading combined IPC + BNS corpus...")
combined = load_combined_corpus()
ipc = [c for c in combined if c.get('corpus', 'ipc') == 'ipc']
bns = [c for c in combined if c.get('corpus', 'ipc') == 'bns']
print(f"  IPC: {len(ipc)} | BNS: {len(bns)} | Total: {len(combined)}")

# ── Remove stale index files ──────────────────────────────────────────────────
import glob
stale = glob.glob('data/vector_index.faiss') + \
        glob.glob('data/embeddings.npy')    + \
        glob.glob('data/bm25_index.pkl')    + \
        glob.glob('data/tfidf_index.pkl')
for f in stale:
    os.remove(f)
    print(f"  Removed stale: {f}")

# ── Rebuild FAISS + embeddings ────────────────────────────────────────────────
print("\n[1/3] Building FAISS vector index...")
from indexing.vector_index import build_index
index, embeddings, model = build_index(combined)
print(f"  FAISS index size   : {index.ntotal}")
print(f"  Embeddings shape   : {embeddings.shape}")

# ── Rebuild BM25 ──────────────────────────────────────────────────────────────
print("\n[2/3] Building BM25 index...")
from indexing.bm25_index import build_bm25
bm25, _ = build_bm25(combined)
print(f"  BM25 vocab size    : {len(combined)}")

# ── Rebuild TF-IDF ────────────────────────────────────────────────────────────
print("\n[3/3] Building TF-IDF index...")
from retrieval.tfidf_baseline import build_tfidf
vectorizer, mat = build_tfidf(combined)
print(f"  TF-IDF matrix shape: {mat.shape}")

print("\nAll indices rebuilt successfully from combined corpus.")
print(f"Corpus: {len(combined)} clauses ({len(ipc)} IPC + {len(bns)} BNS)")
