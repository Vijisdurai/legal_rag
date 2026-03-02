"""
Naive Query Expansion Evaluation Using WordNet
Standalone version to avoid import errors
"""

import sys
import os
import json
import numpy as np
import nltk
from nltk.corpus import wordnet

def naive_rewrite(query: str) -> str:
    """Expands a query using WordNet synonyms."""
    words = query.split()
    expanded_words = set([w.lower() for w in words])
    
    for word in words:
        if len(word) < 4: continue
        
        synsets = wordnet.synsets(word)
        for syn in synsets[:2]: 
            for lemma in syn.lemmas()[:3]: 
                term = lemma.name().replace('_', ' ')
                expanded_words.add(term.lower())
                    
    return " ".join(list(expanded_words))

def recall_at_k(retrieved_sections, relevant_sections, k=5):
    retrieved_ids = [s["section_number"] for s in retrieved_sections[:k]]
    return 1.0 if any(r in retrieved_ids for r in relevant_sections) else 0.0

def run_naive_eval():
    print("Loading data...")
    legal_rag_dir = os.path.dirname(__file__)
    data_dir = os.path.join(legal_rag_dir, "data")
    
    with open(os.path.join(data_dir, "queries.json"), "r", encoding="utf-8") as f:
        test_queries = json.load(f)
        
    with open(os.path.join(data_dir, "clauses_augmented.json"), "r", encoding="utf-8") as f:
        combined_corpus_sections = json.load(f)

    with open(os.path.join(data_dir, "bns_clauses.json"), "r", encoding="utf-8") as f:
        bns_sections = json.load(f)
        
    combined_corpus_sections.extend(bns_sections)

    # We only care about BM25 and Vector, not hybrid, so we can mock the retrieval minimally
    # Let's import the specific indexing loaders to be safe
    import pickle
    import faiss
    from sentence_transformers import SentenceTransformer
    
    # Load BM25
    from rank_bm25 import BM25Okapi
    with open(os.path.join(data_dir, "bm25_index.pkl"), 'rb') as f:
        corpus_tokens = pickle.load(f)
    bm25_index = BM25Okapi(corpus_tokens)
        
    # Load Vector
    index = faiss.read_index(os.path.join(data_dir, 'vector_index.faiss'))
    model = SentenceTransformer('all-MiniLM-L6-v2')

    systems = ["Vector-Only", "BM25-Only"]
    sys_results_original = {s: [] for s in systems}
    sys_results_naive = {s: [] for s in systems}
    
    print(f"\nEvaluating {len(test_queries)} queries...")
    
    for i, q in enumerate(test_queries):
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(test_queries)}")
        
        orig_q = q["query"]
        expanded_q = naive_rewrite(orig_q)
        rel_secs = q["relevant_sections"]
        
        for query_text, results_dict in [(orig_q, sys_results_original), (expanded_q, sys_results_naive)]:
            # BM25 Retrieval
            tokenized_query = query_text.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:5]
            bm25_retrieved = [combined_corpus_sections[idx] for idx in top_bm25_idx]
            
            # Vector Retrieval
            query_embedding = model.encode([query_text], normalize_embeddings=True)
            D, I = index.search(query_embedding, 5)
            vector_retrieved = [combined_corpus_sections[idx] for idx in I[0]]
            
            # Save results
            results_dict["BM25-Only"].append(recall_at_k(bm25_retrieved, rel_secs))
            results_dict["Vector-Only"].append(recall_at_k(vector_retrieved, rel_secs))

    print("\n====================")
    print("Results (Recall@5):")
    for s in systems:
        acc_orig = sum(sys_results_original[s]) / len(test_queries)
        acc_naive = sum(sys_results_naive[s]) / len(test_queries)
        diff = acc_naive - acc_orig
        print(f"  {s}")
        print(f"    Original : {acc_orig:.4f}")
        print(f"    Naive QR : {acc_naive:.4f}")
        pct_change = (diff/acc_orig)*100 if acc_orig > 0 else 0.0
        print(f"    Delta    : {diff:+.4f} ({pct_change:+.1f}%)")
    print("====================")

    out_data = {
        s: {
            "Original R@5": sum(sys_results_original[s]) / len(test_queries),
            "Naive QR R@5": sum(sys_results_naive[s]) / len(test_queries)
        } for s in systems
    }
    
    with open(os.path.join(data_dir, "naive_qr_results.json"), "w") as f:
        json.dump(out_data, f, indent=4)
    print("\nSaved detailed results to data/naive_qr_results.json")

if __name__ == "__main__":
    run_naive_eval()
