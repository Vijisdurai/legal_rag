import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from retrieval.dual_corpus import load_combined_corpus
from indexing.vector_index import get_or_build_index
from indexing.bm25_index import get_or_build_bm25

from retrieval.baseline import vector_search
from retrieval.bm25_baseline import bm25_search
from retrieval.hybrid import hybrid_search
from retrieval.mmr import mmr_rerank

def main():
    clauses = load_combined_corpus()
    index, embeddings, model = get_or_build_index(clauses)
    bm25, _ = get_or_build_bm25(clauses)

    with open("data/queries.json", "r") as f:
        queries = json.load(f)

    # Make a mapping for quick text lookup
    clause_texts = { c['section_number']: c['text'][:200] + "..." for c in clauses }

    def get_clause_text(section_num):
        return clause_texts.get(section_num, "Not found")

    def check_failure(results_sections, ground_truths):
        top_5_sections = results_sections[:5]
        for gt in ground_truths:
            if gt in top_5_sections:
                return False # success
        return True # failure

    failures = {
        'dense': [],
        'bm25': [],
        'hybrid': [],
        'qr': []
    }

    for q in queries:
        original_query = q['query']
        rewritten_query = q.get('rewritten_query', original_query)
        tier = q.get('tier', 'Unknown')
        
        # ground truths
        ground_truths = q.get('relevant_sections', [])
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if not ground_truths and 'relevant_section' in q:
            ground_truths = [q['relevant_section']]
        
        if not ground_truths:
            continue
            
        gt_section = ground_truths[0]
        gt_text = get_clause_text(gt_section)
        
        # 1. Dense Failure
        res_dense = vector_search(original_query, clauses, index, model, top_k=5)
        sec_dense = [r['section_number'] for r in res_dense]
        if check_failure(sec_dense, ground_truths) and len(failures['dense']) < 5:
            top_wrong = sec_dense[0] if sec_dense else "None"
            failures['dense'].append({
                'query': original_query,
                'tier': tier,
                'gt': f"§{gt_section}: {gt_text}",
                'retrieved': f"§{top_wrong}: {get_clause_text(top_wrong)}"
            })
            
        # 2. BM25 Failure
        res_bm25 = bm25_search(original_query, clauses, bm25, top_k=5)
        sec_bm25 = [r['section_number'] for r in res_bm25]
        if check_failure(sec_bm25, ground_truths) and len(failures['bm25']) < 5:
            top_wrong = sec_bm25[0] if sec_bm25 else "None"
            failures['bm25'].append({
                'query': original_query,
                'tier': tier,
                'gt': f"§{gt_section}: {gt_text}",
                'retrieved': f"§{top_wrong}: {get_clause_text(top_wrong)}"
            })

        # 3. Hybrid+MMR Failure on original query
        hyb_cand = hybrid_search(original_query, clauses, index, embeddings, model, bm25, top_k=50)
        res_hyb = mmr_rerank(hyb_cand, top_k=5)
        sec_hyb = [r['section_number'] for r in res_hyb]
        if check_failure(sec_hyb, ground_truths) and len(failures['hybrid']) < 5:
            top_wrong = sec_hyb[0] if sec_hyb else "None"
            failures['hybrid'].append({
                'query': original_query,
                'tier': tier,
                'gt': f"§{gt_section}: {gt_text}",
                'retrieved': f"§{top_wrong}: {get_clause_text(top_wrong)}"
            })

        # 4. QR Failure (Hybrid+MMR fails even with rewritten query)
        qr_cand = hybrid_search(rewritten_query, clauses, index, embeddings, model, bm25, top_k=50)
        res_qr = mmr_rerank(qr_cand, top_k=5)
        sec_qr = [r['section_number'] for r in res_qr]
        if check_failure(sec_qr, ground_truths) and len(failures['qr']) < 5 and original_query != rewritten_query:
            top_wrong = sec_qr[0] if sec_qr else "None"
            failures['qr'].append({
                'query': original_query,
                'rewritten': rewritten_query,
                'tier': tier,
                'gt': f"§{gt_section}: {gt_text}",
                'retrieved': f"§{top_wrong}: {get_clause_text(top_wrong)}"
            })

    print(json.dumps(failures, indent=2))

if __name__ == "__main__":
    main()
