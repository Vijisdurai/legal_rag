"""
main.py
-------
End-to-end pipeline entry point for the Hybrid-MMR Legal RAG system.

Usage:
  python main.py                        # run full pipeline (build + demo queries)
  python main.py --rebuild              # force rebuild all indices
  python main.py --query "your query"   # single query in all modes
  python main.py --eval                 # run evaluation metrics
  python main.py --ui                   # launch Streamlit UI
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ── Stage runners ──────────────────────────────────────────────────────────────

def run_preprocessing(pdf_path: str = None) -> str:
    """Stage 1: Extract and segment clauses from IPC PDF."""
    from preprocessing.extract_text import extract_text
    from preprocessing.segment_clauses import segment_clauses
    from preprocessing.augment_clauses import augment_clauses
    import json

    output_path = os.path.join(BASE_DIR, 'data', 'clauses_augmented.json')

    if pdf_path is None:
        pdf_path = os.path.join(BASE_DIR, 'data', 'ipc.pdf')

    if not os.path.exists(pdf_path):
        print(f"[main] WARNING: PDF not found at {pdf_path}")
        print("[main] Skipping preprocessing — using existing clauses_augmented.json")
        return output_path

    print("\n[Stage 1] Extracting text from PDF...")
    text = extract_text(pdf_path)

    print("[Stage 1] Segmenting into clauses...")
    clauses = segment_clauses(text)
    print(f"[Stage 1] Segmented {len(clauses)} clauses.")

    print("[Stage 1] Augmenting short clauses with canonical IPC text...")
    clauses = augment_clauses(clauses)

    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
    print(f"[Stage 1] Saved {len(clauses)} clauses -> {output_path}")
    return output_path


def build_indices(clauses: list, rebuild: bool = False) -> tuple:
    """Stages 2-3: Build FAISS vector index and BM25 index."""
    from indexing.vector_index import get_or_build_index, INDEX_PATH, EMBEDDINGS_PATH
    from indexing.bm25_index import get_or_build_bm25, BM25_PATH

    if rebuild:
        for path in [INDEX_PATH, EMBEDDINGS_PATH, BM25_PATH]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[main] Deleted: {path}")

    print("\n[Stage 2] Building / loading FAISS vector index...")
    index, embeddings, model = get_or_build_index(clauses)
    print(f"[Stage 2] Vector index ready: {index.ntotal} vectors (dim={index.d})")

    print("\n[Stage 3] Building / loading BM25 index...")
    bm25, _ = get_or_build_bm25(clauses)
    print(f"[Stage 3] BM25 index ready.")

    return index, embeddings, model, bm25


def run_query(query: str,
              clauses: list,
              index, embeddings, model, bm25,
              top_k: int = 5) -> None:
    """Stages 4-5: Run and display hybrid + MMR retrieval for a query."""
    from retrieval.baseline import vector_search
    from retrieval.hybrid import hybrid_search
    from retrieval.mmr import mmr_rerank

    print(f'\n{"="*65}')
    print(f'Query: "{query}"')
    print("=" * 65)

    # Vector-only
    t0 = time.time()
    vec = vector_search(query, clauses, index, model, top_k=top_k)
    t_vec = time.time() - t0

    print(f"\n[Vector-Only]  ({t_vec*1000:.0f}ms)")
    print("-" * 50)
    for r in vec:
        print(f"  #{r['rank']:2}  Sec {r['section_number']:>4}  "
              f"score={r['score']:.4f}  |  {r['snippet'][:65]}...")

    # Hybrid + MMR
    t0 = time.time()
    candidates = hybrid_search(query, clauses, index, embeddings, model, bm25, top_k=50)
    mmr = mmr_rerank(candidates, top_k=top_k)
    t_hyb = time.time() - t0

    print(f"\n[Hybrid + MMR] ({t_hyb*1000:.0f}ms)")
    print("-" * 50)
    for r in mmr:
        print(f"  MMR#{r['mmr_rank']:2}  Sec {r['section_number']:>4}  "
              f"hybrid={r['hybrid_score']:.4f}  "
              f"[vec={r['vector_score']:.3f} bm25={r['bm25_score']:.3f}]  "
              f"|  {r['snippet'][:50]}...")


def run_evaluation() -> None:
    """Stage 6: Run evaluation metrics."""
    import subprocess
    print("\n[Stage 6] Running evaluation...")
    result = subprocess.run(
        [sys.executable, '-m', 'evaluation.metrics'],
        cwd=BASE_DIR,
        capture_output=False
    )
    return result.returncode


def launch_ui() -> None:
    """Stage 7: Launch Streamlit UI."""
    import subprocess
    print("\n[Stage 7] Launching Streamlit UI at http://localhost:8501")
    subprocess.run(
        [sys.executable, '-m', 'streamlit', 'run',
         os.path.join(BASE_DIR, 'app.py'),
         '--server.port', '8501'],
        cwd=BASE_DIR
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid-MMR Legal RAG — IPC Clause Retrieval Pipeline"
    )
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild all indices')
    parser.add_argument('--query', type=str, default=None,
                        help='Single query to run')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation metrics')
    parser.add_argument('--ui', action='store_true',
                        help='Launch Streamlit UI')
    parser.add_argument('--pdf', type=str, default=None,
                        help='Path to IPC PDF (default: data/ipc.pdf)')
    args = parser.parse_args()

    print("=" * 65)
    print("   Hybrid-MMR Legal RAG System — Indian Penal Code")
    print("=" * 65)
    print(f"   BM25 + Dense Embeddings (all-MiniLM-L6-v2) + MMR")
    print("=" * 65)

    # Load clauses (preprocessing already done; use augmented JSON)
    from indexing.vector_index import load_clauses
    clauses_path = os.path.join(BASE_DIR, 'data', 'clauses_augmented.json')

    if not os.path.exists(clauses_path) or args.rebuild:
        run_preprocessing(args.pdf)

    clauses = load_clauses(clauses_path)
    print(f"\n[main] Loaded {len(clauses)} IPC clauses from {clauses_path}")

    # Build / load indices
    index, embeddings, model, bm25 = build_indices(clauses, rebuild=args.rebuild)

    # Dispatch
    if args.ui:
        launch_ui()
        return

    if args.eval:
        run_evaluation()
        return

    if args.query:
        run_query(args.query, clauses, index, embeddings, model, bm25)
        return

    # Default: demo queries
    demo_queries = [
        "punishment for murder",
        "right of private defence",
        "definition of criminal conspiracy",
        "cheating and fraud",
        "cruelty by husband",
    ]
    print(f"\n[main] Running {len(demo_queries)} demo queries...\n")
    for q in demo_queries:
        run_query(q, clauses, index, embeddings, model, bm25, top_k=3)

    print("\n\n[main] Pipeline complete.")
    print("       Run with --ui to launch the Streamlit demo.")
    print("       Run with --eval to compute full evaluation metrics.")
    print("       Run with --query 'your query' to search a specific query.")


if __name__ == '__main__':
    main()
