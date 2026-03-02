"""
retrieval/dual_corpus.py
------------------------
Dual-corpus retrieval across IPC (575 sections) + BNS 2023 (58+ sections).

Features:
  - Search either corpus or both simultaneously
  - Provenance tags: 'ipc' / 'bns' on every result
  - IPC ↔ BNS cross-reference links shown in results
  - Supports all retrieval modes: Vector, BM25, Hybrid+MMR
"""

import os
import sys
import json
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, BASE_DIR)

BNS_CLAUSES_PATH = os.path.join(BASE_DIR, 'data', 'bns_clauses.json')
IPC_CLAUSES_PATH = os.path.join(BASE_DIR, 'data', 'clauses.json')

# ── Corpus loaders ─────────────────────────────────────────────────────────────

def load_bns_clauses(path: str = BNS_CLAUSES_PATH) -> list[dict]:
    """Load BNS clauses; generate file if missing."""
    if not os.path.exists(path):
        print("[dual_corpus] BNS clauses not found — generating...")
        from preprocessing.load_bns_json import load_bns_sections, save_bns_clauses
        clauses = load_bns_sections()
        save_bns_clauses(clauses, path)
        return clauses
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_ipc_clauses(path: str = IPC_CLAUSES_PATH) -> list[dict]:
    """Load IPC clauses and tag with corpus='ipc'."""
    with open(path, encoding='utf-8') as f:
        clauses = json.load(f)
    for c in clauses:
        c.setdefault('corpus', 'ipc')
    return clauses


def load_combined_corpus(ipc_path: str = IPC_CLAUSES_PATH,
                          bns_path: str = BNS_CLAUSES_PATH) -> list[dict]:
    """
    Merge IPC + BNS into a single corpus for dual-corpus search.
    IPC sections are tagged corpus='ipc', BNS sections corpus='bns'.
    Returns: merged list, sorted IPC first then BNS.
    """
    ipc = load_ipc_clauses(ipc_path)
    bns = load_bns_clauses(bns_path)
    print(f"[dual_corpus] IPC: {len(ipc)} clauses | BNS: {len(bns)} clauses")
    return ipc + bns


# ── Dual-corpus search ─────────────────────────────────────────────────────────

def dual_vector_search(query: str,
                        combined_clauses: list[dict],
                        index, model,
                        corpus_filter: str = "both",
                        top_k: int = 5) -> list[dict]:
    """
    Vector search across combined IPC+BNS corpus.

    Args:
        corpus_filter: 'ipc' | 'bns' | 'both'
    """
    from retrieval.baseline import vector_search

    # Search full combined corpus
    results = vector_search(query, combined_clauses, index, model, top_k=top_k * 3)

    # Filter by corpus
    if corpus_filter != "both":
        results = [r for r in results if
                   combined_clauses[_find_idx(r, combined_clauses)].get('corpus') == corpus_filter]

    # Attach corpus tag and cross-reference
    for r in results:
        idx = _find_idx(r, combined_clauses)
        if idx >= 0:
            c = combined_clauses[idx]
            r['corpus'] = c.get('corpus', 'ipc')
            r['ipc_equivalent'] = c.get('ipc_equivalent')
            r['bns_equivalent'] = _get_bns_eq(c)

    return results[:top_k]


def dual_hybrid_mmr_search(query: str,
                            combined_clauses: list[dict],
                            index, embeddings, model, bm25,
                            corpus_filter: str = "both",
                            top_k: int = 5) -> list[dict]:
    """
    Hybrid+MMR search across combined IPC+BNS corpus.

    Args:
        corpus_filter: 'ipc' | 'bns' | 'both'
    """
    from retrieval.hybrid import hybrid_search
    from retrieval.mmr import mmr_rerank

    candidates = hybrid_search(query, combined_clauses, index, embeddings,
                                model, bm25, top_k=50)

    if corpus_filter != "both":
        candidates = [r for r in candidates if
                      combined_clauses[_find_idx(r, combined_clauses)].get('corpus') == corpus_filter]

    results = mmr_rerank(candidates, top_k=top_k)

    # Attach provenance
    for r in results:
        idx = _find_idx(r, combined_clauses)
        if idx >= 0:
            c = combined_clauses[idx]
            r['corpus'] = c.get('corpus', 'ipc')
            r['ipc_equivalent'] = c.get('ipc_equivalent')
            r['bns_equivalent'] = _get_bns_eq(c)

    return results


# ── IPC ↔ BNS cross-reference ──────────────────────────────────────────────────

def get_bns_for_ipc_section(ipc_section: str,
                              bns_clauses: list[dict]) -> dict | None:
    """Given an IPC section number, return corresponding BNS clause (if mapped)."""
    from preprocessing.load_bns_json import get_ipc_to_bns_map
    bns_map = get_ipc_to_bns_map()
    bns_sec = bns_map.get(str(ipc_section))
    if not bns_sec:
        return None
    for c in bns_clauses:
        if c['section_number'] == bns_sec:
            return c
    return None


def get_ipc_for_bns_section(bns_section: str,
                              ipc_clauses: list[dict]) -> dict | None:
    """Given a BNS section number, return corresponding IPC clause (if mapped)."""
    from preprocessing.load_bns_json import get_bns_to_ipc_map
    ipc_map = get_bns_to_ipc_map()
    ipc_sec = ipc_map.get(str(bns_section))
    if not ipc_sec:
        return None
    for c in ipc_clauses:
        if c['section_number'] == ipc_sec:
            return c
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_idx(result: dict, clauses: list[dict]) -> int:
    """Find clause index by section_number match."""
    sec = result.get('section_number')
    for i, c in enumerate(clauses):
        if c['section_number'] == sec and c.get('corpus') == result.get('corpus', c.get('corpus')):
            return i
    # fallback: first match by section number
    for i, c in enumerate(clauses):
        if c['section_number'] == sec:
            return i
    return -1


def _get_bns_eq(clause: dict) -> str | None:
    """Return BNS equivalent for an IPC clause."""
    from preprocessing.load_bns_json import get_ipc_to_bns_map
    if clause.get('corpus') == 'ipc':
        return get_ipc_to_bns_map().get(clause['section_number'])
    return None


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from indexing.vector_index import get_or_build_index
    from indexing.bm25_index import get_or_build_bm25

    combined = load_combined_corpus()
    print(f"Combined corpus: {len(combined)} clauses")

    index, embeddings, model = get_or_build_index(combined)
    bm25, _ = get_or_build_bm25(combined)

    queries = [
        ("punishment for murder", "both"),
        ("punishment for murder", "bns"),
        ("cruelty by husband", "both"),
        ("organised crime", "both"),
        ("terrorist act", "both"),
    ]

    for q, filt in queries:
        print(f'\nQuery: "{q}"  [corpus={filt}]')
        results = dual_hybrid_mmr_search(
            q, combined, index, embeddings, model, bm25,
            corpus_filter=filt, top_k=3
        )
        for r in results:
            corpus = r.get('corpus', '?').upper()
            sec    = r['section_number']
            xref   = r.get('ipc_equivalent') or r.get('bns_equivalent')
            xref_s = f" [≈ {'BNS' if corpus=='IPC' else 'IPC'} §{xref}]" if xref else ""
            print(f"  [{corpus}] §{sec}{xref_s}  hybrid={r['hybrid_score']:.3f}  "
                  f"{r['snippet'][:55]}...")
