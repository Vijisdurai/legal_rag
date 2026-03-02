"""
preprocessing/load_ipc_json.py
------------------------------
Loads the IPC structured JSON (civictech-India/Indian-Law-Penal-Code-Json)
and converts it into the standard clause dict format used throughout the project.

Each clause dict:
  {
    "section_number": "302",
    "title":          "Punishment for murder",
    "chapter":        16,
    "chapter_title":  "of offences affecting the human body",
    "text":           "302. Punishment for murder.\n<section_desc>",
    "length":         <int>
  }

This replaces both the PDF extraction pipeline and augment_clauses.py.
Every section now has complete, clean text straight from the bare act.
"""

import json
import os
import re

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
RAW_PATH = os.path.join(BASE_DIR, 'data', 'ipc_raw.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'clauses.json')
AUGMENTED_PATH = os.path.join(BASE_DIR, 'data', 'clauses_augmented.json')


def section_sort_key(sec_str: str):
    """Sort key for IPC section numbers like '1', '120A', '304B'."""
    num = int(re.sub(r'[A-Za-z]', '', sec_str))
    suffix = re.sub(r'\d', '', sec_str).upper()
    return (num, suffix)


def load_ipc_json(raw_path: str = RAW_PATH) -> list[dict]:
    """
    Load raw IPC JSON and convert to standard clause dicts.

    Returns:
        List of clause dicts sorted by section number.
    """
    with open(raw_path, encoding='utf-8') as f:
        raw = json.load(f)

    clauses = []
    seen = set()

    for entry in raw:
        sec_num = str(entry.get('Section', '')).strip()
        title = entry.get('section_title', '').strip()
        desc = entry.get('section_desc', '').strip()
        chapter = entry.get('chapter', 0)
        chapter_title = entry.get('chapter_title', '').strip().title()

        if not sec_num or sec_num in seen:
            continue
        seen.add(sec_num)

        # Build rich combined text: "Section X. Title.\nDescription"
        text = f"{sec_num}. {title}.\n{desc}" if desc else f"{sec_num}. {title}."

        clauses.append({
            "section_number": sec_num,
            "title": title,
            "chapter": chapter,
            "chapter_title": chapter_title,
            "text": text,
            "length": len(text),
        })

    # Sort by section number
    clauses.sort(key=lambda c: section_sort_key(c['section_number']))

    return clauses


def save_clauses(clauses: list[dict],
                 path: str = OUTPUT_PATH,
                 also_save_augmented: bool = True) -> None:
    """Save clause list to JSON. Also mirrors to clauses_augmented.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
    print(f"[load_ipc_json] Saved {len(clauses)} clauses -> {path}")

    if also_save_augmented:
        with open(AUGMENTED_PATH, 'w', encoding='utf-8') as f:
            json.dump(clauses, f, indent=2, ensure_ascii=False)
        print(f"[load_ipc_json] Mirrored  -> {AUGMENTED_PATH}")


if __name__ == '__main__':
    clauses = load_ipc_json()

    # Stats
    lengths = [c['length'] for c in clauses]
    short = sum(1 for l in lengths if l < 100)
    print(f"\n[load_ipc_json] Total clauses : {len(clauses)}")
    print(f"[load_ipc_json] Short (<100)  : {short} ({short/len(clauses)*100:.1f}%)")
    print(f"[load_ipc_json] Avg length    : {sum(lengths)/len(lengths):.0f} chars")
    print(f"[load_ipc_json] Min / Max     : {min(lengths)} / {max(lengths)}")

    # Sample key sections
    print("\nSample sections:")
    for c in clauses:
        if c['section_number'] in ['302', '376', '323', '420', '498A']:
            print(f"  Sec {c['section_number']:>4}  ({c['length']:>4} chars): {c['text'][:100]}")

    save_clauses(clauses)
