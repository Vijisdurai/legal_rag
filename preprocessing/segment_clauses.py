"""
segment_clauses.py
------------------
Splits the raw IPC body text into individual section-level clauses.

The IPC body text format is:
    <number>. <Title>.—<clause text>
    (or continuation without "Section" keyword)

Outputs structured JSON:
    [{ "section_number": "1", "text": "...", "length": 123 }, ...]
"""

import re
import json
import os

# Section header pattern:
# Matches lines like:
#   "1. Title and extent of operation..."
#   "120B. Punishment of criminal conspiracy."
#   "  302. Punishment for murder.—Whoever commits murder..."
# The section number must appear at or near the start of a line.
SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s{0,12}(\d{1,3}[A-Z]?)\.\s+[A-Z"]',  # number dot space CAPITAL
)


def segment_clauses(text: str) -> list[dict]:
    """
    Split the IPC body text into individual clauses by section header.

    Args:
        text: Clean extracted IPC body text.

    Returns:
        List of dicts: { section_number, text, length }.
    """
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        raise ValueError(
            "No section headers found. Check that the PDF text was extracted correctly."
        )

    print(f"[segment_clauses] Found {len(matches)} potential section headers.")

    # Deduplicate: IPC sections 1–511, keep only valid range
    clauses = []
    seen_sections = set()

    for i, match in enumerate(matches):
        section_number = match.group(1).strip()

        # Validate section number is in reasonable range
        num = int(re.sub(r'[A-Z]', '', section_number))
        if num < 1 or num > 600:
            continue
        if section_number in seen_sections:
            continue
        seen_sections.add(section_number)

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        clause_text = text[start:end].strip()

        # Skip very short matches (likely ToC remnants)
        if len(clause_text) < 30:
            continue

        clauses.append({
            "section_number": section_number,
            "text": clause_text,
            "length": len(clause_text)
        })

    # Sort by section number
    clauses.sort(key=lambda c: (int(re.sub(r'[A-Z]', '', c['section_number'])),
                                 c['section_number']))

    return clauses


def save_clauses(clauses: list[dict], output_path: str) -> None:
    """Save clause list to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clauses, f, indent=2, ensure_ascii=False)
    print(f"[segment_clauses] Saved {len(clauses)} clauses to: {output_path}")


def load_clauses(json_path: str) -> list[dict]:
    """Load clause list from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    from preprocessing.extract_text import extract_text

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    pdf_path = os.path.join(base_dir, 'data', 'ipc.pdf')
    output_path = os.path.join(base_dir, 'data', 'clauses.json')

    # Step 1: Extract body text
    text = extract_text(pdf_path)

    # Step 2: Segment into clauses
    clauses = segment_clauses(text)

    # Step 3: Save
    save_clauses(clauses, output_path)

    # Step 4: Print summary
    print(f"\n--- Summary ---")
    print(f"Total clauses: {len(clauses)}")
    if clauses:
        print(f"Shortest: {min(c['length'] for c in clauses)} chars")
        print(f"Longest:  {max(c['length'] for c in clauses)} chars")
        avg = sum(c['length'] for c in clauses) / len(clauses)
        print(f"Average:  {avg:.0f} chars")
        print("\n--- First 5 clauses ---")
        for clause in clauses[:5]:
            print(f"  Section {clause['section_number']}: {clause['text'][:100]}...")
