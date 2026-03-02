"""
extract_text.py
---------------
Extracts text from the IPC PDF using pdfplumber with layout-aware extraction.
Falls back to OCR (pytesseract + pdf2image) if the PDF is scanned.

Strategy:
- Skip Table of Contents pages (detected by "SECTIONS" header without body content)
- Use layout=True for proper column ordering
- Clean formatting artifacts
"""

import pdfplumber
import re

# OCR threshold: if extracted text across all body pages < this many chars, treat as scanned
OCR_THRESHOLD = 5000

# A reliable marker that appears on the first body page of the IPC
# (Preamble section title)
BODY_START_MARKERS = [
    r'PREAMBLE',
    r'whereas it is expedient',
    r'Act No\.\s*45 of 1860',
    r'1\.\s+Title and extent',
]


def _is_toc_page(text: str) -> bool:
    """Return True if the page looks like a Table of Contents page."""
    low = text.lower()
    # ToC pages typically start with "SECTIONS" and have many numbered items
    has_sections_header = 'sections' in low[:100]
    # Check for dense number-dot patterns typical of ToC
    toc_entries = re.findall(r'^\s*\d{1,3}\b', text, re.MULTILINE)
    return has_sections_header or len(toc_entries) > 15


def _find_body_start_page(pdf) -> int:
    """Find the first page index where actual IPC body text starts."""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text(layout=True) or ""
        # Look for preamble / title section
        for marker in BODY_START_MARKERS:
            if re.search(marker, text, re.IGNORECASE):
                return i
    # Fallback: skip first 15 pages (typical ToC length)
    return 15


def clean_text(text: str) -> str:
    """Remove common PDF formatting artifacts."""
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n', text)
    # Remove page number lines (standalone numbers like "— 12 —" or just a digit line)
    text = re.sub(r'(?m)^\s*[-–—]?\s*\d+\s*[-–—]?\s*$', '', text)
    # Strip per-line whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines)
    return text.strip()


def extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract body text only (skip ToC pages) using pdfplumber with layout=True."""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        body_start = _find_body_start_page(pdf)
        print(f"[extract_text] Total pages: {total_pages}, body starts at page {body_start + 1}")

        pages_text = []
        for i in range(body_start, total_pages):
            page = pdf.pages[i]
            text = page.extract_text(layout=True) or ""
            if text.strip():
                pages_text.append(text)

    return '\n'.join(pages_text)


def extract_with_ocr(pdf_path: str) -> str:
    """OCR fallback using pdf2image + pytesseract for scanned PDFs."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        raise ImportError(
            "OCR fallback requires pdf2image and pytesseract.\n"
            "Also install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "And Poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
        )

    print("[OCR] Converting PDF pages to images...")
    images = convert_from_path(pdf_path, dpi=300)
    pages_text = []
    for i, image in enumerate(images):
        print(f"[OCR] Processing page {i + 1}/{len(images)}...")
        text = pytesseract.image_to_string(image, lang='eng')
        pages_text.append(text)
    return '\n'.join(pages_text)


def extract_text(pdf_path: str) -> str:
    """
    Main extraction function.
    Tries pdfplumber (body pages only) first;
    falls back to full-PDF OCR if extracted text is too short.

    Args:
        pdf_path: Path to the IPC PDF.

    Returns:
        Cleaned extracted text as a single string.
    """
    print(f"[extract_text] Reading: {pdf_path}")

    raw_text = extract_with_pdfplumber(pdf_path)

    if len(raw_text.strip()) < OCR_THRESHOLD:
        print(f"[extract_text] Very little text extracted ({len(raw_text)} chars). "
              "Falling back to OCR...")
        raw_text = extract_with_ocr(pdf_path)
    else:
        print(f"[extract_text] Extracted {len(raw_text)} characters via pdfplumber.")

    cleaned = clean_text(raw_text)
    print(f"[extract_text] After cleaning: {len(cleaned)} characters.")
    return cleaned


if __name__ == "__main__":
    import os
    pdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ipc.pdf')
    text = extract_text(pdf_path)
    print("\n--- First 800 characters of extracted text ---")
    print(text[:800])
