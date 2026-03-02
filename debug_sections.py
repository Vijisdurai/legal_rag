import pdfplumber, os

pdf_path = os.path.join('data', 'ipc.pdf')

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[18]  # Page 19
    print(f"Page size: {page.width} x {page.height}")
    # Extract text using extract_text with layout
    text = page.extract_text(layout=True)
    print("=== PAGE 19 layout=True ===")
    print(text[:2000] if text else "(empty)")
