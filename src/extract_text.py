from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
import json
import re

# --- Point to our locally installed OCR tools ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Rajeev\ai_customer_satisfaction_pdf\poppler-26.09.0\Library\bin"

PDF_DIR = "data/pdfs"
OUTPUT_FILE = "data/extracted_feedback.json"

MIN_TEXT_LENGTH = 20  # if pypdf extracts less than this, assume it's a scanned/image PDF


def extract_text_pypdf(pdf_path: str) -> str:
    """Fast path: extract text directly from a digital PDF's text layer."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def extract_text_ocr(pdf_path: str) -> str:
    """Fallback path: convert PDF pages to images and run OCR (for scanned/handwritten forms)."""
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> tuple[str, str]:
    """
    Try pypdf first (fast, accurate for digital PDFs).
    Fall back to OCR if the text layer is missing or too short (scanned/handwritten).
    Returns (extracted_text, method_used).
    """
    text = extract_text_pypdf(pdf_path)

    if len(text) < MIN_TEXT_LENGTH:
        print(f"  -> No usable text layer found, falling back to OCR...")
        text = extract_text_ocr(pdf_path)
        return text, "ocr"

    return text, "digital"


def parse_fields(raw_text: str) -> dict:
    """Pull structured fields out of the raw PDF text using regex."""

    def find(pattern, text, default=""):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    name = find(r"Name:\s*(.+)", raw_text)
    date = find(r"Date:\s*(.+)", raw_text)
    email = find(r"Email:\s*(.+)", raw_text)

    feedback_match = re.search(r"Feedback:\s*(.+)", raw_text, re.DOTALL)
    feedback_text = feedback_match.group(1).strip() if feedback_match else ""
    feedback_text = " ".join(feedback_text.split())

    return {
        "name": name,
        "date": date,
        "email": email,
        "feedback_text": feedback_text,
    }


def main():
    results = []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        raw_text, method = extract_text_from_pdf(pdf_path)
        fields = parse_fields(raw_text)

        results.append({
            "filename": filename,
            "raw_text": raw_text,
            "extraction_method": method,
            **fields,
        })
        print(f"Extracted ({method}): {filename} -> {fields['feedback_text'][:50]}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} extracted forms to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()