from pypdf import PdfReader
import os
import json
import re

PDF_DIR = "data/pdfs"
OUTPUT_FILE = "data/extracted_feedback.json"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Read all text out of a single PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()


def parse_fields(raw_text: str) -> dict:
    """Pull structured fields out of the raw PDF text using regex."""

    def find(pattern, text, default=""):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else default

    name = find(r"Name:\s*(.+)", raw_text)
    date = find(r"Date:\s*(.+)", raw_text)
    email = find(r"Email:\s*(.+)", raw_text)

    # Everything after "Feedback:" is the actual feedback text
    feedback_match = re.search(r"Feedback:\s*(.+)", raw_text, re.DOTALL)
    feedback_text = feedback_match.group(1).strip() if feedback_match else ""
    # collapse newlines from wrapped lines back into one sentence
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
        raw_text = extract_text_from_pdf(pdf_path)
        fields = parse_fields(raw_text)

        results.append({
            "filename": filename,
            "raw_text": raw_text,
            **fields,
        })
        print(f"Extracted: {filename} -> {fields['feedback_text'][:50]}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} extracted forms to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()