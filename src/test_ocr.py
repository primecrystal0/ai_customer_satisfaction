from src.extract_text import extract_text_from_pdf, parse_fields
import os

SCANNED_DIR = "data/scanned_test_pdfs"


def main():
    for filename in os.listdir(SCANNED_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(SCANNED_DIR, filename)
        print(f"\n--- {filename} ---")

        raw_text, method = extract_text_from_pdf(pdf_path)
        fields = parse_fields(raw_text)

        print(f"Method used: {method}")
        print(f"Extracted feedback: {fields['feedback_text']}")


if __name__ == "__main__":
    main()