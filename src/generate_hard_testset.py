from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
import os

fake = Faker()

# Deliberately harder / more ambiguous examples than the training-style ones.
# These test genuine reasoning, not memorized phrasing.
HARD_TEST_CASES = [
    ("Honestly it was fine, nothing special but nothing wrong either.", "Good"),
    ("The product works but I expected a bit more for the price.", "Need Improvements"),
    ("I guess it's okay, though I probably won't be back.", "Poor"),
    ("Really impressed, didn't think it would be this smooth.", "Excellent"),
    ("It's fine I suppose, could definitely be better in a few areas.", "Need Improvements"),
    ("Not what I hoped for, I'm honestly considering switching providers.", "Poor"),
    ("Solid effort from the team, a couple rough edges but I'd recommend it.", "Good"),
    ("The support was quick but didn't actually fix my problem.", "Need Improvements"),
    ("Beyond happy with this, genuinely one of the best experiences I've had.", "Excellent"),
    ("It's alright I guess, wouldn't go out of my way to use it again.", "Poor"),
]

OUTPUT_DIR = "data/hard_test_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_pdf(index: int, true_label: str, text: str):
    filename = f"{OUTPUT_DIR}/hardtest_{index}_{true_label.replace(' ', '_')}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 800, "Customer Feedback Form")

    c.setFont("Helvetica", 11)
    c.drawString(50, 770, f"Name: {fake.name()}")
    c.drawString(50, 750, f"Date: {fake.date_this_year()}")
    c.drawString(50, 730, f"Email: {fake.email()}")

    c.drawString(50, 690, "Feedback:")
    lines = wrap(text, width=80)
    y = 670
    for line in lines:
        c.drawString(50, y, line)
        y -= 20

    c.save()
    print(f"Created: {filename}")


def main():
    for i, (text, label) in enumerate(HARD_TEST_CASES):
        generate_pdf(i, label, text)


if __name__ == "__main__":
    main()