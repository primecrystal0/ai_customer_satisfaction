from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import random
import os
from textwrap import wrap

fake = Faker()

# Sample feedback sentences for each satisfaction level
# (we write these ourselves so we control the ground-truth label)
FEEDBACK_TEMPLATES = {
    "Excellent": [
        "Absolutely loved the service, the staff went above and beyond!",
        "This exceeded my expectations, I will definitely recommend it to friends.",
        "Fantastic experience from start to finish, truly outstanding support.",
    ],
    "Good": [
        "Overall a good experience, though the checkout was a bit slow.",
        "Satisfied with the product, just wish delivery was a little faster.",
        "Pretty good service, minor issue with packaging but nothing major.",
    ],
    "Need Improvements": [
        "The wait was a bit long and the app was confusing to navigate.",
        "Product is okay but the instructions could be clearer.",
        "Service was average, there is definitely room for improvement.",
    ],
    "Poor": [
        "Terrible experience, I want a refund immediately.",
        "Extremely disappointed, the staff was rude and unhelpful.",
        "This was a waste of money, I will never use this again.",
    ],
}

OUTPUT_DIR = "data/pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_feedback_pdf(index: int, label: str, text: str):
    """Create one fake PDF feedback form."""
    filename = f"{OUTPUT_DIR}/feedback_{index}_{label.replace(' ', '_')}.pdf"
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


def main(num_forms: int = 20):
    for i in range(num_forms):
        label = random.choice(list(FEEDBACK_TEMPLATES.keys()))
        text = random.choice(FEEDBACK_TEMPLATES[label])
        generate_feedback_pdf(i, label, text)


if __name__ == "__main__":
    main(num_forms=20)