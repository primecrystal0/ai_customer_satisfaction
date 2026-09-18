from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import os

fake = Faker()

OUTPUT_DIR = "data/scanned_test_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCANNED_TEST_CASES = [
    ("The staff were incredibly helpful and friendly, best visit ever!", "Excellent"),
    ("Decent service overall, minor wait but nothing serious.", "Good"),
    ("The process was confusing and took longer than expected.", "Need Improvements"),
    ("Very unhappy with this, will not be returning again.", "Poor"),
]


def create_scanned_style_image(index: int, text: str, label: str):
    """Create a feedback form rendered as a flat image (simulating a scanned paper form)."""
    img = Image.new("RGB", (900, 500), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 24)
        font_body = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        # fallback if arial isn't found on this system
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    draw.text((40, 30), "Customer Feedback Form", fill="black", font=font_title)
    draw.text((40, 90), f"Name: {fake.name()}", fill="black", font=font_body)
    draw.text((40, 120), f"Date: {fake.date_this_year()}", fill="black", font=font_body)
    draw.text((40, 150), f"Email: {fake.email()}", fill="black", font=font_body)
    draw.text((40, 200), "Feedback:", fill="black", font=font_body)

    # simple manual word-wrap
    words = text.split()
    line = ""
    y = 230
    for word in words:
        test_line = f"{line} {word}".strip()
        if len(test_line) > 60:
            draw.text((40, y), line, fill="black", font=font_body)
            y += 25
            line = word
        else:
            line = test_line
    draw.text((40, y), line, fill="black", font=font_body)

    # Save as an image-only PDF (no text layer at all)
    filename = f"{OUTPUT_DIR}/scanned_{index}_{label.replace(' ', '_')}.pdf"
    img.save(filename, "PDF")
    print(f"Created: {filename}")


def main():
    for i, (text, label) in enumerate(SCANNED_TEST_CASES):
        create_scanned_style_image(i, text, label)


if __name__ == "__main__":
    main()