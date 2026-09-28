from PIL import Image
import pytesseract
import cv2
import re

# --------------------------------------------------
# Tesseract Path
# --------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_serial_number(image_path):

    print("Testing:", image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return "NOT_FOUND"

    print("Shape:", img.shape)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # OCR
    text = pytesseract.image_to_string(
        rgb,
        config="--psm 11"
    )

    print("\nRAW OCR OUTPUT\n")
    print(text)

    # Extract possible serial-like strings
    matches = re.findall(
        r"[A-Z0-9]{4,12}",
        text
    )

    print("\nALL MATCHES\n")
    print(matches)

    best_match = "NOT_FOUND"

    for m in matches:

        digit_count = sum(
            c.isdigit()
            for c in m
        )

        if digit_count >= 4:
            best_match = m
            break

    return best_match


if __name__ == "__main__":

    image_path = r"test_images\IMG20221010133541.jpg"

    serial_number = extract_serial_number(
        image_path
    )

    print()
    print("FINAL SERIAL NUMBER")
    print(serial_number)