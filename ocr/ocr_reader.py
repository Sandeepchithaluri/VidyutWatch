import easyocr
import os

def extract_text_from_bill(image_path):
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at '{image_path}'")
        return None

    print(f"Reading bill: {image_path}")
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path)

    print(f"\nTotal text blocks detected: {len(results)}")
    print("\n--- EXTRACTED TEXT ---")

    extracted_text = ""
    for (bbox, text, confidence) in results:
        if confidence > 0.4:  # increased threshold for cleaner output
            print(f"[{confidence:.2f}] {text}")
            extracted_text += text + "\n"

    return extracted_text


if __name__ == "__main__":
    image_path = "sample_bill.jpg"
    text = extract_text_from_bill(image_path)

    if text:
        with open("ocr/extracted_text.txt", "w") as f:
            f.write(text)
        print("\n✅ Saved to ocr/extracted_text.txt")
    else:
        print("No text extracted.")