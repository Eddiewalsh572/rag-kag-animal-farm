from pathlib import Path

from pypdf import PdfReader


# This file lives in src/ingestion/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep input and output paths in one place so they are easy to explain and change.
RAW_PDF_PATH = PROJECT_ROOT / "data" / "raw" / "animal_farm.pdf"
EXTRACTED_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_extracted.txt"


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from every page in a PDF."""
    reader = PdfReader(pdf_path)
    page_texts = []

    # Loop through each page and collect the text that pypdf can read.
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Some PDF pages may not contain extractable text, so we skip empty pages.
        if text:
            page_texts.append(text)
        else:
            print(f"No extractable text found on page {page_number}.")

    # Join pages with blank lines so page boundaries do not run together.
    return "\n\n".join(page_texts)


def main() -> None:
    """Extract text from the source PDF and save it as a local text file."""
    if not RAW_PDF_PATH.exists():
        raise FileNotFoundError(
            f"Expected PDF at {RAW_PDF_PATH}. "
            "Place the source file there before running this script."
        )

    extracted_text = extract_pdf_text(RAW_PDF_PATH)

    # Make sure the processed data folder exists before writing the text file.
    EXTRACTED_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTED_TEXT_PATH.write_text(extracted_text, encoding="utf-8")

    print(f"Extracted text saved to: {EXTRACTED_TEXT_PATH}")
    print(f"Extracted roughly {len(extracted_text):,} characters.")


if __name__ == "__main__":
    main()
