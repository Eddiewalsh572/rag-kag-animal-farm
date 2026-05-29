import re
from pathlib import Path


# This file is in src/ingestion/.
# parents[2] moves up to the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The cleaner reads the text extracted from the PDF.
EXTRACTED_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_extracted.txt"

# The cleaner writes a separate cleaned file so the raw extraction is preserved.
CLEANED_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_cleaned.txt"


def normalize_whitespace(text: str) -> str:
    """Make whitespace more consistent before applying other cleaning rules."""
    # Different systems can write line endings differently. This makes them all
    # normal "\n" line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # pypdf often extracts tab characters between words. Spaces are easier to
    # read and easier to clean.
    text = text.replace("\t", " ")

    # Treat non-breaking spaces as normal spaces.
    text = text.replace("\xa0", " ")

    return text


def remove_metadata_before_story(text: str) -> str:
    """Remove title page metadata and table of contents before Chapter 1."""
    # In this PDF extraction, the real story begins with a decorative drop cap:
    # "M" on one line, followed by "r. Jones..." on the next line.
    #
    # Everything before that is source metadata, update information, and the
    # table of contents, so we remove it from the cleaned story text.
    story_start = re.search(r"(?m)^M\s*\n\s*r\.\s+Jones", text)

    if story_start:
        return text[story_start.start() :]

    # If a future PDF extracts differently, keep the text instead of risking
    # accidental deletion of story content.
    return text


def normalize_drop_caps(text: str) -> str:
    """Join decorative first-letter lines back to the following word."""
    # Example:
    # M
    # r. Jones
    #
    # becomes:
    # Mr. Jones
    return re.sub(r"(?m)^([A-Z])\s*\n\s*([a-z])", r"\1\2", text)


def normalize_chapter_heading_text(line: str) -> str:
    """Normalize spaced chapter headings such as 'C HAPTER 1'."""
    # This handles a one-line spacing artifact.
    return re.sub(r"\bC\s+HAPTER\s+(\d+)\b", r"CHAPTER \1", line)


def clean_lines(text: str) -> str:
    """Remove repeated source lines while preserving story lines."""
    cleaned_lines = []
    pending_chapter_heading = None
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]
        normalized_line = re.sub(r"\s+", " ", line).strip()
        lower_line = normalized_line.lower()

        # Some PDF chapter side headings are split across three lines:
        # C
        # HAPTER
        # 1
        #
        # They are layout artifacts, not story sentences. We normalize the idea
        # to "CHAPTER 1", but skip the artifact here because real chapter breaks
        # are added from the chapter URL markers below.
        next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        third_line = lines[index + 2].strip() if index + 2 < len(lines) else ""

        if normalized_line == "C" and next_line == "HAPTER" and third_line.isdigit():
            index += 3
            continue

        # Remove source URL lines. When the URL points to chapter N, it appears
        # just before the next chapter begins, so we use it to add a clean
        # chapter label and then skip the noisy URL itself.
        if "ebooks.adelaide.edu.au" in lower_line:
            chapter_match = re.search(r"chapter(\d+)\.html", lower_line)

            if chapter_match:
                chapter_number = int(chapter_match.group(1))

                if chapter_number < 10:
                    pending_chapter_heading = f"CHAPTER {chapter_number + 1}"

            index += 1
            continue

        # Remove repeated source update timestamps.
        if lower_line.startswith("last updated"):
            index += 1
            continue

        # Remove repeated PDF header/footer title lines.
        if lower_line == "animal farm, by george orwell":
            index += 1
            continue

        # Remove source/publisher metadata that may appear at the end.
        if lower_line in {
            "this web edition published by:",
            "ebooks@adelaide",
            "the university of adelaide library",
            "university of adelaide",
            "south australia 5005",
        }:
            index += 1
            continue

        # Add the clean chapter label after metadata has been skipped and right
        # before the next real story line.
        if pending_chapter_heading and normalized_line:
            cleaned_lines.append("")
            cleaned_lines.append(pending_chapter_heading)
            cleaned_lines.append("")
            pending_chapter_heading = None

        cleaned_lines.append(normalize_chapter_heading_text(line))
        index += 1

    return "\n".join(cleaned_lines)


def fix_pdf_spacing(text: str) -> str:
    """Fix common PDF spacing artifacts without changing the story meaning."""
    # Remove trailing spaces at the end of lines.
    text = re.sub(r"[ ]+$", "", text, flags=re.MULTILINE)

    # Join words split by a hyphen at a line break.
    # Example: "pop-\nholes" becomes "popholes".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join words split after an apostrophe.
    text = re.sub(r"(\w[’'])\n(\w)", r"\1\2", text)

    # Join line-wrapped sentences when the first line does not look finished.
    # This keeps paragraphs readable while avoiding aggressive rewriting.
    text = re.sub(r"([a-z,;:])\n([a-zA-Z])", r"\1 \2", text)

    # Normalize repeated spaces inside lines.
    text = re.sub(r" {2,}", " ", text)

    # Keep paragraph breaks, but reduce long blank gaps from the PDF.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def clean_text(text: str) -> str:
    """Run all beginner-friendly cleaning steps in order."""
    text = normalize_whitespace(text)
    text = remove_metadata_before_story(text)
    text = normalize_drop_caps(text)
    text = clean_lines(text)

    # Chapter 1 begins after removed front matter, so add a simple heading.
    text = f"CHAPTER 1\n\n{text}"

    text = fix_pdf_spacing(text)
    return text.strip()


def main() -> None:
    """Read extracted text, clean it, and save the cleaned text file."""
    if not EXTRACTED_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"Expected extracted text at {EXTRACTED_TEXT_PATH}. "
            "Run extract_pdf_text.py first."
        )

    extracted_text = EXTRACTED_TEXT_PATH.read_text(encoding="utf-8")
    cleaned_text = clean_text(extracted_text)

    CLEANED_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANED_TEXT_PATH.write_text(cleaned_text, encoding="utf-8")

    print(f"Cleaned text saved to: {CLEANED_TEXT_PATH}")
    print(f"Kept roughly {len(cleaned_text):,} characters.")


if __name__ == "__main__":
    main()
