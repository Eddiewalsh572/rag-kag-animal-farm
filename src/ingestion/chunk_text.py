import json
from pathlib import Path


# This file lives in src/ingestion/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The chunking step starts from the cleaned text and creates a JSON file.
CLEANED_TEXT_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_cleaned.txt"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_chunks.json"

# A chunk is the amount of text we give the retriever at one time.
# 150 words is small enough to stay focused, but large enough to hold an idea.
CHUNK_SIZE_WORDS = 150

# Overlap repeats a few words from the previous chunk in the next chunk.
# This helps preserve context when an idea crosses a chunk boundary.
OVERLAP_WORDS = 30


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    words = text.split()
    chunks = []
    start_index = 0
    chunk_id = 1

    while start_index < len(words):
        end_index = start_index + chunk_size_words
        chunk_words = words[start_index:end_index]
        chunk_text_value = " ".join(chunk_words)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text_value,
                "word_count": len(chunk_words),
            }
        )

        # Move forward by the chunk size minus overlap.
        # With 150 size and 30 overlap, each new chunk starts 120 words later.
        start_index += chunk_size_words - overlap_words
        chunk_id += 1

    return chunks


def main() -> None:
    """Read cleaned text, split it into chunks, and save the chunks as JSON."""
    if not CLEANED_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"Expected cleaned text at {CLEANED_TEXT_PATH}. "
            "Run clean_text.py before chunking."
        )

    cleaned_text = CLEANED_TEXT_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(cleaned_text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"Created {len(chunks)} chunks.")
    print(f"Chunks saved to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
