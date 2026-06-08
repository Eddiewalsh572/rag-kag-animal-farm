import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# This file lives in src/embeddings/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The embedding step reads the chunks created by src/ingestion/chunk_text.py.
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_chunks.json"

# The embedding step writes a new file that keeps each chunk plus its vector.
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_embeddings.json"

# This local model runs on your computer and does not require an API key.
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunk records from the JSON file."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Expected chunks at {chunks_path}. Run chunk_text.py first."
        )

    return json.loads(chunks_path.read_text(encoding="utf-8"))


def create_embedding(model: SentenceTransformer, text: str) -> list[float]:
    """Create one local embedding vector for one chunk of text."""
    # SentenceTransformer returns a NumPy array by default.
    # JSON cannot save NumPy arrays directly, so we convert it to a normal list.
    embedding = model.encode(text)
    return embedding.tolist()


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[dict]:
    """Create embeddings for all chunks and keep the original chunk metadata."""
    embedded_chunks = []
    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        if index == 1 or index % 25 == 0 or index == total_chunks:
            print(f"Embedding chunk {index} of {total_chunks}...")

        embedding = create_embedding(model, chunk["text"])

        embedded_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
                "embedding": embedding,
            }
        )

    return embedded_chunks


def save_embeddings(embedded_chunks: list[dict], embeddings_path: Path) -> None:
    """Save embedded chunks to a JSON file."""
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.write_text(
        json.dumps(embedded_chunks, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Load chunks, create local embeddings, and save them for retrieval."""
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Using local embedding model: {DEFAULT_LOCAL_EMBEDDING_MODEL}")

    # Load the model once, then reuse it for every chunk.
    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)

    embedded_chunks = embed_chunks(chunks, model)
    save_embeddings(embedded_chunks, EMBEDDINGS_PATH)

    print(f"Saved {len(embedded_chunks)} embedded chunks to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
