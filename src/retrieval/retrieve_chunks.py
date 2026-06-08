import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# This file lives in src/retrieval/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Retrieval starts from the saved chunk embeddings.
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_embeddings.json"

# The question must use the same model that created the chunk embeddings.
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# For now, keep retrieval simple and return the top 5 matches.
TOP_K = 6

# Hardcoded test question so the retrieval logic is easy to follow first.
TEST_QUESTION = "How does Napoleon gain power on the farm?"


def load_embedded_chunks(embeddings_path: Path) -> list[dict]:
    """Load embedded chunks from the JSON file."""
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Expected embeddings at {embeddings_path}. Run embed_chunks.py first."
        )

    return json.loads(embeddings_path.read_text(encoding="utf-8"))


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compare two vectors and return a similarity score."""
    array_a = np.array(vector_a)
    array_b = np.array(vector_b)

    length_a = np.linalg.norm(array_a)
    length_b = np.linalg.norm(array_b)

    # Avoid dividing by zero if a bad or empty vector ever appears.
    if length_a == 0 or length_b == 0:
        return 0.0

    return float(np.dot(array_a, array_b) / (length_a * length_b))


def retrieve_top_chunks(
    question: str,
    embedded_chunks: list[dict],
    model: SentenceTransformer,
    top_k: int,
) -> list[dict]:
    """Embed the question and find the most similar saved chunks."""
    question_embedding = model.encode(question).tolist()
    scored_chunks = []

    for chunk in embedded_chunks:
        similarity = cosine_similarity(question_embedding, chunk["embedding"])

        scored_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
                "similarity": similarity,
            }
        )

    scored_chunks.sort(key=lambda chunk: chunk["similarity"], reverse=True)
    return scored_chunks[:top_k]


def expand_question_for_retrieval(question: str) -> str:
    """Add a few helpful search terms for abstract questions."""
    lowercase_question = question.lower()

    if "fear" in lowercase_question or "afraid" in lowercase_question:
        related_terms = "dogs growls cowered shivers executions confessions threats Jones"
        return f"{question} {related_terms}"

    squealer_manipulation_words = [
        "manipulate",
        "manipulates",
        "manipulation",
        "lie",
        "lies",
        "propaganda",
    ]

    asks_about_squealer = "squealer" in lowercase_question
    asks_about_manipulation = False

    for word in squealer_manipulation_words:
        if word in lowercase_question:
            asks_about_manipulation = True

    if asks_about_squealer and asks_about_manipulation:
        related_terms = (
            "lies statistics figures Snowball Jones windmill Boxer commandments "
            "remember propaganda"
        )
        return f"{question} {related_terms}"

    return question


def preview_text(text: str, max_characters: int = 350) -> str:
    """Create a short, readable preview of a chunk."""
    preview = " ".join(text.split())

    if len(preview) <= max_characters:
        return preview

    return preview[:max_characters].rstrip() + "..."


def main() -> None:
    """Load embeddings, embed a test question, and print the best matches."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = TEST_QUESTION

    retrieval_query = expand_question_for_retrieval(question)
    embedded_chunks = load_embedded_chunks(EMBEDDINGS_PATH)

    print(f"Loaded {len(embedded_chunks)} embedded chunks.")
    print(f"Using local embedding model: {DEFAULT_LOCAL_EMBEDDING_MODEL}")

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    top_chunks = retrieve_top_chunks(retrieval_query, embedded_chunks, model, TOP_K)

    print(f"\nQuestion: {question}")
    print(f"Retrieval query: {retrieval_query}")
    print(f"\nTop {TOP_K} matching chunks:\n")

    for rank, chunk in enumerate(top_chunks, start=1):
        print(f"{rank}. Chunk ID: {chunk['chunk_id']}")
        print(f"   Similarity: {chunk['similarity']:.4f}")
        print(f"   Word Count: {chunk['word_count']}")
        print("   Preview:")
        print(f"   {preview_text(chunk['text'])}\n")


if __name__ == "__main__":
    main()
