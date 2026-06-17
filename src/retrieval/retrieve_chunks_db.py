import os
import re
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# This file lives in src/retrieval/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
SEMANTIC_CANDIDATE_LIMIT = 20
SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25
MIN_KEYWORD_LENGTH = 3
TEST_QUESTION = "How does Napoleon gain power on the farm?"

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "at",
    "as",
    "into",
    "about",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "how",
    "what",
    "why",
    "when",
    "role",
    "play",
    "plays",
    "played",
    "where",
    "who",
    "whom",
    "which",
    "happen",
    "happens",
}


def load_database_url() -> str:
    """Load DATABASE_URL from the local .env file."""
    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL was not found. Add it to your local .env file, for "
            "example: DATABASE_URL=postgresql://postgres:postgres@localhost:5432/animal_farm_rag_kag"
        )

    return database_url


def vector_to_pgvector_string(embedding: list[float]) -> str:
    """Convert a Python embedding list into the string format pgvector accepts."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def tokenize_for_keyword_search(text: str) -> list[str]:
    """Turn text into simple lowercase keyword tokens."""
    lowercase_text = text.lower()
    tokens = re.findall(r"[a-z]+", lowercase_text)
    keyword_tokens = []

    for token in tokens:
        if len(token) < MIN_KEYWORD_LENGTH:
            continue

        if token in STOPWORDS:
            continue

        keyword_tokens.append(token)

    return keyword_tokens


def keyword_overlap_score(query: str, chunk_text: str) -> float:
    """Score how many query keywords also appear in the chunk text."""
    query_terms = set(tokenize_for_keyword_search(query))
    chunk_terms = set(tokenize_for_keyword_search(chunk_text))

    if not query_terms:
        return 0.0

    matching_terms = query_terms.intersection(chunk_terms)
    return len(matching_terms) / len(query_terms)


def retrieve_semantic_candidates_from_db(
    connection: psycopg.Connection,
    question_embedding: list[float],
    candidate_limit: int,
) -> list[dict]:
    """Use pgvector to retrieve the most semantically similar chunk candidates."""
    question_vector = vector_to_pgvector_string(question_embedding)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                chunks.id AS db_chunk_id,
                chunks.chunk_index,
                chunks.text,
                chunks.word_count,
                chunks.embedding <=> %s::vector AS semantic_distance
            FROM chunks
            WHERE chunks.embedding IS NOT NULL
            ORDER BY chunks.embedding <=> %s::vector
            LIMIT %s;
            """,
            (question_vector, question_vector, candidate_limit),
        )
        rows = cursor.fetchall()

    candidates = []

    for row in rows:
        db_chunk_id, chunk_index, text, word_count, semantic_distance = row
        semantic_similarity = 1 - float(semantic_distance)

        candidates.append(
            {
                "db_chunk_id": db_chunk_id,
                "chunk_index": chunk_index,
                "text": text,
                "word_count": word_count,
                "semantic_distance": float(semantic_distance),
                "semantic_similarity": semantic_similarity,
            }
        )

    return candidates


def retrieve_top_chunks_from_db(
    question: str,
    model: SentenceTransformer,
    top_k: int,
) -> list[dict]:
    """Retrieve top chunks from Postgres using pgvector plus keyword reranking."""
    database_url = load_database_url()
    question_embedding = model.encode(question).tolist()
    connection = None

    try:
        connection = psycopg.connect(database_url)
        candidates = retrieve_semantic_candidates_from_db(
            connection,
            question_embedding,
            SEMANTIC_CANDIDATE_LIMIT,
        )
    finally:
        if connection is not None:
            connection.close()

    scored_chunks = []

    for candidate in candidates:
        keyword_score = keyword_overlap_score(question, candidate["text"])
        hybrid_score = (
            (SEMANTIC_WEIGHT * candidate["semantic_similarity"])
            + (KEYWORD_WEIGHT * keyword_score)
        )

        scored_chunk = candidate.copy()
        scored_chunk["keyword_score"] = keyword_score
        scored_chunk["similarity"] = hybrid_score
        scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda chunk: chunk["similarity"], reverse=True)
    return scored_chunks[:top_k]


def preview_text(text: str, max_characters: int = 350) -> str:
    """Create a short, readable preview of a chunk."""
    preview = " ".join(text.split())

    if len(preview) <= max_characters:
        return preview

    return preview[:max_characters].rstrip() + "..."


def main() -> None:
    """Run a simple DB-backed retrieval test from the terminal."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = TEST_QUESTION

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    top_chunks = retrieve_top_chunks_from_db(question, model, TOP_K)

    print(f"Question: {question}")
    print("Retrieval source: Postgres + pgvector")
    print(f"Retrieval model: {DEFAULT_LOCAL_EMBEDDING_MODEL}")

    if not top_chunks:
        print("No chunks were found in Postgres. Run store_chunks_embeddings.py first.")
        return

    print("\nTop chunks:\n")

    for rank, chunk in enumerate(top_chunks, start=1):
        print(f"{rank}. Chunk index: {chunk['chunk_index']}")
        print(f"   DB chunk id: {chunk['db_chunk_id']}")
        print(f"   Hybrid score: {chunk['similarity']:.4f}")
        print(f"   Semantic similarity: {chunk['semantic_similarity']:.4f}")
        print(f"   Keyword score: {chunk['keyword_score']:.4f}")
        print(f"   Word count: {chunk['word_count']}")
        print("   Preview:")
        print(f"   {preview_text(chunk['text'])}\n")


if __name__ == "__main__":
    main()
