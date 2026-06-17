import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# This file lives in src/db/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_embeddings.json"

DOCUMENT_TITLE = "Animal Farm"
DOCUMENT_AUTHOR = "George Orwell"
DOCUMENT_SOURCE_TYPE = "pdf"
DOCUMENT_SOURCE_PATH = "data/raw/animal_farm.pdf"
EXPECTED_EMBEDDING_DIMENSIONS = 384


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


def load_embedded_chunks(embeddings_path: Path) -> list[dict]:
    """Load embedded chunk records from JSON."""
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Expected embeddings file at {embeddings_path}. Run embed_chunks.py first."
        )

    return json.loads(embeddings_path.read_text(encoding="utf-8"))


def validate_embeddings(embedded_chunks: list[dict]) -> None:
    """Check that every embedding has the expected vector length."""
    for chunk in embedded_chunks:
        chunk_id = chunk.get("chunk_id")
        embedding = chunk.get("embedding", [])
        embedding_length = len(embedding)

        if embedding_length != EXPECTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Chunk {chunk_id} has embedding length {embedding_length}; "
                f"expected {EXPECTED_EMBEDDING_DIMENSIONS}."
            )


def vector_to_pgvector_string(embedding: list[float]) -> str:
    """Convert a Python list into the string format pgvector accepts."""
    return "[" + ",".join(str(value) for value in embedding) + "]"


def upsert_document(cursor: psycopg.Cursor) -> int:
    """Insert or update the Animal Farm document row and return its id."""
    cursor.execute(
        """
        INSERT INTO documents (title, author, source_type, source_path, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (title, author)
        DO UPDATE SET
            source_type = EXCLUDED.source_type,
            source_path = EXCLUDED.source_path,
            updated_at = NOW()
        RETURNING id;
        """,
        (
            DOCUMENT_TITLE,
            DOCUMENT_AUTHOR,
            DOCUMENT_SOURCE_TYPE,
            DOCUMENT_SOURCE_PATH,
        ),
    )

    return cursor.fetchone()[0]


def delete_existing_chunks(cursor: psycopg.Cursor, document_id: int) -> int:
    """Delete old chunks for this document so the script can be rerun safely."""
    cursor.execute("DELETE FROM chunks WHERE document_id = %s;", (document_id,))
    return cursor.rowcount


def insert_chunks(
    cursor: psycopg.Cursor,
    document_id: int,
    embedded_chunks: list[dict],
) -> None:
    """Insert embedded chunks into the chunks table."""
    for chunk in embedded_chunks:
        cursor.execute(
            """
            INSERT INTO chunks (
                document_id,
                chunk_index,
                text,
                word_count,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            (
                document_id,
                chunk["chunk_id"],
                chunk["text"],
                chunk["word_count"],
                vector_to_pgvector_string(chunk["embedding"]),
            ),
        )


def count_chunks_for_document(cursor: psycopg.Cursor, document_id: int) -> int:
    """Count chunks stored for one document."""
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE document_id = %s;", (document_id,))
    return cursor.fetchone()[0]


def store_chunks_in_database(embedded_chunks: list[dict], database_url: str) -> None:
    """Store the Animal Farm document and chunks in Postgres."""
    connection = psycopg.connect(database_url)

    try:
        print("Connected to database")

        with connection.cursor() as cursor:
            document_id = upsert_document(cursor)
            print(f"Stored document id: {document_id}")

            deleted_count = delete_existing_chunks(cursor, document_id)
            print(f"Deleted old chunks for document: {deleted_count}")

            insert_chunks(cursor, document_id, embedded_chunks)
            print(f"Inserted {len(embedded_chunks)} chunks")

            stored_count = count_chunks_for_document(cursor, document_id)
            print(f"Verified DB chunk count for document: {stored_count}")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    """Load local embeddings JSON and store it in Postgres."""
    database_url = load_database_url()
    embedded_chunks = load_embedded_chunks(EMBEDDINGS_PATH)
    print(f"Loaded {len(embedded_chunks)} embedded chunks")

    validate_embeddings(embedded_chunks)
    store_chunks_in_database(embedded_chunks, database_url)


if __name__ == "__main__":
    main()
