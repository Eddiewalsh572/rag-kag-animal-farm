import os
import sys
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# This file lives in src/generation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so this script can reuse the
# DB-backed retrieval module when run directly from the terminal.
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retrieve_chunks_db import (  # noqa: E402
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    TOP_K,
    load_database_url,
    preview_text,
    retrieve_top_chunks_from_db,
)


ENV_PATH = PROJECT_ROOT / ".env"
OPENCODE_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_MODEL = "gpt-5.5"
MAX_EVIDENCE_CHARACTERS = 1500
TEST_QUESTION = "How does Napoleon gain power on the farm?"


def load_opencode_key() -> str:
    """Load OPENCODE_KEY from the local .env file."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("OPENCODE_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENCODE_KEY was not found. Add it to your local .env file before "
            "running this script."
        )

    return api_key


def trim_evidence_text(text: str) -> str:
    """Keep evidence text readable without making the prompt too large."""
    trimmed_text = " ".join(text.split())

    if len(trimmed_text) <= MAX_EVIDENCE_CHARACTERS:
        return trimmed_text

    return trimmed_text[:MAX_EVIDENCE_CHARACTERS].rstrip() + "..."


def fetch_neighbor_chunks_from_db(
    retrieved_chunks: list[dict],
    neighbor_distance: int = 1,
) -> list[dict]:
    """Fetch previous and next chunks around each retrieved chunk from Postgres."""
    database_url = load_database_url()
    retrieved_by_index = {}
    chunk_indexes_to_fetch = set()

    for chunk in retrieved_chunks:
        chunk_index = chunk["chunk_index"]
        retrieved_by_index[chunk_index] = chunk

        for neighbor_index in range(
            chunk_index - neighbor_distance,
            chunk_index + neighbor_distance + 1,
        ):
            chunk_indexes_to_fetch.add(neighbor_index)

    if not chunk_indexes_to_fetch:
        return []

    connection = None

    try:
        connection = psycopg.connect(database_url)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunks.id AS db_chunk_id,
                    chunks.chunk_index,
                    chunks.text,
                    chunks.word_count
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE documents.title = %s
                  AND documents.author = %s
                  AND chunks.chunk_index = ANY(%s)
                ORDER BY chunks.chunk_index;
                """,
                ("Animal Farm", "George Orwell", list(chunk_indexes_to_fetch)),
            )
            rows = cursor.fetchall()
    finally:
        if connection is not None:
            connection.close()

    evidence_chunks = []

    for row in rows:
        db_chunk_id, chunk_index, text, word_count = row

        if chunk_index in retrieved_by_index:
            retrieved_chunk = retrieved_by_index[chunk_index]
            evidence_chunks.append(
                {
                    "db_chunk_id": db_chunk_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "word_count": word_count,
                    "source": "retrieved",
                    "similarity": retrieved_chunk["similarity"],
                    "semantic_similarity": retrieved_chunk["semantic_similarity"],
                    "keyword_score": retrieved_chunk["keyword_score"],
                }
            )
        else:
            evidence_chunks.append(
                {
                    "db_chunk_id": db_chunk_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "word_count": word_count,
                    "source": "neighbor",
                    "similarity": 0.0,
                    "semantic_similarity": 0.0,
                    "keyword_score": 0.0,
                }
            )

    return evidence_chunks


def build_prompt(question: str, evidence_chunks: list[dict]) -> str:
    """Build a grounded RAG prompt from DB-backed evidence chunks."""
    evidence_parts = []

    for chunk in evidence_chunks:
        evidence_parts.append(
            f"[Chunk {chunk['chunk_index']}]\n"
            f"{trim_evidence_text(chunk['text'])}"
        )

    evidence_text = "\n\n---\n\n".join(evidence_parts)

    return f"""Answer the user's question about Animal Farm.
Use only the provided evidence.
If the evidence is not enough, say so.
Answer in 3-5 sentences.
Cite chunk IDs like [Chunk 114].
Do not invent citations.
Do not use outside knowledge.

Question:
{question}

Evidence:
{evidence_text}

Answer:"""


def generate_opencode_answer(prompt: str) -> str:
    """Send the prompt to OpenCode and return the generated answer."""
    api_key = load_opencode_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0",
    }
    payload = {
        "model": OPENCODE_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENCODE_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("The OpenCode generation request failed.") from error

    return response.json()["choices"][0]["message"]["content"].strip()


def print_text_evidence_used(evidence_chunks: list[dict]) -> None:
    """Print DB-backed evidence chunks in a readable format."""
    for rank, chunk in enumerate(evidence_chunks, start=1):
        print(f"{rank}. Chunk index: {chunk['chunk_index']}")
        print(f"   DB chunk id: {chunk['db_chunk_id']}")
        print(f"   Source: {chunk['source']}")
        print(f"   Hybrid score: {chunk['similarity']:.4f}")
        print(f"   Semantic similarity: {chunk['semantic_similarity']:.4f}")
        print(f"   Keyword score: {chunk['keyword_score']:.4f}")
        print("   Preview:")
        print(f"   {preview_text(chunk['text'])}\n")


def generate_rag_answer_for_question(question: str) -> dict:
    """Run the DB-backed RAG flow and return the answer plus evidence."""
    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    retrieved_chunks = retrieve_top_chunks_from_db(question, model, TOP_K)

    if not retrieved_chunks:
        return {
            "answer": "No chunks were found in Postgres. Run store_chunks_embeddings.py first.",
            "evidence_chunks": [],
        }

    evidence_chunks = fetch_neighbor_chunks_from_db(retrieved_chunks)
    prompt = build_prompt(question, evidence_chunks)
    answer = generate_opencode_answer(prompt)

    return {
        "answer": answer,
        "evidence_chunks": evidence_chunks,
    }


def main() -> None:
    """Generate a RAG answer using chunks retrieved from Postgres."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = TEST_QUESTION

    result = generate_rag_answer_for_question(question)
    answer = result["answer"]
    evidence_chunks = result["evidence_chunks"]

    print(f"Question: {question}")
    print("Retrieval source: Postgres + pgvector")

    print("\nGenerated DB-backed RAG answer:")
    print(answer)

    print("\nText evidence used:")
    print_text_evidence_used(evidence_chunks)

    print("\nGenerated DB-backed RAG answer repeated")
    print(answer)


if __name__ == "__main__":
    main()
