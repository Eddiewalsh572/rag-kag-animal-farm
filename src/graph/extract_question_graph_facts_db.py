import json
import os
import sys
from pathlib import Path

import psycopg
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# Add the project root to Python's import path so this script can reuse the
# DB-backed retrieval module when run directly from the terminal.
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retrieve_chunks_db import (  # noqa: E402
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    TOP_K,
    preview_text,
    retrieve_top_chunks_from_db,
)


OPENCODE_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_MODEL = "gpt-5.5"
QUERY_EXTRACTED_CONFIDENCE = "llm_query_extracted"
REVIEW_STATUS = "unreviewed"
REQUIRED_FIELDS = [
    "source_entity",
    "relation",
    "target_entity",
    "description",
]


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
            source = "retrieved"
        else:
            source = "neighbor"

        evidence_chunks.append(
            {
                "db_chunk_id": db_chunk_id,
                "chunk_index": chunk_index,
                "text": text,
                "word_count": word_count,
                "source": source,
            }
        )

    return evidence_chunks


def format_evidence_for_prompt(evidence_chunks: list[dict]) -> str:
    """Format evidence chunks for the extraction prompt."""
    evidence_parts = []

    for chunk in evidence_chunks:
        evidence_parts.append(
            f"Chunk index: {chunk['chunk_index']}\n"
            f"Source: {chunk['source']}\n"
            "Text:\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(evidence_parts)


def build_extraction_prompt(question: str, evidence_chunks: list[dict]) -> str:
    """Build one prompt asking OpenCode for question-focused graph facts."""
    evidence_text = format_evidence_for_prompt(evidence_chunks)

    return f"""You are extracting knowledge graph facts from George Orwell's Animal Farm.

User question:
{question}

Use only the provided evidence chunks.
Extract graph facts that help answer the user question.
Do not use outside knowledge.
Prefer precise relation wording.
Be careful with claims, rumors, propaganda, or reported speech.
For example, prefer:
Squealer --claims--> Boxer died in hospital
over:
Boxer --dies in--> hospital
if the evidence only says Squealer announced it.
If no useful facts are present, return {{"facts": []}}.
Return JSON only. No markdown. No explanation outside JSON.

Return this JSON shape:
{{
  "facts": [
    {{
      "source_entity": "...",
      "relation": "...",
      "target_entity": "...",
      "description": "...",
      "evidence_chunk_ids": [112, 113]
    }}
  ]
}}

Evidence chunks:
{evidence_text}"""


def call_opencode(prompt: str) -> str:
    """Call OpenCode and return the raw model response content."""
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

    response = requests.post(
        OPENCODE_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def parse_graph_facts(response_text: str, evidence_chunk_indexes: set[int]) -> list[dict]:
    """Parse and lightly validate the model JSON response."""
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as error:
        print("OpenCode returned invalid JSON.")
        print("Raw response:")
        print(response_text)
        raise ValueError("Could not parse OpenCode graph facts JSON.") from error

    facts = parsed_response.get("facts", [])

    if not isinstance(facts, list):
        raise ValueError("OpenCode response must contain a 'facts' list.")

    valid_facts = []

    for index, fact in enumerate(facts, start=1):
        valid_fact = validate_graph_fact(fact, index)

        if valid_fact is None:
            continue

        evidence_chunk_ids = valid_fact.get("evidence_chunk_ids", [])

        if not evidence_chunk_ids:
            valid_fact["evidence_chunk_ids"] = []
        else:
            for chunk_id in evidence_chunk_ids:
                if chunk_id not in evidence_chunk_indexes:
                    print(
                        "Warning: fact "
                        f"{index} cites chunk {chunk_id}, which was not in the evidence."
                    )

        valid_facts.append(valid_fact)

    return valid_facts


def validate_graph_fact(fact: dict, index: int):
    """Validate one graph fact returned by OpenCode."""
    for field in REQUIRED_FIELDS:
        if not fact.get(field):
            print(f"Skipping fact {index}; missing required field: {field}")
            return None

    return fact.copy()


def delete_existing_question_extractions(cursor: psycopg.Cursor, question: str) -> int:
    """Delete previous query-focused extractions for the same question."""
    review_notes = f"Extracted for question: {question}"

    cursor.execute(
        """
        DELETE FROM graph_fact_candidates
        WHERE confidence = %s
          AND review_notes = %s;
        """,
        (QUERY_EXTRACTED_CONFIDENCE, review_notes),
    )
    return cursor.rowcount


def insert_question_extracted_fact(
    cursor: psycopg.Cursor,
    fact: dict,
    question: str,
) -> None:
    """Insert one question-focused graph fact candidate into Postgres."""
    cursor.execute(
        """
        INSERT INTO graph_fact_candidates (
            source_entity,
            relation,
            target_entity,
            description,
            evidence_chunk_ids,
            confidence,
            review_status,
            review_notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (
            fact["source_entity"],
            fact["relation"],
            fact["target_entity"],
            fact["description"],
            fact.get("evidence_chunk_ids", []),
            QUERY_EXTRACTED_CONFIDENCE,
            REVIEW_STATUS,
            f"Extracted for question: {question}",
        ),
    )


def print_evidence_chunks(evidence_chunks: list[dict]) -> None:
    """Print a compact list of evidence chunks used for extraction."""
    print("\nEvidence chunks used:")

    for chunk in evidence_chunks:
        print(f"- Chunk {chunk['chunk_index']}, source {chunk['source']}")
        print(f"  Preview: {preview_text(chunk['text'])}")


def print_graph_facts(facts: list[dict]) -> None:
    """Print extracted graph facts in a readable format."""
    print("\nExtracted graph facts:")

    if not facts:
        print("No graph facts were extracted.")
        return

    for index, fact in enumerate(facts, start=1):
        print(
            f"{index}. {fact['source_entity']} --{fact['relation']}--> "
            f"{fact['target_entity']}"
        )
        print(f"   Description: {fact['description']}")
        print(f"   Evidence chunks: {fact.get('evidence_chunk_ids', [])}")


def main() -> None:
    """Retrieve chunks, extract graph facts, and store candidates in Postgres."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print('Usage: python3 src/graph/extract_question_graph_facts_db.py "What happens to Boxer?"')
        return

    print(f"Question: {question}")
    print("Retrieval source: Postgres + pgvector")

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    retrieved_chunks = retrieve_top_chunks_from_db(question, model, TOP_K)

    if not retrieved_chunks:
        print("No chunks were found in Postgres. Run store_chunks_embeddings.py first.")
        return

    evidence_chunks = fetch_neighbor_chunks_from_db(retrieved_chunks)
    evidence_chunk_indexes = set()

    for chunk in evidence_chunks:
        evidence_chunk_indexes.add(chunk["chunk_index"])

    print_evidence_chunks(evidence_chunks)

    prompt = build_extraction_prompt(question, evidence_chunks)
    response_text = call_opencode(prompt)
    facts = parse_graph_facts(response_text, evidence_chunk_indexes)

    print_graph_facts(facts)

    database_url = load_database_url()
    connection = psycopg.connect(database_url)

    try:
        with connection.cursor() as cursor:
            delete_existing_question_extractions(cursor, question)

            for fact in facts:
                insert_question_extracted_fact(cursor, fact, question)

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        "\nSummary:\n"
        f"Inserted {len(facts)} question-focused graph candidates into Postgres."
    )


if __name__ == "__main__":
    main()
