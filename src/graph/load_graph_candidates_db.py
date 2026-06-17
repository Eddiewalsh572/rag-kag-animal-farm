import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
EXTRACTED_FACTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "animal_farm_graph_facts_extracted.json"
)

DEFAULT_CONFIDENCE = "llm_extracted"
DEFAULT_REVIEW_STATUS = "unreviewed"
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


def load_extracted_candidates(path: Path) -> list[dict]:
    """Load extracted graph candidates from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected extracted graph facts at {path}. Run extract_graph_facts.py first."
        )

    graph_data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(graph_data, list):
        return graph_data

    if isinstance(graph_data, dict) and isinstance(graph_data.get("facts"), list):
        return graph_data["facts"]

    raise ValueError(
        "Extracted graph facts JSON must be a list or a dictionary with a 'facts' list."
    )


def validate_candidate(candidate: dict, index: int) -> None:
    """Check that a candidate has the required fields."""
    for field in REQUIRED_FIELDS:
        if not candidate.get(field):
            raise ValueError(f"Candidate {index} is missing required field: {field}")


def delete_existing_llm_candidates(cursor: psycopg.Cursor) -> int:
    """Delete old LLM-extracted candidates so reruns do not create duplicates."""
    cursor.execute(
        "DELETE FROM graph_fact_candidates WHERE confidence = %s;",
        (DEFAULT_CONFIDENCE,),
    )
    return cursor.rowcount


def insert_candidate(cursor: psycopg.Cursor, candidate: dict) -> None:
    """Insert one graph fact candidate into Postgres."""
    evidence_chunk_ids = candidate.get("evidence_chunk_ids", [])
    confidence = candidate.get("confidence", DEFAULT_CONFIDENCE)

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
            candidate["source_entity"],
            candidate["relation"],
            candidate["target_entity"],
            candidate["description"],
            evidence_chunk_ids,
            confidence,
            DEFAULT_REVIEW_STATUS,
            None,
        ),
    )


def count_llm_candidates(cursor: psycopg.Cursor) -> int:
    """Count LLM-extracted graph fact candidates in Postgres."""
    cursor.execute(
        "SELECT COUNT(*) FROM graph_fact_candidates WHERE confidence = %s;",
        (DEFAULT_CONFIDENCE,),
    )
    return cursor.fetchone()[0]


def load_candidates_into_database(candidates: list[dict], database_url: str) -> None:
    """Load extracted graph candidates into Postgres."""
    connection = psycopg.connect(database_url)

    try:
        print("Connected to database")

        with connection.cursor() as cursor:
            deleted_count = delete_existing_llm_candidates(cursor)
            print(f"Deleted old llm_extracted candidates: {deleted_count}")

            for index, candidate in enumerate(candidates, start=1):
                validate_candidate(candidate, index)
                insert_candidate(cursor, candidate)

            print(f"Inserted {len(candidates)} graph candidates")

            stored_count = count_llm_candidates(cursor)
            print(f"Verified DB candidate count: {stored_count}")

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    """Load extracted graph fact candidates from JSON into Postgres."""
    database_url = load_database_url()
    candidates = load_extracted_candidates(EXTRACTED_FACTS_PATH)
    print(f"Loaded {len(candidates)} extracted graph candidates from JSON")

    load_candidates_into_database(candidates, database_url)


if __name__ == "__main__":
    main()
