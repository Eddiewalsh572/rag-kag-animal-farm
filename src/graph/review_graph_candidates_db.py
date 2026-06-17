import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


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


def search_candidates(cursor: psycopg.Cursor, search_term: str) -> list[dict]:
    """Find candidates where the search term appears in key text fields."""
    search_pattern = f"%{search_term}%"

    cursor.execute(
        """
        SELECT
            id,
            source_entity,
            relation,
            target_entity,
            description,
            evidence_chunk_ids,
            confidence,
            review_status,
            review_notes
        FROM graph_fact_candidates
        WHERE source_entity ILIKE %s
           OR relation ILIKE %s
           OR target_entity ILIKE %s
           OR description ILIKE %s
        ORDER BY id;
        """,
        (search_pattern, search_pattern, search_pattern, search_pattern),
    )

    return rows_to_candidates(cursor.fetchall())


def load_all_candidates(cursor: psycopg.Cursor) -> list[dict]:
    """Load every graph fact candidate for review."""
    cursor.execute(
        """
        SELECT
            id,
            source_entity,
            relation,
            target_entity,
            description,
            evidence_chunk_ids,
            confidence,
            review_status,
            review_notes
        FROM graph_fact_candidates
        ORDER BY id;
        """
    )

    return rows_to_candidates(cursor.fetchall())


def rows_to_candidates(rows: list[tuple]) -> list[dict]:
    """Convert database rows into dictionaries that are easier to print."""
    candidates = []

    for row in rows:
        (
            candidate_id,
            source_entity,
            relation,
            target_entity,
            description,
            evidence_chunk_ids,
            confidence,
            review_status,
            review_notes,
        ) = row

        candidates.append(
            {
                "id": candidate_id,
                "source_entity": source_entity,
                "relation": relation,
                "target_entity": target_entity,
                "description": description,
                "evidence_chunk_ids": evidence_chunk_ids or [],
                "confidence": confidence,
                "review_status": review_status,
                "review_notes": review_notes,
            }
        )

    return candidates


def print_candidate(candidate: dict, rank: int) -> None:
    """Print one candidate in a clean review format."""
    evidence_chunk_ids = []

    for chunk_id in candidate["evidence_chunk_ids"]:
        evidence_chunk_ids.append(str(chunk_id))

    review_notes = candidate["review_notes"]

    if review_notes is None:
        review_notes = ""

    print(f"{rank}. Candidate ID: {candidate['id']}")
    print(f"   Source entity: {candidate['source_entity']}")
    print(f"   Relation: {candidate['relation']}")
    print(f"   Target entity: {candidate['target_entity']}")
    print(f"   Description: {candidate['description']}")
    print(f"   Evidence chunk IDs: {', '.join(evidence_chunk_ids)}")
    print(f"   Confidence: {candidate['confidence']}")
    print(f"   Review status: {candidate['review_status']}")
    print(f"   Review notes: {review_notes}")


def print_usage() -> None:
    """Print examples for using this review script."""
    print("Usage:")
    print("  python3 src/graph/review_graph_candidates_db.py Boxer")
    print("  python3 src/graph/review_graph_candidates_db.py Squealer")
    print("  python3 src/graph/review_graph_candidates_db.py --all")


def main() -> None:
    """Search and display graph fact candidates from Postgres."""
    search_term = " ".join(sys.argv[1:]).strip()

    if not search_term:
        print_usage()
        return

    database_url = load_database_url()
    connection = None

    try:
        connection = psycopg.connect(database_url)

        with connection.cursor() as cursor:
            if search_term == "--all":
                candidates = load_all_candidates(cursor)
                print("Reviewing all graph fact candidates")
            else:
                candidates = search_candidates(cursor, search_term)
                print(f"Search term: {search_term}")

        print(f"Matching candidates: {len(candidates)}")

        if not candidates:
            print("No matching graph fact candidates found.")
            return

        print()

        for rank, candidate in enumerate(candidates, start=1):
            print_candidate(candidate, rank)
            print()
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
