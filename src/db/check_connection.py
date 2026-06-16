import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


# This file lives in src/db/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

TABLES_TO_CHECK = [
    "documents",
    "chunks",
    "graph_nodes",
    "graph_edges",
    "graph_fact_candidates",
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


def print_database_name(connection: psycopg.Connection) -> None:
    """Print the name of the connected database."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_database();")
        database_name = cursor.fetchone()[0]

    print(f"Connected database: {database_name}")


def print_vector_extension_status(connection: psycopg.Connection) -> None:
    """Check whether the pgvector extension is installed."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');"
        )
        vector_exists = cursor.fetchone()[0]

    if vector_exists:
        print("pgvector extension: installed")
    else:
        print("pgvector extension: missing")


def print_table_row_counts(connection: psycopg.Connection) -> None:
    """Print row counts for the main project tables."""
    print("Table row counts:")

    with connection.cursor() as cursor:
        for table_name in TABLES_TO_CHECK:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"- {table_name}: {row_count}")


def main() -> None:
    """Connect to Postgres and print a small health report."""
    database_url = load_database_url()
    connection = None

    try:
        connection = psycopg.connect(database_url)
        print_database_name(connection)
        print_vector_extension_status(connection)
        print_table_row_counts(connection)
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
