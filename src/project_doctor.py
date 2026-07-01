import json
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PATH = PROJECT_ROOT / ".venv" / "bin" / "python"
ENV_PATH = PROJECT_ROOT / ".env"
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "animal_farm_eval_questions.json"

REQUIRED_FILES = [
    PROJECT_ROOT / "src" / "db" / "check_connection.py",
    PROJECT_ROOT / "src" / "generation" / "generate_answer_db.py",
    PROJECT_ROOT / "src" / "generation" / "generate_kag_answer_db.py",
    PROJECT_ROOT / "src" / "evaluation" / "run_rag_kag_eval.py",
    PROJECT_ROOT / "src" / "evaluation" / "summarize_eval_results.py",
    EVAL_QUESTIONS_PATH,
]

TABLES_TO_CHECK = [
    "documents",
    "chunks",
    "graph_nodes",
    "graph_edges",
    "graph_fact_candidates",
]


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def check_python_environment() -> bool:
    if PYTHON_PATH.exists():
        print("Python environment: OK")
        return True

    print("Python environment: FAILED")
    print(f"Missing file: {relative_path(PYTHON_PATH)}")
    return False


def check_required_files() -> bool:
    missing_files = [path for path in REQUIRED_FILES if not path.exists()]

    if not missing_files:
        print("Required files: OK")
        return True

    print("Required files: FAILED")
    for path in missing_files:
        print(f"Missing file: {relative_path(path)}")

    return False


def check_evaluation_questions() -> bool:
    try:
        questions = json.loads(EVAL_QUESTIONS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("Evaluation questions: FAILED")
        print(f"Missing file: {relative_path(EVAL_QUESTIONS_PATH)}")
        return False
    except json.JSONDecodeError as error:
        print("Evaluation questions: FAILED")
        print(f"Could not load JSON: {error}")
        return False

    if not isinstance(questions, list):
        print("Evaluation questions: FAILED")
        print("Expected the evaluation questions file to contain a JSON list.")
        return False

    print(f"Evaluation questions: {len(questions)} found")
    return True


def load_database_url() -> str | None:
    load_dotenv(dotenv_path=ENV_PATH)
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        print("Database connection: FAILED")
        print("DATABASE_URL was not found in .env")
        return None

    return database_url


def short_error_message(error: Exception) -> str:
    return str(error).splitlines()[0]


def connect_to_database(database_url: str):
    try:
        import psycopg

        return psycopg.connect(database_url)
    except ImportError:
        print("Database connection: FAILED")
        print("Missing Python package: psycopg")
    except Exception as error:
        print("Database connection: FAILED")
        print(f"Reason: {short_error_message(error)}")
        print(
            "Possible fix: make sure Docker Desktop is running, then run "
            "docker compose up -d"
        )

    return None


def check_pgvector(connection) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');"
        )
        vector_exists = cursor.fetchone()[0]

    if vector_exists:
        print("pgvector extension: OK")
        return True

    print("pgvector extension: FAILED")
    return False


def get_table_counts(connection) -> dict[str, int]:
    counts = {}

    with connection.cursor() as cursor:
        for table_name in TABLES_TO_CHECK:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            counts[table_name] = cursor.fetchone()[0]

    return counts


def get_chunks_with_embeddings_count(connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
        return cursor.fetchone()[0]


def print_database_content(counts: dict[str, int], chunks_with_embeddings: int) -> None:
    print()
    print("Database content:")
    print()
    print(f"* documents: {counts['documents']}")
    print(f"* chunks: {counts['chunks']}")
    print(f"* chunks with embeddings: {chunks_with_embeddings}")
    print(f"* graph_nodes: {counts['graph_nodes']}")
    print(f"* graph_edges: {counts['graph_edges']}")
    print(f"* graph_fact_candidates: {counts['graph_fact_candidates']}")

    if counts["graph_nodes"] == 0 and counts["graph_edges"] == 0:
        print()
        print(
            "Note: graph_nodes and graph_edges may be 0 because the current KAG "
            "flow stores graph-style facts as candidates first."
        )


def check_database() -> bool:
    database_url = load_database_url()
    if not database_url:
        return False

    connection = connect_to_database(database_url)
    if connection is None:
        return False

    try:
        print("Database connection: OK")
        pgvector_ok = check_pgvector(connection)
        counts = get_table_counts(connection)
        chunks_with_embeddings = get_chunks_with_embeddings_count(connection)
        print_database_content(counts, chunks_with_embeddings)
        return pgvector_ok
    except Exception as error:
        print("Database content: FAILED")
        print(f"Reason: {short_error_message(error)}")
        return False
    finally:
        connection.close()


def main() -> int:
    print("Animal Farm RAG/KAG Project Doctor")
    print()

    checks = [
        check_python_environment(),
        check_required_files(),
        check_evaluation_questions(),
        check_database(),
    ]

    print()
    if all(checks):
        print("Status: Ready for demo")
        return 0

    print("Status: Not ready yet")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
