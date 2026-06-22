import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# This file lives in src/evaluation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so this script can import the
# existing DB-backed generation modules when run directly from the terminal.
sys.path.append(str(PROJECT_ROOT))

from src.generation.generate_answer_db import generate_rag_answer_for_question  # noqa: E402
from src.generation.generate_kag_answer_db import generate_kag_answer_for_question  # noqa: E402
from src.retrieval.retrieve_chunks_db import preview_text  # noqa: E402


EVAL_QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "animal_farm_eval_questions.json"
)
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "rag_kag_eval_results.json"


def parse_args() -> argparse.Namespace:
    """Read simple command-line options for choosing eval questions and mode."""
    parser = argparse.ArgumentParser(
        description="Run DB-backed RAG/KAG evaluation questions for manual review."
    )
    parser.add_argument(
        "--id",
        dest="question_id",
        help="Run only the evaluation question with this exact id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N questions after filtering.",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "kag", "both"],
        default="both",
        help="Choose whether to run RAG, KAG, or both. Default: both.",
    )
    return parser.parse_args()


def load_eval_questions(path: Path) -> list[dict]:
    """Load the manual evaluation questions from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation questions file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("The evaluation questions file should contain a JSON array.")

    return questions


def filter_eval_questions(
    eval_questions: list[dict],
    question_id: Optional[str],
    limit: Optional[int],
) -> Optional[list[dict]]:
    """Apply --id and --limit so development runs can stay small."""
    filtered_questions = eval_questions

    if question_id:
        filtered_questions = [
            item for item in filtered_questions if item.get("id") == question_id
        ]

        if not filtered_questions:
            print(f"No evaluation question found with id: {question_id}")
            return None

    if limit is not None:
        if limit < 1:
            print("--limit must be 1 or greater.")
            return None

        filtered_questions = filtered_questions[:limit]

    return filtered_questions


def summarize_evidence_chunks(evidence_chunks: list[dict]) -> list[dict]:
    """Keep the result file readable by saving chunk metadata and short previews."""
    summaries = []

    for chunk in evidence_chunks:
        summaries.append(
            {
                "chunk_index": chunk.get("chunk_index"),
                "db_chunk_id": chunk.get("db_chunk_id"),
                "source": chunk.get("source"),
                "similarity": chunk.get("similarity"),
                "semantic_similarity": chunk.get("semantic_similarity"),
                "keyword_score": chunk.get("keyword_score"),
                "word_count": chunk.get("word_count"),
                "preview": preview_text(chunk.get("text", "")),
            }
        )

    return summaries


def summarize_graph_facts(graph_facts: list[dict]) -> list[dict]:
    """Save the extracted graph facts used by KAG in a simple reviewable format."""
    summaries = []

    for fact in graph_facts:
        summaries.append(
            {
                "source_entity": fact.get("source_entity"),
                "relation": fact.get("relation"),
                "target_entity": fact.get("target_entity"),
                "description": fact.get("description"),
                "evidence_chunk_ids": fact.get("evidence_chunk_ids", []),
                "confidence": fact.get("confidence"),
            }
        )

    return summaries


def run_single_eval_question(eval_item: dict, mode: str) -> dict:
    """Run one question through DB-backed RAG, DB-backed KAG, or both."""
    question = eval_item["question"]
    rag_result = None
    kag_result = None

    if mode in ["rag", "both"]:
        print("Running RAG...")
        rag_result = generate_rag_answer_for_question(question)

    if mode in ["kag", "both"]:
        print("Running KAG...")
        kag_result = generate_kag_answer_for_question(question)

    return {
        "id": eval_item.get("id"),
        "question": question,
        "category": eval_item.get("category"),
        "difficulty": eval_item.get("difficulty"),
        "best_answer_type": eval_item.get("best_answer_type"),
        "expected_keywords": eval_item.get("expected_keywords", []),
        "expected_answer_points": eval_item.get("expected_answer_points", []),
        "rag_answer": rag_result["answer"] if rag_result else None,
        "kag_answer": kag_result["answer"] if kag_result else None,
        "rag_evidence_chunks": (
            summarize_evidence_chunks(rag_result.get("evidence_chunks", []))
            if rag_result
            else None
        ),
        "kag_evidence_chunks": (
            summarize_evidence_chunks(kag_result.get("evidence_chunks", []))
            if kag_result
            else None
        ),
        "kag_graph_facts": (
            summarize_graph_facts(kag_result.get("graph_facts", []))
            if kag_result
            else None
        ),
    }


def save_results(results: list[dict], path: Path) -> None:
    """Write evaluation results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main() -> None:
    """Run every manual eval question and save side-by-side RAG/KAG outputs."""
    args = parse_args()
    eval_questions = load_eval_questions(EVAL_QUESTIONS_PATH)
    eval_questions = filter_eval_questions(
        eval_questions,
        args.question_id,
        args.limit,
    )

    if eval_questions is None:
        return

    results = []
    total_questions = len(eval_questions)

    for index, eval_item in enumerate(eval_questions, start=1):
        print(
            f"\nRunning evaluation question {index}/{total_questions}: "
            f"{eval_item.get('id')}"
        )
        result = run_single_eval_question(eval_item, args.mode)
        results.append(result)

    save_results(results, RESULTS_PATH)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
