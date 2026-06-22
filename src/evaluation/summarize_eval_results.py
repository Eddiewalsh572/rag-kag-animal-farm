import json
import re
from pathlib import Path
from typing import Optional


# This file lives in src/evaluation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
    / "rag_kag_eval_results.json"
)

# Matches simple chunk citations like [Chunk 112].
CHUNK_CITATION_PATTERN = re.compile(r"\[Chunk \d+\]")


def load_eval_results(path: Path) -> list[dict]:
    """Load the saved RAG/KAG evaluation results from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation results file was not found: {path}\n\n"
            "Run one of these commands first:\n"
            "python3 src/evaluation/run_rag_kag_eval.py --id boxer_fate\n"
            "python3 src/evaluation/run_rag_kag_eval.py"
        )

    with path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError("The evaluation results file should contain a JSON array.")

    return results


def answer_exists(answer: Optional[str]) -> bool:
    """Return True when an answer is present and not empty."""
    return bool(answer and answer.strip())


def count_keyword_hits(answer: Optional[str], expected_keywords: list[str]) -> int:
    """Count expected keywords that appear in the answer, ignoring case."""
    if not answer:
        return 0

    lowercase_answer = answer.lower()
    keyword_hits = 0

    for keyword in expected_keywords:
        if keyword.lower() in lowercase_answer:
            keyword_hits += 1

    return keyword_hits


def has_chunk_citation(answer: Optional[str]) -> bool:
    """Check whether an answer includes citations like [Chunk 112]."""
    if not answer:
        return False

    return bool(CHUNK_CITATION_PATTERN.search(answer))


def count_items(value: Optional[list]) -> int:
    """Safely count list fields that may be null when a mode was skipped."""
    if not value:
        return 0

    return len(value)


def keyword_hit_rate(hit_count: int, keyword_count: int) -> float:
    """Convert keyword hits into a simple 0.0 to 1.0 rate."""
    if keyword_count == 0:
        return 0.0

    return hit_count / keyword_count


def print_question_summary(result: dict) -> tuple[float, float, bool, bool]:
    """Print one result item and return values needed for the final summary."""
    expected_keywords = result.get("expected_keywords", [])
    total_keywords = len(expected_keywords)
    rag_answer = result.get("rag_answer")
    kag_answer = result.get("kag_answer")

    rag_keyword_hits = count_keyword_hits(rag_answer, expected_keywords)
    kag_keyword_hits = count_keyword_hits(kag_answer, expected_keywords)
    rag_hit_rate = keyword_hit_rate(rag_keyword_hits, total_keywords)
    kag_hit_rate = keyword_hit_rate(kag_keyword_hits, total_keywords)
    kag_graph_fact_count = count_items(result.get("kag_graph_facts"))
    has_both_answers = answer_exists(rag_answer) and answer_exists(kag_answer)

    print("---")
    print(f"ID: {result.get('id')}")
    print(f"Question: {result.get('question')}")
    print(f"Category: {result.get('category')}")
    print(f"Difficulty: {result.get('difficulty')}")
    print(f"Expected best answer type: {result.get('best_answer_type')}")
    print(f"RAG answer exists: {answer_exists(rag_answer)}")
    print(f"KAG answer exists: {answer_exists(kag_answer)}")
    print(f"RAG keyword hits: {rag_keyword_hits}")
    print(f"KAG keyword hits: {kag_keyword_hits}")
    print(f"Total expected keywords: {total_keywords}")
    print(f"RAG has chunk citations: {has_chunk_citation(rag_answer)}")
    print(f"KAG has chunk citations: {has_chunk_citation(kag_answer)}")
    print(f"RAG evidence chunks: {count_items(result.get('rag_evidence_chunks'))}")
    print(f"KAG evidence chunks: {count_items(result.get('kag_evidence_chunks'))}")
    print(f"KAG graph facts: {kag_graph_fact_count}")

    return rag_hit_rate, kag_hit_rate, kag_graph_fact_count > 0, has_both_answers


def average(values: list[float]) -> float:
    """Compute a simple average safely."""
    if not values:
        return 0.0

    return sum(values) / len(values)


def main() -> None:
    """Print a concise summary of saved RAG/KAG evaluation results."""
    results = load_eval_results(RESULTS_PATH)
    rag_hit_rates = []
    kag_hit_rates = []
    questions_with_kag_graph_facts = 0
    questions_with_both_answers = 0

    print("RAG/KAG evaluation summary")
    print(f"Source file: {RESULTS_PATH}")

    for result in results:
        rag_hit_rate, kag_hit_rate, has_graph_facts, has_both_answers = (
            print_question_summary(result)
        )
        rag_hit_rates.append(rag_hit_rate)
        kag_hit_rates.append(kag_hit_rate)

        if has_graph_facts:
            questions_with_kag_graph_facts += 1

        if has_both_answers:
            questions_with_both_answers += 1

    print("---")
    print("Overall summary")
    print(f"Total questions summarized: {len(results)}")
    print(f"Average RAG keyword hit rate: {average(rag_hit_rates):.2%}")
    print(f"Average KAG keyword hit rate: {average(kag_hit_rates):.2%}")
    print(f"Questions with KAG graph facts: {questions_with_kag_graph_facts}")
    print(f"Questions with both RAG and KAG answers: {questions_with_both_answers}")


if __name__ == "__main__":
    main()
