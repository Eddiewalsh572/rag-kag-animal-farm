import json
import re
from pathlib import Path
from typing import Optional, Union


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
CHUNK_CITATION_PATTERN = re.compile(r"\[Chunk\s+\d+\]")

ScoreDetails = dict[str, Union[int, float, bool]]


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


def keyword_hit_rate(answer: Optional[str], expected_keywords: list[str]) -> float:
    """Return the fraction of expected keywords that appear in the answer."""
    if not expected_keywords:
        return 0.0

    return count_keyword_hits(answer, expected_keywords) / len(expected_keywords)


def has_chunk_citations(answer: Optional[str]) -> bool:
    """Check whether an answer includes citations like [Chunk 112]."""
    if not answer:
        return False

    return bool(CHUNK_CITATION_PATTERN.search(answer))


def compute_answer_score(
    answer: Optional[str], expected_keywords: list[str]
) -> ScoreDetails:
    """Compute a lightweight 0-100 score for one saved answer."""
    if not answer_exists(answer):
        return {
            "keyword_hits": 0,
            "keyword_total": len(expected_keywords),
            "keyword_hit_rate": 0.0,
            "keyword_score": 0,
            "has_citations": False,
            "citation_score": 0,
            "simple_score": 0,
        }

    hit_rate = keyword_hit_rate(answer, expected_keywords)
    keyword_score = round(hit_rate * 80)
    citation_score = 20 if has_chunk_citations(answer) else 0

    return {
        "keyword_hits": count_keyword_hits(answer, expected_keywords),
        "keyword_total": len(expected_keywords),
        "keyword_hit_rate": hit_rate,
        "keyword_score": keyword_score,
        "has_citations": citation_score > 0,
        "citation_score": citation_score,
        "simple_score": keyword_score + citation_score,
    }


def choose_suggested_winner(
    rag_score: int,
    kag_score: int,
    rag_answer: Optional[str],
    kag_answer: Optional[str],
) -> str:
    """Suggest the higher-scoring answer, while leaving missing pairs for review."""
    if not answer_exists(rag_answer) and not answer_exists(kag_answer):
        return "manual review"

    if rag_score > kag_score:
        return "RAG"

    if kag_score > rag_score:
        return "KAG"

    return "tie"


def count_items(value: Optional[list]) -> int:
    """Safely count list fields that may be null when a mode was skipped."""
    if not value:
        return 0

    return len(value)


def print_question_summary(
    result: dict,
) -> tuple[float, float, int, int, bool, bool, str]:
    """Print one result item and return values needed for the final summary."""
    expected_keywords = result.get("expected_keywords", [])
    rag_answer = result.get("rag_answer")
    kag_answer = result.get("kag_answer")

    rag_score = compute_answer_score(rag_answer, expected_keywords)
    kag_score = compute_answer_score(kag_answer, expected_keywords)
    kag_graph_fact_count = count_items(result.get("kag_graph_facts"))
    has_both_answers = answer_exists(rag_answer) and answer_exists(kag_answer)
    suggested_winner = choose_suggested_winner(
        int(rag_score["simple_score"]),
        int(kag_score["simple_score"]),
        rag_answer,
        kag_answer,
    )

    print("---")
    print(f"ID: {result.get('id')}")
    print(f"Question: {result.get('question')}")
    print(f"Category: {result.get('category')}")
    print(f"Difficulty: {result.get('difficulty')}")
    print(f"Expected best answer type: {result.get('best_answer_type')}")
    print(f"RAG answer exists: {answer_exists(rag_answer)}")
    print(f"KAG answer exists: {answer_exists(kag_answer)}")
    print(
        "RAG keyword hits: "
        f"{rag_score['keyword_hits']}/{rag_score['keyword_total']}"
    )
    print(
        "KAG keyword hits: "
        f"{kag_score['keyword_hits']}/{kag_score['keyword_total']}"
    )
    print(f"RAG citation check: {rag_score['has_citations']}")
    print(f"KAG citation check: {kag_score['has_citations']}")
    print(
        f"RAG evidence chunk count: {count_items(result.get('rag_evidence_chunks'))}"
    )
    print(
        f"KAG evidence chunk count: {count_items(result.get('kag_evidence_chunks'))}"
    )
    print(f"KAG graph fact count: {kag_graph_fact_count}")
    print(f"RAG simple score: {rag_score['simple_score']}/100")
    print(f"KAG simple score: {kag_score['simple_score']}/100")
    print(f"Suggested winner: {suggested_winner}")

    return (
        float(rag_score["keyword_hit_rate"]),
        float(kag_score["keyword_hit_rate"]),
        int(rag_score["simple_score"]),
        int(kag_score["simple_score"]),
        kag_graph_fact_count > 0,
        has_both_answers,
        suggested_winner,
    )


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
    rag_simple_scores = []
    kag_simple_scores = []
    questions_with_kag_graph_facts = 0
    questions_with_both_answers = 0
    suggested_winner_counts = {
        "RAG": 0,
        "KAG": 0,
        "tie": 0,
        "manual review": 0,
    }

    print("RAG/KAG evaluation summary")
    print(f"Source file: {RESULTS_PATH}")

    for result in results:
        (
            rag_hit_rate,
            kag_hit_rate,
            rag_simple_score,
            kag_simple_score,
            has_graph_facts,
            has_both_answers,
            suggested_winner,
        ) = (
            print_question_summary(result)
        )
        rag_hit_rates.append(rag_hit_rate)
        kag_hit_rates.append(kag_hit_rate)
        rag_simple_scores.append(rag_simple_score)
        kag_simple_scores.append(kag_simple_score)
        suggested_winner_counts[suggested_winner] += 1

        if has_graph_facts:
            questions_with_kag_graph_facts += 1

        if has_both_answers:
            questions_with_both_answers += 1

    print("---")
    print("Overall summary")
    print(f"Total questions summarized: {len(results)}")
    print(f"Average RAG keyword hit rate: {average(rag_hit_rates):.2%}")
    print(f"Average KAG keyword hit rate: {average(kag_hit_rates):.2%}")
    print(f"Average RAG simple score: {average(rag_simple_scores):.1f}/100")
    print(f"Average KAG simple score: {average(kag_simple_scores):.1f}/100")
    print(f"Questions with KAG graph facts: {questions_with_kag_graph_facts}")
    print(f"Questions with both RAG and KAG answers: {questions_with_both_answers}")
    print("Suggested winner counts:")
    print(f"  RAG: {suggested_winner_counts['RAG']}")
    print(f"  KAG: {suggested_winner_counts['KAG']}")
    print(f"  tie: {suggested_winner_counts['tie']}")
    print(f"  manual review: {suggested_winner_counts['manual review']}")


if __name__ == "__main__":
    main()
