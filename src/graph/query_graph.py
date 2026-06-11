import json
import re
import sys
from pathlib import Path


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The graph query script reads the manual seed graph facts.
GRAPH_FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_graph_facts.json"

MAX_GRAPH_FACTS = 5

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "from",
    "by",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "how",
    "what",
    "why",
    "when",
    "where",
    "who",
    "whom",
    "which",
    "role",
    "play",
    "plays",
    "played",
}


def load_graph_facts(path: Path) -> dict:
    """Load the full graph facts JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def tokenize_query(query: str) -> list[str]:
    """Turn a natural language query into simple searchable tokens."""
    lowercase_query = query.lower()
    words = re.findall(r"[a-z]+", lowercase_query)
    tokens = []

    for word in words:
        if len(word) < 3:
            continue

        if word in STOPWORDS:
            continue

        tokens.append(word)

    return tokens


def graph_fact_search_text(fact: dict) -> str:
    """Combine searchable graph fact fields into one lowercase string."""
    searchable_fields = [
        "source_entity",
        "relation",
        "target_entity",
        "description",
    ]
    searchable_parts = []

    for field in searchable_fields:
        searchable_parts.append(str(fact.get(field, "")).lower())

    return " ".join(searchable_parts)


def score_graph_fact(fact: dict, query: str) -> int:
    """Count how many query tokens appear in a graph fact."""
    query_tokens = tokenize_query(query)
    searchable_text = graph_fact_search_text(fact)
    score = 0

    for token in query_tokens:
        if token in searchable_text:
            score += 1

    return score


def fact_matches_query(fact: dict, query: str) -> bool:
    """Check whether the query matches the searchable fact fields."""
    query_tokens = tokenize_query(query)

    if not query_tokens:
        lowercase_query = query.lower()
        searchable_text = graph_fact_search_text(fact)
        return lowercase_query in searchable_text

    return score_graph_fact(fact, query) > 0


def find_matching_facts(graph_data: dict, query: str) -> list[dict]:
    """Return the best matching graph facts for the query."""
    matching_facts = []

    for fact in graph_data["facts"]:
        score = score_graph_fact(fact, query)

        if score > 0:
            fact_copy = fact.copy()
            fact_copy["match_score"] = score
            matching_facts.append(fact_copy)

    matching_facts.sort(key=lambda fact: fact["match_score"], reverse=True)
    return matching_facts[:MAX_GRAPH_FACTS]


def print_fact(fact: dict) -> None:
    """Print one graph fact in a readable format."""
    evidence_chunk_ids = []

    for chunk_id in fact["evidence_chunk_ids"]:
        evidence_chunk_ids.append(str(chunk_id))

    print(fact["fact_id"])
    print(f"{fact['source_entity']} --{fact['relation']}--> {fact['target_entity']}")
    print(f"Description: {fact['description']}")
    print(f"Evidence chunks: {', '.join(evidence_chunk_ids)}")
    print(f"Confidence: {fact['confidence']}")

    if "match_score" in fact:
        print(f"Match score: {fact['match_score']}")


def main() -> None:
    """Search the manual graph facts from the terminal."""
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        print('Usage: python3 src/graph/query_graph.py "dogs"')
        return

    graph_data = load_graph_facts(GRAPH_FACTS_PATH)
    matching_facts = find_matching_facts(graph_data, query)

    print(f"Graph query: {query}")
    print(f"Matching facts: {len(matching_facts)}")

    if not matching_facts:
        print("No matching graph facts found.")
        return

    print()

    for fact in matching_facts:
        print_fact(fact)
        print()


if __name__ == "__main__":
    main()
