import json
import sys
from pathlib import Path


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The graph query script reads the manual seed graph facts.
GRAPH_FACTS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_graph_facts.json"


def load_graph_facts(path: Path) -> dict:
    """Load the full graph facts JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def fact_matches_query(fact: dict, query: str) -> bool:
    """Check whether the query appears in one of the searchable fact fields."""
    lowercase_query = query.lower()
    searchable_fields = [
        "source_entity",
        "relation",
        "target_entity",
        "description",
    ]

    for field in searchable_fields:
        field_value = str(fact.get(field, "")).lower()

        if lowercase_query in field_value:
            return True

    return False


def find_matching_facts(graph_data: dict, query: str) -> list[dict]:
    """Return graph facts that match the query."""
    matching_facts = []

    for fact in graph_data["facts"]:
        if fact_matches_query(fact, query):
            matching_facts.append(fact)

    return matching_facts


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
