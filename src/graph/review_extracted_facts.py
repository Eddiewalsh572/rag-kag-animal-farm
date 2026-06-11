import json
import sys
from pathlib import Path


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXTRACTED_GRAPH_FACTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "animal_farm_graph_facts_extracted.json"
)


def load_extracted_graph_facts(path: Path) -> dict:
    """Load the extracted graph facts JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def format_fact(fact: dict) -> str:
    """Return one extracted fact as a readable multi-line string."""
    evidence_chunk_ids = []

    for chunk_id in fact.get("evidence_chunk_ids", []):
        evidence_chunk_ids.append(str(chunk_id))

    return (
        f"{fact.get('fact_id', 'missing_fact_id')}\n"
        f"{fact.get('source_entity', 'missing_source')} "
        f"--{fact.get('relation', 'missing_relation')}--> "
        f"{fact.get('target_entity', 'missing_target')}\n"
        f"Description: {fact.get('description', 'missing_description')}\n"
        f"Evidence chunks: {', '.join(evidence_chunk_ids)}\n"
        f"Confidence: {fact.get('confidence', 'missing_confidence')}"
    )


def fact_contains_term(fact: dict, term: str) -> bool:
    """Check whether a fact contains the search term in a key text field."""
    lowercase_term = term.lower()
    searchable_fields = [
        "fact_id",
        "source_entity",
        "relation",
        "target_entity",
        "description",
        "confidence",
    ]

    for field in searchable_fields:
        field_value = str(fact.get(field, "")).lower()

        if lowercase_term in field_value:
            return True

    return False


def filter_facts(facts: list[dict], term: str = "") -> list[dict]:
    """Return all facts or only facts that match the optional search term."""
    if not term:
        return facts

    matching_facts = []

    for fact in facts:
        if fact_contains_term(fact, term):
            matching_facts.append(fact)

    return matching_facts


def main() -> None:
    """Review extracted graph facts from the terminal."""
    search_term = " ".join(sys.argv[1:]).strip()
    graph_data = load_extracted_graph_facts(EXTRACTED_GRAPH_FACTS_PATH)
    facts = graph_data["facts"]
    matching_facts = filter_facts(facts, search_term)

    print("Extracted graph facts review")
    print(f"Source file: {EXTRACTED_GRAPH_FACTS_PATH}")
    print(f"Total extracted facts: {len(facts)}")
    print(f"Matching facts: {len(matching_facts)}")

    if search_term:
        print(f"Search term: {search_term}")

    if not matching_facts:
        print("No matching extracted facts found.")
        return

    print()

    for index, fact in enumerate(matching_facts):
        if index > 0:
            print("---")

        print(format_fact(fact))


if __name__ == "__main__":
    main()
