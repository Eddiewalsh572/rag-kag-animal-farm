import sys
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer


# This file lives in src/generation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so this script can reuse project
# modules when run directly from the terminal.
sys.path.append(str(PROJECT_ROOT))

from src.graph.extract_question_graph_facts_db import (  # noqa: E402
    QUERY_EXTRACTED_CONFIDENCE,
    build_extraction_prompt,
    call_opencode,
    delete_existing_question_extractions,
    fetch_neighbor_chunks_from_db,
    insert_question_extracted_fact,
    load_database_url,
    parse_graph_facts,
)
from src.retrieval.retrieve_chunks_db import (  # noqa: E402
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    TOP_K,
    preview_text,
    retrieve_top_chunks_from_db,
)


def format_text_evidence_for_prompt(evidence_chunks: list[dict]) -> str:
    """Format text chunks for the final KAG answer prompt."""
    evidence_parts = []

    for chunk in evidence_chunks:
        evidence_parts.append(
            f"[Chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n---\n\n".join(evidence_parts)


def format_graph_facts_for_prompt(graph_facts: list[dict]) -> str:
    """Format extracted graph facts for the final KAG answer prompt."""
    if not graph_facts:
        return "No graph facts were extracted."

    formatted_facts = []

    for fact in graph_facts:
        formatted_facts.append(
            f"{fact['source_entity']} --{fact['relation']}--> "
            f"{fact['target_entity']}\n"
            f"Description: {fact['description']}\n"
            f"Evidence chunks: {fact.get('evidence_chunk_ids', [])}"
        )

    return "\n\n---\n\n".join(formatted_facts)


def build_kag_answer_prompt(
    question: str,
    evidence_chunks: list[dict],
    graph_facts: list[dict],
) -> str:
    """Build the final DB-backed KAG answer prompt."""
    text_evidence = format_text_evidence_for_prompt(evidence_chunks)
    graph_evidence = format_graph_facts_for_prompt(graph_facts)

    return f"""Answer the user's question about Animal Farm.
Use only the provided text evidence and extracted graph facts.
The graph facts are extracted from the provided evidence, but if the text evidence and graph facts conflict, trust the text evidence.
Be careful with claims, rumors, propaganda, or reported speech.
If evidence only says Squealer claimed something, say that Squealer claimed it.
Answer in 4-6 sentences.
Cite text chunks like [Chunk 112].
Do not cite graph facts as if they are chunks.
Do not invent citations.
Do not use outside knowledge.

Question:
{question}

Extracted graph facts:
{graph_evidence}

Text evidence:
{text_evidence}

Answer:"""


def generate_kag_answer(prompt: str) -> str:
    """Call OpenCode/GPT-5.5 to generate the final KAG answer."""
    return call_opencode(prompt)


def store_question_graph_facts(question: str, graph_facts: list[dict]) -> None:
    """Store question-focused graph facts in Postgres for later review."""
    database_url = load_database_url()
    connection = psycopg.connect(database_url)

    try:
        with connection.cursor() as cursor:
            delete_existing_question_extractions(cursor, question)

            for fact in graph_facts:
                insert_question_extracted_fact(cursor, fact, question)

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def print_graph_facts_used(graph_facts: list[dict]) -> None:
    """Print extracted graph facts used by the answer."""
    print("\nExtracted graph facts used:")

    if not graph_facts:
        print("No graph facts were extracted.")
        return

    for index, fact in enumerate(graph_facts, start=1):
        print(
            f"{index}. {fact['source_entity']} --{fact['relation']}--> "
            f"{fact['target_entity']}"
        )
        print(f"   Description: {fact['description']}")
        print(f"   Evidence chunks: {fact.get('evidence_chunk_ids', [])}")


def print_text_evidence_used(evidence_chunks: list[dict]) -> None:
    """Print text chunks used by the answer."""
    print("\nText evidence used:")

    for index, chunk in enumerate(evidence_chunks, start=1):
        print(f"{index}. Chunk index: {chunk['chunk_index']}")
        print(f"   DB chunk id: {chunk['db_chunk_id']}")
        print(f"   Source: {chunk['source']}")
        print(f"   Preview: {preview_text(chunk['text'])}")


def main() -> None:
    """Generate a DB-backed KAG answer from retrieved chunks and extracted facts."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print('Usage: python3 src/generation/generate_kag_answer_db.py "What happens to Boxer?"')
        return

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    retrieved_chunks = retrieve_top_chunks_from_db(question, model, TOP_K)

    if not retrieved_chunks:
        print("No chunks were found in Postgres. Run store_chunks_embeddings.py first.")
        return

    evidence_chunks = fetch_neighbor_chunks_from_db(retrieved_chunks)
    evidence_chunk_indexes = set()

    for chunk in evidence_chunks:
        evidence_chunk_indexes.add(chunk["chunk_index"])

    extraction_prompt = build_extraction_prompt(question, evidence_chunks)
    extraction_response = call_opencode(extraction_prompt)
    graph_facts = parse_graph_facts(extraction_response, evidence_chunk_indexes)
    store_question_graph_facts(question, graph_facts)

    answer_prompt = build_kag_answer_prompt(question, evidence_chunks, graph_facts)
    answer = generate_kag_answer(answer_prompt)

    print(f"Question: {question}")
    print("Retrieval source: Postgres + pgvector")
    print("Graph extraction source: OpenCode/GPT-5.5 over retrieved DB chunks")

    print("\nGenerated DB-backed KAG answer:")
    print(answer)

    print_graph_facts_used(graph_facts)
    print_text_evidence_used(evidence_chunks)

    print("\nGenerated DB-backed KAG answer repeated")
    print(answer)

    print(
        "\nSummary:"
        f"\nStored {len(graph_facts)} graph candidates with confidence "
        f"{QUERY_EXTRACTED_CONFIDENCE}."
    )


if __name__ == "__main__":
    main()
