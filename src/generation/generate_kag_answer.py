import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


# This file lives in src/generation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so this script can reuse project
# modules when run directly from the terminal.
sys.path.append(str(PROJECT_ROOT))

from src.generation.generate_answer import (  # noqa: E402
    add_neighbor_chunks,
    generate_opencode_answer,
    trim_evidence_text,
)
from src.graph.query_graph import (  # noqa: E402
    GRAPH_FACTS_PATH,
    find_matching_facts,
    load_graph_facts,
)
from src.retrieval.retrieve_chunks import (  # noqa: E402
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    TEST_QUESTION,
    TOP_K,
    expand_question_for_retrieval,
    load_embedded_chunks,
    preview_text,
    retrieve_top_chunks,
)


def format_graph_facts(graph_facts: list[dict]) -> str:
    """Format graph facts clearly for the KAG prompt."""
    if not graph_facts:
        return "No matching graph facts were found."

    formatted_facts = []

    for fact in graph_facts:
        evidence_chunk_ids = []

        for chunk_id in fact["evidence_chunk_ids"]:
            evidence_chunk_ids.append(str(chunk_id))

        formatted_facts.append(
            f"Fact ID: {fact['fact_id']}\n"
            "Relationship: "
            f"{fact['source_entity']} --{fact['relation']}--> "
            f"{fact['target_entity']}\n"
            f"Description: {fact['description']}\n"
            f"Evidence chunks: {', '.join(evidence_chunk_ids)}"
        )

    return "\n\n---\n\n".join(formatted_facts)


def build_kag_prompt(
    question: str,
    evidence_chunks: list[dict],
    graph_facts: list[dict],
) -> str:
    """Build a prompt that combines graph facts and retrieved text chunks."""
    graph_facts_text = format_graph_facts(graph_facts)
    chunk_parts = []

    for chunk in evidence_chunks:
        source = chunk.get("source", "retrieved")

        chunk_parts.append(
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Source: {source}\n"
            f"Similarity: {chunk['similarity']:.4f}\n"
            "Text:\n"
            f"{trim_evidence_text(chunk['text'])}"
        )

    retrieved_chunks_text = "\n\n---\n\n".join(chunk_parts)

    return f"""You are answering questions about George Orwell's Animal Farm.
Use both the structured graph facts and the retrieved text evidence.
Graph facts summarize entity relationships and are grounded in evidence chunk IDs.
Retrieved chunks contain the original text evidence.
If the graph facts and retrieved chunks disagree, trust the retrieved chunks.
If the evidence is not enough, say the evidence is not enough.
Answer in 3-5 sentences.
Cite chunk IDs for major factual claims using [Chunk 51].
When using a graph fact, mention the fact relationship in natural language but still cite the supporting chunk IDs.
Do not invent facts.
Do not invent citations.

Question:
{question}

Structured graph facts:
{graph_facts_text}

Retrieved text chunks:
{retrieved_chunks_text}

Answer:"""


def print_graph_facts_used(graph_facts: list[dict]) -> None:
    """Print graph facts in a readable terminal format."""
    if not graph_facts:
        print("No matching graph facts found.")
        return

    for fact in graph_facts:
        evidence_chunk_ids = []

        for chunk_id in fact["evidence_chunk_ids"]:
            evidence_chunk_ids.append(str(chunk_id))

        print(fact["fact_id"])
        print(f"{fact['source_entity']} --{fact['relation']}--> {fact['target_entity']}")
        print(f"Description: {fact['description']}")
        print(f"Evidence chunks: {', '.join(evidence_chunk_ids)}")
        print()


def print_text_evidence_used(evidence_chunks: list[dict]) -> None:
    """Print retrieved text evidence in a readable terminal format."""
    for rank, chunk in enumerate(evidence_chunks, start=1):
        source = chunk.get("source", "retrieved")

        print(f"{rank}. Chunk ID: {chunk['chunk_id']}")
        print(f"   Source: {source}")
        print(f"   Similarity: {chunk['similarity']:.4f}")
        print("   Preview:")
        print(f"   {preview_text(chunk['text'])}\n")


def main() -> None:
    """Generate an answer using both RAG chunks and KAG graph facts."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = TEST_QUESTION

    retrieval_query = expand_question_for_retrieval(question)
    embedded_chunks = load_embedded_chunks(EMBEDDINGS_PATH)

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    top_chunks = retrieve_top_chunks(retrieval_query, embedded_chunks, model, TOP_K)
    evidence_chunks = add_neighbor_chunks(top_chunks, embedded_chunks)

    graph_data = load_graph_facts(GRAPH_FACTS_PATH)
    graph_facts = find_matching_facts(graph_data, question)

    prompt = build_kag_prompt(question, evidence_chunks, graph_facts)
    answer = generate_opencode_answer(prompt)

    print(f"Question: {question}")
    print(f"Retrieval query: {retrieval_query}")

    print("\nGenerated KAG answer:")
    print(answer)

    print("\nGraph facts used:")
    print_graph_facts_used(graph_facts)

    print("Text evidence used:")
    print_text_evidence_used(evidence_chunks)


if __name__ == "__main__":
    main()
