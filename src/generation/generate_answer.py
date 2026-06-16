import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# This file lives in src/generation/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to Python's import path so this script can reuse
# functions from src/retrieval/retrieve_chunks.py when run directly.
sys.path.append(str(PROJECT_ROOT))

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


ENV_PATH = PROJECT_ROOT / ".env"
OPENCODE_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_MODEL = "gpt-5.5"
MAX_EVIDENCE_CHARACTERS = 1500


def trim_evidence_text(text: str, max_characters: int = MAX_EVIDENCE_CHARACTERS) -> str:
    """Keep evidence chunks readable without making the prompt too large."""
    trimmed_text = " ".join(text.split())

    if len(trimmed_text) <= max_characters:
        return trimmed_text

    return trimmed_text[:max_characters].rstrip() + "..."


def build_prompt(question: str, top_chunks: list[dict]) -> str:
    """Build the prompt that will be sent to OpenCode."""
    evidence_parts = []

    for chunk in top_chunks:
        evidence_parts.append(
            "Chunk ID: "
            f"{chunk['chunk_id']}\n"
            f"Similarity: {chunk['similarity']:.4f}\n"
            "Text:\n"
            f"{trim_evidence_text(chunk['text'])}"
        )

    evidence_text = "\n\n---\n\n".join(evidence_parts)

    return f"""You are answering questions about George Orwell's Animal Farm.
Use only the evidence below.
If the evidence is not enough, say the evidence is not enough.
Answer in 3-5 sentences.
Every major factual claim should include a chunk citation like [Chunk 70].
Only cite chunks that directly support the claim.
Do not invent citations.
Do not invent facts.

Question:
{question}

Evidence:
{evidence_text}

Answer:"""


def add_neighbor_chunks(
    top_chunks: list[dict],
    embedded_chunks: list[dict],
    neighbor_distance: int = 1,
) -> list[dict]:
    """Add nearby chunks so the prompt has surrounding context."""
    chunks_by_id = {}

    for chunk in embedded_chunks:
        chunks_by_id[chunk["chunk_id"]] = chunk

    retrieved_chunk_ids = set()

    for chunk in top_chunks:
        retrieved_chunk_ids.add(chunk["chunk_id"])

    evidence_chunks = []
    added_chunk_ids = set()

    for retrieved_chunk in top_chunks:
        chunk_id = retrieved_chunk["chunk_id"]
        start_id = chunk_id - neighbor_distance
        end_id = chunk_id + neighbor_distance

        for neighbor_id in range(start_id, end_id + 1):
            if neighbor_id not in chunks_by_id:
                continue

            if neighbor_id in added_chunk_ids:
                continue

            chunk_copy = chunks_by_id[neighbor_id].copy()

            if neighbor_id in retrieved_chunk_ids:
                chunk_copy["source"] = "retrieved"
                chunk_copy["similarity"] = retrieved_chunk["similarity"]
            else:
                chunk_copy["source"] = "neighbor"
                chunk_copy["similarity"] = 0.0

            evidence_chunks.append(chunk_copy)
            added_chunk_ids.add(neighbor_id)

    return evidence_chunks


def generate_opencode_answer(prompt: str) -> str:
    """Send the prompt to OpenCode and return the generated answer."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("OPENCODE_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENCODE_KEY was not found. Add it to your local .env file before "
            "running this script."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0",
    }
    payload = {
        "model": OPENCODE_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENCODE_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("The OpenCode generation request failed.") from error

    return response.json()["choices"][0]["message"]["content"].strip()


def main() -> None:
    """Retrieve evidence for a question, then generate an answer from it."""
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = TEST_QUESTION

    retrieval_query = expand_question_for_retrieval(question)
    embedded_chunks = load_embedded_chunks(EMBEDDINGS_PATH)

    model = SentenceTransformer(DEFAULT_LOCAL_EMBEDDING_MODEL)
    top_chunks = retrieve_top_chunks(retrieval_query, embedded_chunks, model, TOP_K)
    evidence_chunks = add_neighbor_chunks(top_chunks, embedded_chunks)

    prompt = build_prompt(question, evidence_chunks)
    answer = generate_opencode_answer(prompt)

    print(f"Question: {question}")
    print(f"Retrieval query: {retrieval_query}")

    print("\nGenerated answer:")
    print(answer)

    print("\nEvidence used:")

    for rank, chunk in enumerate(evidence_chunks, start=1):
        print(f"{rank}. Chunk ID: {chunk['chunk_id']}")
        print(f"   Source: {chunk['source']}")
        print(f"   Similarity: {chunk['similarity']:.4f}")
        print("   Preview:")
        print(f"   {preview_text(chunk['text'])}\n")

    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
