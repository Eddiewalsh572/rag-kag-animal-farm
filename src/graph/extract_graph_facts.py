import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


# This file lives in src/graph/, so parents[2] points to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "animal_farm_graph_facts_extracted.json"
ENV_PATH = PROJECT_ROOT / ".env"

OPENCODE_CHAT_COMPLETIONS_URL = "https://opencode.ai/zen/v1/chat/completions"
OPENCODE_MODEL = "gpt-5.5"

SELECTED_CHUNK_IDS = [51, 52, 56, 64, 75, 77, 78, 79, 103, 114, 116, 117, 127]


def load_chunks(path: Path) -> list[dict]:
    """Load the existing chunk JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def select_chunks(chunks: list[dict], selected_chunk_ids: list[int]) -> list[dict]:
    """Return selected chunks in the same order as selected_chunk_ids."""
    chunks_by_id = {}

    for chunk in chunks:
        chunks_by_id[chunk["chunk_id"]] = chunk

    selected_chunks = []

    for chunk_id in selected_chunk_ids:
        if chunk_id in chunks_by_id:
            selected_chunks.append(chunks_by_id[chunk_id])

    return selected_chunks


def build_extraction_prompt(chunk: dict) -> str:
    """Build a prompt asking OpenCode to extract graph facts from one chunk."""
    return f"""You are extracting knowledge graph facts from George Orwell's Animal Farm.

Extract only facts directly supported by the chunk.
Do not infer beyond the chunk.
Do not include vague themes unless the relationship is clearly supported.
Return valid JSON only.
Do not include markdown.
Do not include explanation text.

Return this exact JSON shape:

{{
  "facts": [
    {{
      "source_entity": "string",
      "relation": "string",
      "target_entity": "string",
      "description": "string",
      "evidence_chunk_ids": [{chunk['chunk_id']}],
      "confidence": "llm_extracted"
    }}
  ]
}}

Rules:
- source_entity should be short and normalized, like "Napoleon", "Squealer", "dogs", "Boxer", "Snowball", "commandments", or "animals".
- relation should be short, lowercase, and readable, like "uses", "chases", "manipulates", "rewrites", "changes", "is taken to", "silences".
- target_entity should be short and normalized.
- evidence_chunk_ids must contain the current chunk ID.
- If the chunk does not contain clear relationship facts, return:
  {{"facts": []}}
- Return at most 3 facts for the chunk.

Chunk ID:
{chunk['chunk_id']}

Chunk text:
{chunk['text']}"""


def call_opencode(prompt: str) -> dict:
    """Call OpenCode and parse the response as JSON."""
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
        raise RuntimeError("The OpenCode graph extraction request failed.") from error

    response_content = response.json()["choices"][0]["message"]["content"].strip()

    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        print("OpenCode returned text that was not valid JSON.")
        print("Raw model response:")
        print(response_content)
        return {"facts": []}


def extract_facts_from_chunks(chunks: list[dict]) -> list[dict]:
    """Extract candidate graph facts from selected chunks."""
    extracted_facts = []

    for chunk in chunks:
        print(f"Extracting graph facts from chunk {chunk['chunk_id']}...")
        prompt = build_extraction_prompt(chunk)
        extraction_result = call_opencode(prompt)

        for fact in extraction_result.get("facts", []):
            extracted_facts.append(fact)

    facts_with_ids = []

    for index, fact in enumerate(extracted_facts, start=1):
        fact_copy = fact.copy()
        fact_copy["fact_id"] = f"extracted_fact_{index:03d}"
        facts_with_ids.append(fact_copy)

    return facts_with_ids


def save_extracted_graph_facts(facts: list[dict], output_path: Path) -> None:
    """Save extracted graph facts to a separate JSON file."""
    graph_data = {
        "metadata": {
            "source": "Animal Farm",
            "description": "OpenCode-extracted graph facts for the Animal Farm RAG/KAG prototype.",
            "schema_version": "1.0",
            "extraction_method": "opencode_gpt_5_5",
            "selected_chunk_ids": SELECTED_CHUNK_IDS,
        },
        "facts": facts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(graph_data, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Extract graph facts from selected chunks and save them separately."""
    chunks = load_chunks(CHUNKS_PATH)
    selected_chunks = select_chunks(chunks, SELECTED_CHUNK_IDS)

    print(f"Selected {len(selected_chunks)} chunks for graph extraction.")

    extracted_facts = extract_facts_from_chunks(selected_chunks)
    save_extracted_graph_facts(extracted_facts, OUTPUT_PATH)

    print(f"Saved {len(extracted_facts)} extracted graph facts to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
