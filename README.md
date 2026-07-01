# Animal Farm RAG/KAG Learning Project

This is a local, beginner-friendly project for learning how a DB-backed RAG/KAG pipeline works using Animal Farm as the source document.

The goal is explainability. Each step should be easy to inspect and explain: PDF ingestion, text extraction, cleaning, chunking, local embeddings, PostgreSQL/pgvector storage, retrieval, answer generation, and lightweight evaluation.

This is not meant to be a large production system. It is a clean learning project that shows the main pieces of an end-to-end RAG/KAG workflow.

## Current Architecture

The current supported path is PostgreSQL + pgvector backed:

1. Store the source PDF locally in `data/raw/`.
2. Extract text into `data/processed/animal_farm_extracted.txt`.
3. Clean the extracted text into `data/processed/animal_farm_cleaned.txt`.
4. Split the cleaned text into chunks in `data/processed/animal_farm_chunks.json`.
5. Create local Hugging Face sentence-transformer embeddings in `data/processed/animal_farm_embeddings.json`.
6. Load chunks and 384-dimensional embeddings into PostgreSQL with pgvector.
7. Retrieve relevant chunks from PostgreSQL using pgvector similarity plus simple keyword reranking.
8. Generate RAG or KAG answers from retrieved evidence.
9. Run lightweight RAG/KAG evaluation and summarize the saved results.

The local processed files are rebuild artifacts. They are intentionally kept locally and ignored by Git so the database can be rebuilt from the source document when needed.

## RAG vs KAG

RAG retrieves text chunks from PostgreSQL and uses those chunks as evidence for an answer. The answer should cite the supporting chunks.

KAG starts with the same retrieved text evidence, then extracts graph-style facts from that evidence for the current question. Those facts help organize relationships such as who did what, who claimed something, or what event led to another event.

In the current implementation, graph-style facts are stored first as candidates in the `graph_fact_candidates` table. The `graph_nodes` and `graph_edges` tables may be empty right now, and that is okay. Promoting reviewed candidates into persistent graph nodes and edges is a future improvement.

## How To Run

Start PostgreSQL/pgvector:

```bash
docker compose up -d
```

Check project readiness:

```bash
make doctor
make check
```

Run RAG and KAG examples:

```bash
make rag QUESTION="What happens to Boxer?"
make kag QUESTION="What happens to Boxer?"
```

Run a small evaluation and summarize it:

```bash
make eval ID=boxer_fate MODE=both
make summary
```

## Makefile Commands

| Command | What it does |
| --- | --- |
| `make doctor` | Runs a readable project readiness check: required files, eval questions, database connection, pgvector, and row counts. |
| `make check` | Checks the PostgreSQL connection, pgvector extension, and main table counts. |
| `make rag QUESTION="..."` | Runs the current DB-backed RAG answer flow. |
| `make kag QUESTION="..."` | Runs the current DB-backed KAG answer flow with question-focused graph fact candidates. |
| `make eval ID=boxer_fate MODE=both` | Runs one saved evaluation question through RAG, KAG, or both. |
| `make eval MODE=both` | Runs the saved evaluation set through RAG, KAG, or both. |
| `make summary` | Summarizes the latest saved evaluation results. |
| `make demo-boxer` | Runs the Boxer RAG, KAG, evaluation, and summary demo. |
| `make demo-dogs` | Runs the Napoleon's dogs evaluation demo and summary. |
| `make clean-cache` | Removes Python `__pycache__` folders and `.pyc` files. |

## Project Structure

- `src/ingestion/`: extract, clean, and chunk the source text.
- `src/embeddings/`: create local sentence-transformer embeddings.
- `src/db/`: check the database and load embedded chunks into PostgreSQL.
- `src/retrieval/`: retrieve chunks from PostgreSQL using pgvector.
- `src/generation/`: generate DB-backed RAG and KAG answers.
- `src/graph/`: extract and review graph-style fact candidates.
- `src/evaluation/`: run and summarize lightweight RAG/KAG evaluation.
- `sql/`: PostgreSQL and pgvector schema.
- `data/evaluation/`: saved evaluation questions.

## Evaluation

Evaluation is intentionally lightweight and heuristic. It is meant to help compare RAG and KAG outputs during development, not to be a perfect judge of answer quality.

The evaluation runner uses saved questions from `data/evaluation/animal_farm_eval_questions.json`. It records RAG answers, KAG answers, evidence chunk summaries, extracted graph facts, expected keyword hits, citation checks, evidence counts, graph fact counts, and a simple score.

Saved evaluation outputs go under `data/evaluation/results/`, which is ignored by Git.

## Source And Data Safety

This project is for private educational learning. Do not redistribute the full book text in documentation, examples, generated outputs, or public commits.

Keep these local files untracked unless there is a deliberate reason to add them:

- `.env`
- `data/raw/animal_farm.pdf`
- generated files in `data/processed/`
- saved evaluation results in `data/evaluation/results/`
- presentation PDFs

The `.env` file should never be committed.

## Current Limitations And Next Steps

- Graph-style facts are stored as candidates, but they are not yet fully promoted into persistent reviewed `graph_nodes` and `graph_edges`.
- Evaluation is simple and heuristic.
- There is no UI or API layer yet.
- A useful next step would be testing the same DB-backed flow on more business-like documents.
