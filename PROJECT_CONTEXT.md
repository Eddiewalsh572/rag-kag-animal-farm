# Animal Farm RAG/KAG Project Context

## Project Goal

This project is a beginner-to-intermediate RAG/KAG learning project built around Animal Farm.

The goal is to understand and explain an end-to-end DB-backed pipeline: source document ingestion, text cleaning, chunking, local embeddings, PostgreSQL/pgvector storage, retrieval, RAG answer generation, KAG-style graph fact extraction, and lightweight evaluation.

The project should stay clean enough to explain to a tech lead without hiding the important steps behind a large framework.

## Current Supported Path

The main supported path is PostgreSQL + pgvector.

The older local JSON retrieval/generation/manual-graph prototype path has been removed. Local JSON files are still used as rebuild artifacts where they make sense, especially for chunks and embeddings before loading data into the database.

Current high-level flow:

1. `data/raw/animal_farm.pdf`
2. `src/ingestion/extract_pdf_text.py`
3. `src/ingestion/clean_text.py`
4. `src/ingestion/chunk_text.py`
5. `src/embeddings/embed_chunks.py`
6. `src/db/store_chunks_embeddings.py`
7. `src/retrieval/retrieve_chunks_db.py`
8. `src/generation/generate_answer_db.py`
9. `src/generation/generate_kag_answer_db.py`
10. `src/evaluation/run_rag_kag_eval.py`
11. `src/evaluation/summarize_eval_results.py`

## Current Design Choices

### Ingestion

The project keeps ingestion steps separate so each transformation is easy to inspect:

- extract text from the PDF
- clean repeated PDF artifacts and spacing issues
- chunk the cleaned text
- generate embeddings for each chunk
- load embedded chunks into PostgreSQL

The generated files in `data/processed/` are local rebuild artifacts. They should generally stay untracked, but they are not obsolete.

### Embeddings

The project uses local Hugging Face sentence-transformer embeddings with `sentence-transformers/all-MiniLM-L6-v2`.

The embeddings are 384-dimensional, matching the pgvector column defined in the database schema.

### Retrieval

Current retrieval uses `src/retrieval/retrieve_chunks_db.py`.

It retrieves from PostgreSQL using pgvector similarity and then applies simple keyword reranking. This keeps the retrieval logic readable while still using the database-backed vector store.

### RAG

Current RAG uses `src/generation/generate_answer_db.py`.

It retrieves relevant chunks from PostgreSQL, expands evidence with nearby chunks, builds a grounded prompt, and asks the generator to answer using only the provided evidence.

### KAG

Current KAG uses:

- `src/generation/generate_kag_answer_db.py`
- `src/graph/extract_question_graph_facts_db.py`
- `src/graph/review_graph_candidates_db.py`
- the `graph_fact_candidates` database table

The KAG flow extracts graph-style facts from retrieved evidence for the current question. These facts are stored as candidates first.

The `graph_nodes` and `graph_edges` tables may currently be empty. That is expected for this stage of the project. A future improvement would promote reviewed candidates into persistent graph nodes and edges.

## Evaluation

Evaluation uses:

- `data/evaluation/animal_farm_eval_questions.json`
- `src/evaluation/run_rag_kag_eval.py`
- `src/evaluation/summarize_eval_results.py`

The evaluator is lightweight and heuristic. It compares RAG and KAG answers using expected keywords, citation checks, evidence chunk counts, graph fact counts, and a simple score. It is useful for development review, but it is not a perfect answer-quality evaluator.

Evaluation result files are written to `data/evaluation/results/`, which should stay ignored.

## Useful Commands

```bash
docker compose up -d
make doctor
make check
make rag QUESTION="What happens to Boxer?"
make kag QUESTION="What happens to Boxer?"
make eval ID=boxer_fate MODE=both
make summary
```

Other useful shortcuts:

```bash
make demo-boxer
make demo-dogs
make clean-cache
```

## Data And Git Safety

Do not commit `.env`.

These are local artifacts and should generally stay untracked unless intentionally added:

- `data/raw/animal_farm.pdf`
- `data/processed/animal_farm_extracted.txt`
- `data/processed/animal_farm_cleaned.txt`
- `data/processed/animal_farm_chunks.json`
- `data/processed/animal_farm_embeddings.json`
- `data/evaluation/results/`
- presentation PDFs

## Current Limitations

- Graph fact candidates are not yet promoted into persistent reviewed graph nodes and graph edges.
- Evaluation is heuristic and should be treated as a development aid.
- There is no UI or API layer yet.
- The next useful expansion would be trying the same pipeline on more business-like documents.
