# Animal Farm RAG/KAG Learning Project

This is a beginner-friendly project for learning how a simple RAG/KAG pipeline works using Animal Farm as the source document.

The goal is explainability. Each step should be easy to inspect and explain: PDF ingestion, text extraction, cleaning, chunking, embeddings, retrieval, answer generation, and a simple knowledge graph.

## Source Document

This project will use a publicly accessible Internet Archive PDF of Animal Farm as the source document.

Important constraints:

- This project is for private educational learning only.
- Do not redistribute the full book text in this README, examples, generated outputs, or public commits.
- If the PDF is provided, store it locally in `data/raw/`.
- Extracted text should be written to `data/processed/`.
- Documentation should describe the process, not reproduce the book text.

## Folder Structure

- `data/raw/`: stores the original source PDF provided by the user.
- `data/processed/`: stores local intermediate files such as extracted text, cleaned text, chunks, and embeddings.
- `src/ingestion/`: code for loading the PDF, extracting text, cleaning text, and chunking text.
- `src/embeddings/`: code for turning chunks into vectors.
- `src/retrieval/`: code for finding relevant chunks for a question.
- `src/generation/`: code for generating answers from retrieved context.
- `src/graph/`: code for the simple manually created knowledge graph.
- `tests/`: tests for beginner-friendly pipeline pieces, starting with chunking.

## Ingestion Plan

The first version of ingestion should stay simple:

1. Place the source PDF in `data/raw/`.
2. Extract plain text from the PDF into `data/processed/`.
3. Clean common PDF extraction artifacts, such as page numbers, repeated headers, extra whitespace, and broken line wrapping.
4. Save the cleaned text as a separate local processed file.
5. Split the cleaned text into chunks.
6. Manually inspect a few chunks to confirm they are useful for retrieval.

No PDF extraction code has been written yet. The documentation is being updated first so the source approach is clear before implementation.

## Initial Design Choices

Start with paragraph-based chunking:

- `chunk_size`: 150 words
- `overlap`: 30 words

Start retrieval with:

- `top_k`: 3

Use the same embedding model for both stored chunks and user questions so they are compared in the same vector space.

## Knowledge Graph Plan

Start with a simple manually created knowledge graph.

Example entity types:

- Character
- Place
- Theme
- Event
- Object

Example relationships:

- Napoleon OPPOSES Snowball
- Squealer USES Propaganda
- Old Major INSPIRES Rebellion
- Boxer REPRESENTS Loyalty
- Pigs CONTROL Animal Farm
- Commandments CHANGE_OVER_TIME
