# Animal Farm RAG/KAG Learning Project

## Project Goal

This project is a beginner-friendly RAG/KAG system built around Animal Farm using a publicly accessible Internet Archive PDF as the source document.

The main goal is not to build the most advanced system. The goal is to understand and explain each design decision clearly, including how text is cleaned, chunked, embedded, retrieved, and used to generate answers.

This project should be simple enough that I can explain each step to my tech lead.

## Source Document Approach

The project will use a publicly accessible Internet Archive PDF of Animal Farm as the starting source document.

This project is for private educational learning only. The full book text should not be redistributed in the README, examples, generated sample outputs, or committed documentation.

If the user provides the PDF file, store it locally in:

- data/raw/

The pipeline should then extract text from the PDF into:

- data/processed/

The extracted text is an intermediate local artifact for learning how ingestion works. It should be handled carefully and should not be copied into documentation or shared as generated output.

## Why Animal Farm?

Animal Farm is shorter and more structured than the previous Frankenstein project. This makes it easier to manually inspect chunks, test retrieval quality, and build a simple knowledge graph around characters, themes, and major events.

Because Animal Farm may still be copyrighted in some jurisdictions, the project should use only a source PDF that the user provides or identifies as publicly accessible for their private educational use. The project should avoid redistributing the full text.

## Main Learning Questions

1. How do we load and clean text?
2. How do we split text into chunks?
3. Why did we choose a certain chunk size?
4. How do embeddings turn chunks into vectors?
5. Why should chunks and questions use the same embedding model?
6. How do we retrieve the most relevant chunks?
7. Why did we choose a certain number of retrieved chunks?
8. How does a simple knowledge graph help organize relationships?
9. How can RAG and KAG work together?

## Initial RAG Design Choices

### Ingestion

Start with a simple PDF ingestion flow:

1. Store the source PDF in data/raw/.
2. Extract plain text from the PDF into data/processed/.
3. Clean obvious extraction artifacts such as repeated headers, page numbers, extra whitespace, and broken line wrapping.
4. Save the cleaned text as a local processed file.
5. Chunk the cleaned text for retrieval.

Reasoning:

Using the PDF makes the project closer to a real document ingestion pipeline while still staying beginner-friendly. Keeping each step separate makes it easier to explain what changed between the original PDF, extracted text, cleaned text, and final chunks.

### Chunking

Start with paragraph-based chunking.

Initial settings:

- chunk_size: 150 words
- overlap: 30 words

Reasoning:

Animal Farm is a shorter source than Frankenstein, so smaller chunks should make retrieval more focused. A 150-word chunk is large enough to contain a useful idea, but small enough to avoid pulling in too much unrelated context.

The 30-word overlap helps preserve context when an idea is split across two chunks.

### Retrieval

Start with:

- top_k: 3

Reasoning:

Because the dataset is small, retrieving too many chunks may add unnecessary noise. Three chunks should provide enough supporting context while keeping the answer focused.

### Embeddings

Use one embedding model for both stored chunks and user questions.

Reasoning:

Chunks and questions need to be embedded into the same vector space so similarity comparisons are meaningful.

## Initial KAG Design Choices

Start with a simple manually created knowledge graph.

Do not begin with automatic graph extraction.

Reasoning:

Manual graph creation is better for learning because each node and edge can be inspected and explained. Automatic extraction can be added later after the basic graph structure is understood.

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

## Build Order

1. Create project structure
2. Add the publicly accessible Internet Archive PDF to data/raw/ when provided
3. Extract text from the PDF into data/processed/
4. Clean the extracted text
5. Chunk the cleaned text
6. Inspect chunks manually
7. Generate embeddings
8. Store chunks and embeddings
9. Retrieve similar chunks for a question
10. Generate an answer using retrieved context
11. Add a simple knowledge graph
12. Combine RAG chunks with KAG relationships

## Important Project Constraint

Keep the project beginner-friendly.

Avoid unnecessary frameworks, complex infrastructure, or advanced optimization until the basic RAG pipeline is fully understood.

Do not include long excerpts or the full book text in documentation, examples, generated outputs, or public commits. Use short summaries, metadata, or small private local snippets only when needed for debugging.
