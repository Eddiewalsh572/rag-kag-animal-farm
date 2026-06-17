CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  author TEXT,
  source_type TEXT,
  source_path TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(title, author)
);

CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  word_count INTEGER,
  embedding VECTOR(384),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS graph_nodes (
  id BIGSERIAL PRIMARY KEY,
  label TEXT NOT NULL,
  type TEXT,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(label, type)
);

CREATE TABLE IF NOT EXISTS graph_edges (
  id BIGSERIAL PRIMARY KEY,
  source_node_id BIGINT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  target_node_id BIGINT NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  description TEXT,
  evidence_chunk_id BIGINT REFERENCES chunks(id) ON DELETE SET NULL,
  confidence TEXT,
  review_status TEXT DEFAULT 'approved',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS graph_fact_candidates (
  id BIGSERIAL PRIMARY KEY,
  source_entity TEXT NOT NULL,
  relation TEXT NOT NULL,
  target_entity TEXT NOT NULL,
  description TEXT,
  evidence_chunk_ids INTEGER[],
  confidence TEXT DEFAULT 'llm_extracted',
  review_status TEXT DEFAULT 'unreviewed',
  review_notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_document_chunk_index_idx
  ON chunks(document_id, chunk_index);

CREATE INDEX IF NOT EXISTS graph_nodes_label_idx
  ON graph_nodes(label);

CREATE INDEX IF NOT EXISTS graph_edges_source_node_id_idx
  ON graph_edges(source_node_id);

CREATE INDEX IF NOT EXISTS graph_edges_target_node_id_idx
  ON graph_edges(target_node_id);

CREATE INDEX IF NOT EXISTS graph_fact_candidates_review_status_idx
  ON graph_fact_candidates(review_status);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
  ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
