CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS prediction_logs (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  engine_id TEXT NOT NULL,
  model_version TEXT NOT NULL,
  prediction DOUBLE PRECISION NOT NULL,
  features JSONB NOT NULL
);

-- Later (RAG):
-- CREATE TABLE documents (
--   id BIGSERIAL PRIMARY KEY,
--   source TEXT,
--   content TEXT,
--   embedding VECTOR(384)  -- if using a 384-dim embed model
-- );
-- CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops);
