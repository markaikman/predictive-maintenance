CREATE TABLE IF NOT EXISTS model_registry (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  name TEXT NOT NULL,
  stage TEXT NOT NULL CHECK (stage IN ('dev', 'staging', 'prod')),
  run_id TEXT NOT NULL,
  artifact_path TEXT NOT NULL, -- e.g. "model/model.pkl" or "model"
  notes TEXT,
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_active
ON model_registry (name, stage)
WHERE is_active = TRUE;
