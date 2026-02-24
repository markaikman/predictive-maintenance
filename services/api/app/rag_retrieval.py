from __future__ import annotations

import os
import numpy as np
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from app.db import get_engine

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _pg_vector_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


SQL_HYBRID = text(
    """
    SELECT
        c.id,
        d.title,
        d.source,
        d.uri,
        c.content,
        (1 - (c.embedding <=> :q_emb)) AS similarity,
        ts_rank(c.content_tsv, websearch_to_tsquery('english', :q_txt)) AS kw_rank,
        CASE WHEN d.source = 'db' THEN 0.08 ELSE 0 END AS src_boost,
        (
          (1 - (c.embedding <=> :q_emb))
          + 0.15 * ts_rank(c.content_tsv, websearch_to_tsquery('english', :q_txt))
          + CASE WHEN d.source = 'db' THEN 0.08 ELSE 0 END
        ) AS score
    FROM rag_chunks c
    JOIN rag_documents d ON d.id = c.document_id
    ORDER BY score DESC
    LIMIT :k;
    """
)


def retrieve_chunks(q: str, k: int = 8) -> list[dict]:
    emb = _get_embedder().encode([q], normalize_embeddings=True)
    emb = np.asarray(emb[0], dtype=np.float32)

    engine = get_engine()
    with engine.begin() as conn:
        rows = (
            conn.execute(
                SQL_HYBRID,
                {"q_emb": _pg_vector_literal(emb), "q_txt": q, "k": k},
            )
            .mappings()
            .all()
        )

    return [dict(r) for r in rows]
