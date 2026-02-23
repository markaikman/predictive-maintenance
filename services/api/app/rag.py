from __future__ import annotations

import os
import numpy as np
from fastapi import APIRouter, Query
from sqlalchemy import text

from app.db import get_engine

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder = None

router = APIRouter(prefix="/rag", tags=["rag"])


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def pg_vector_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


@router.get("/search")
def rag_search(q: str = Query(..., min_length=2), k: int = Query(8, ge=1, le=25)):
    emb = get_embedder().encode([q], normalize_embeddings=True)
    emb = np.asarray(emb[0], dtype=np.float32)

    sql = text(
        """
        SELECT
          c.id,
          d.title,
          d.source,
          d.uri,
          c.content,
          1 - (c.embedding <=> :q_emb) AS similarity
        FROM rag_chunks c
        JOIN rag_documents d ON d.id = c.document_id
        ORDER BY c.embedding <=> :q_emb
        LIMIT :k
        """
    )

    engine = get_engine()
    with engine.begin() as conn:
        rows = (
            conn.execute(sql, {"q_emb": pg_vector_literal(emb), "k": k})
            .mappings()
            .all()
        )

    return {"query": q, "k": k, "results": rows}
