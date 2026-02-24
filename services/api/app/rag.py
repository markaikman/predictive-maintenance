from __future__ import annotations

import os
import numpy as np
from fastapi import APIRouter, Query

from app.rag_retrieval import retrieve_chunks

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
    rows = retrieve_chunks(q=q, k=k)
    return {"query": q, "k": k, "results": rows}
