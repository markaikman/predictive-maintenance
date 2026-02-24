# app/rag_answer.py (or inside app/main.py)
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from openai import OpenAI

from .rag_retrieval import retrieve_chunks

router = APIRouter(prefix="/rag", tags=["rag"])


def _openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _dedupe_citations(cites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for c in cites:
        key = (c.get("source"), c.get("uri"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _format_context(
    results: list[dict[str, Any]], max_chars: int = 12000
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build a context string with stable citation ids: [S1], [S2], ...
    Returns (context_text, citations_metadata).
    """
    parts: list[str] = []
    cites: list[dict[str, Any]] = []
    total = 0

    # prevent model from contradicting itself w/ "active snapshot wins" rule
    has_active = any(r.get("uri") == "db://model_registry/active" for r in results)
    if has_active:
        results = [r for r in results if r.get("uri") != "db://model_registry/history"]

    for i, r in enumerate(results, start=1):
        tag = f"S{i}"
        snippet = (r.get("content") or "").strip()
        header = f"[{tag}] {r.get('title')} ({r.get('source')} | {r.get('uri')})"

        block = f"{header}\n{snippet}\n"
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

        cites.append(
            {
                "tag": tag,
                "chunk_id": r.get("id"),
                "title": r.get("title"),
                "source": r.get("source"),
                "uri": r.get("uri"),
                "similarity": r.get("similarity"),
                "kw_rank": r.get("kw_rank"),
                "score": r.get("score"),
            }
        )
        cites = _dedupe_citations(cites)
        cites = cites[:5]  # cap at 5 for

    return "\n---\n".join(parts), cites


@router.get("/answer")
def rag_answer(
    q: str = Query(..., min_length=2),
    k: int = Query(5, ge=1, le=12),
):
    """
    Returns: answer + citations.
    Assumes you already have a function that executes the rag/search query and returns rows as dicts.
    """
    # 1) retrieve (reuse your existing search function)
    # implement by calling the same SQL used in /rag/search
    results = retrieve_chunks(q=q, k=k)  # must return list[dict]

    if not results:
        return {
            "answer": "I couldn’t find anything relevant in the knowledge base.",
            "citations": [],
        }

    context_text, citations = _format_context(results)

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = _openai_client()

    # 2) generate answer (Responses API)
    # Responses API supports simple input strings OR structured items; we’ll do structured for clarity. :contentReference[oaicite:3]{index=3}
    instructions = (
        "You are a helpful assistant for this repository.\n"
        "Answer the user's question using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Cite sources inline like [S1], [S2] using the tags provided.\n"
        "If a source has uri starting with 'db://model_registry/active', treat it as authoritative.\n"
        "Keep it concise and practical."
    )

    user_input = f"QUESTION:\n{q}\n\n" f"CONTEXT:\n{context_text}"

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
        # You can tune temperature; the API supports it. :contentReference[oaicite:4]{index=4}
        temperature=0.2,
    )

    # Python SDK exposes output_text as a convenience in Responses. :contentReference[oaicite:5]{index=5}
    answer = getattr(resp, "output_text", None) or str(resp)

    return {"answer": answer, "citations": citations}
