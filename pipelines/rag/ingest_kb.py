from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re

import numpy as np
from sqlalchemy import create_engine, text

# Local embeddings (no API key needed)
from sentence_transformers import SentenceTransformer


EMBED_DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Doc:
    source: str
    title: str
    uri: str
    content: str
    metadata: dict


def chunk_text(s: str, *, max_chars: int = 1100, overlap: int = 200) -> list[str]:
    s = re.sub(r"\r\n", "\n", s).strip()
    if not s:
        return []
    chunks = []
    i = 0
    n = len(s)
    while i < n:
        j = min(i + max_chars, n)
        chunk = s[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_db_snapshot_docs(eng) -> list[Doc]:
    docs: list[Doc] = []
    with eng.begin() as conn:
        # Active dev/prod for your main model
        rows = (
            conn.execute(
                text(
                    """
                SELECT stage, run_id, notes, artifact_path, created_at
                FROM model_registry
                WHERE name = 'cmapss_fd001_rul' AND is_active = TRUE
                ORDER BY stage
                """
                )
            )
            .mappings()
            .all()
        )

        content = {
            "name": "cmapss_fd001_rul",
            "active": [dict(r) for r in rows],
        }
        docs.append(
            Doc(
                source="db",
                title="Model Registry Snapshot (active dev/prod)",
                uri="db://model_registry/active",
                content=json.dumps(content, indent=2, default=str),
                metadata={"type": "model_registry_snapshot"},
            )
        )

        # Recent promotions/history (last 20 rows)
        hist = (
            conn.execute(
                text(
                    """
                SELECT stage, run_id, is_active, notes, artifact_path, created_at
                FROM model_registry
                WHERE name = 'cmapss_fd001_rul'
                ORDER BY created_at DESC
                LIMIT 20
                """
                )
            )
            .mappings()
            .all()
        )

        docs.append(
            Doc(
                source="db",
                title="Model Registry History (last 20)",
                uri="db://model_registry/history",
                content=json.dumps([dict(r) for r in hist], indent=2, default=str),
                metadata={"type": "model_registry_history"},
            )
        )

    return docs


def collect_sources(repo_root: Path) -> list[Doc]:
    docs: list[Doc] = []

    # README
    readme = repo_root / "README.md"
    if readme.exists():
        docs.append(
            Doc(
                source="repo",
                title="README",
                uri=str(readme.relative_to(repo_root)),
                content=read_text_file(readme),
                metadata={"type": "readme"},
            )
        )

    # Eval reports
    eval_dir = repo_root / "artifacts" / "eval"
    if eval_dir.exists():
        for p in sorted(eval_dir.glob("*.json")):
            docs.append(
                Doc(
                    source="artifact",
                    title=f"Eval report: {p.name}",
                    uri=str(p.relative_to(repo_root)),
                    content=read_text_file(p),
                    metadata={"type": "eval_report"},
                )
            )

    # Baseline stats
    models_dir = repo_root / "artifacts" / "models"
    if models_dir.exists():
        for p in sorted(models_dir.glob("baseline_stats*.json")):
            docs.append(
                Doc(
                    source="artifact",
                    title=f"Baseline stats: {p.name}",
                    uri=str(p.relative_to(repo_root)),
                    content=read_text_file(p),
                    metadata={"type": "baseline_stats"},
                )
            )

    return docs


def pg_vector_literal(v: np.ndarray) -> str:
    # pgvector accepts: '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


def upsert_doc_and_chunks(
    eng,
    doc: Doc,
    chunks: list[str],
    embeddings: np.ndarray,
):
    # Dedupe by (source, uri)
    with eng.begin() as conn:
        existing = conn.execute(
            text(
                """
                SELECT id FROM rag_documents
                WHERE source = :source AND uri = :uri
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"source": doc.source, "uri": doc.uri},
        ).fetchone()

        if existing:
            doc_id = int(existing[0])
            # Replace: delete old chunks and update doc content
            conn.execute(
                text("DELETE FROM rag_chunks WHERE document_id = :doc_id"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text(
                    """
                    UPDATE rag_documents
                    SET title = :title,
                        content = :content,
                        metadata = :metadata
                    WHERE id = :doc_id
                    """
                ),
                {
                    "title": doc.title,
                    "content": doc.content,
                    "metadata": json.dumps(doc.metadata),
                    "doc_id": doc_id,
                },
            )
        else:
            row = conn.execute(
                text(
                    """
                    INSERT INTO rag_documents (source, title, uri, content, metadata)
                    VALUES (:source, :title, :uri, :content, :metadata)
                    RETURNING id
                    """
                ),
                {
                    "source": doc.source,
                    "title": doc.title,
                    "uri": doc.uri,
                    "content": doc.content,
                    "metadata": json.dumps(doc.metadata),
                },
            ).fetchone()
            doc_id = int(row[0])

        # Insert chunks
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                text(
                    """
                    INSERT INTO rag_chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES (:document_id, :chunk_index, :content, :embedding, :metadata)
                    """
                ),
                {
                    "document_id": doc_id,
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": pg_vector_literal(emb),
                    "metadata": json.dumps({"uri": doc.uri, "title": doc.title}),
                },
            )


def main():
    repo_root = Path(__file__).resolve().parents[2]

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL env var is required (point it at dsdb).")

    docs = collect_sources(repo_root)
    if not docs:
        print("No documents found to ingest.")
        return

    print(f"Embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    eng = create_engine(db_url, pool_pre_ping=True)

    db_docs = collect_db_snapshot_docs(eng)
    docs.extend(db_docs)

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc.content)
        if not chunks:
            continue

        # Embed in batches
        emb = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # cosine-friendly
        )
        emb = np.asarray(emb, dtype=np.float32)

        if emb.shape[1] != EMBED_DIM:
            raise ValueError(
                f"Embedding dim mismatch: got {emb.shape[1]}, expected {EMBED_DIM}"
            )

        upsert_doc_and_chunks(eng, doc, chunks, emb)
        total_chunks += len(chunks)
        print(f"Ingested: {doc.title}  chunks={len(chunks)}")

    # Helpful for IVFFLAT index performance after ingest
    with eng.begin() as conn:
        conn.execute(text("ANALYZE rag_chunks"))

    print(f"\nDone. Documents={len(docs)}  Total chunks={total_chunks}")


if __name__ == "__main__":
    main()
