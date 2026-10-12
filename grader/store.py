"""Chroma vector store with strict per-submission filtering.

Every student chunk carries course, assignment, student and version metadata,
and every student query filters on all four plus role="student". Reference
answers and instructions use other roles and can never match a student query.
"""
from __future__ import annotations

import re

import chromadb

from .models import ROLE_REFERENCE, ROLE_STUDENT, Chunk, Scope


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:60]


class ChunkStore:
    def __init__(self, embedder, path: str | None = None):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        # one collection per embedding model, because vector sizes differ
        self.col = self.client.get_or_create_collection(
            f"chunks_{_safe(embedder.name)}", metadata={"hnsw:space": "cosine"})

    # ---- writing -------------------------------------------------------
    def replace_submission(self, scope: Scope, chunks: list[Chunk]) -> int:
        """Store one submission version, replacing any earlier ingest of the same version."""
        self.col.delete(where=scope.as_filter())
        meta = {"role": ROLE_STUDENT, "course": scope.course, "assignment": scope.assignment,
                "student": scope.student, "version": scope.version}
        return self._add(chunks, meta, prefix=f"{scope.course}|{scope.assignment}|{scope.student}|{scope.version}")

    def replace_reference(self, course: str, assignment: str, chunks: list[Chunk]) -> int:
        where = {"$and": [{"role": ROLE_REFERENCE}, {"course": course}, {"assignment": assignment}]}
        self.col.delete(where=where)
        meta = {"role": ROLE_REFERENCE, "course": course, "assignment": assignment,
                "student": "", "version": ""}
        return self._add(chunks, meta, prefix=f"ref|{course}|{assignment}")

    def _add(self, chunks: list[Chunk], meta: dict, prefix: str) -> int:
        if not chunks:
            return 0
        ids = [f"{prefix}|{c.chunk_id}" for c in chunks]
        metas = [{**meta, "file": c.file, "page": -1 if c.page is None else c.page,
                  "line_start": c.line_start, "line_end": c.line_end,
                  "chunk_id": c.chunk_id, "ocr": c.ocr} for c in chunks]
        docs = [c.text for c in chunks]
        for i in range(0, len(docs), 64):
            self.col.add(ids=ids[i:i + 64], documents=docs[i:i + 64], metadatas=metas[i:i + 64],
                         embeddings=self.embedder.embed(docs[i:i + 64]))
        return len(chunks)

    # ---- reading -------------------------------------------------------
    def search(self, scope: Scope, query: str, k: int = 6) -> list[Chunk]:
        total = self.count(scope)
        if total == 0:
            return []
        res = self.col.query(query_embeddings=self.embedder.embed([query]),
                             n_results=min(k, total), where=scope.as_filter())
        return [self._to_chunk(d, m) for d, m in zip(res["documents"][0], res["metadatas"][0])]

    def all_chunks(self, scope: Scope) -> list[Chunk]:
        """Every chunk of one submission, in file/page/line order (for broad search)."""
        res = self.col.get(where=scope.as_filter())
        chunks = [self._to_chunk(d, m) for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(chunks, key=lambda c: (c.file, c.page or 0, c.line_start))

    def count(self, scope: Scope) -> int:
        return len(self.col.get(where=scope.as_filter(), include=[])["ids"])

    def reference_chunks(self, course: str, assignment: str, query: str, k: int = 3) -> list[Chunk]:
        where = {"$and": [{"role": ROLE_REFERENCE}, {"course": course}, {"assignment": assignment}]}
        n = len(self.col.get(where=where, include=[])["ids"])
        if n == 0:
            return []
        res = self.col.query(query_embeddings=self.embedder.embed([query]), n_results=min(k, n), where=where)
        return [self._to_chunk(d, m) for d, m in zip(res["documents"][0], res["metadatas"][0])]

    @staticmethod
    def _to_chunk(doc: str, m: dict) -> Chunk:
        return Chunk(chunk_id=m["chunk_id"], text=doc, file=m["file"],
                     page=None if m["page"] == -1 else m["page"],
                     line_start=m["line_start"], line_end=m["line_end"], char_start=0, char_end=len(doc),
                     ocr=bool(m.get("ocr", False)))
