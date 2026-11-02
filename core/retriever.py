"""Semantic retrieval with similarity scores and source tracking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from core.vector_store import VectorStoreManager
from utils.helpers import source_label


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: Optional[int]
    rows: Optional[str]
    chunk_index: int
    score: float

    @property
    def label(self) -> str:
        return source_label({"source": self.source, "page": self.page, "rows": self.rows})

    def to_dict(self) -> dict:
        data = asdict(self)
        data["score"] = round(float(self.score), 4)
        return data


class Retriever:
    def __init__(self, vectors: VectorStoreManager) -> None:
        self.vectors = vectors

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.0) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks whose similarity is >= `score_threshold`, best first."""
        if not query or not query.strip() or not self.vectors.has_index():
            return []
        fetch_k = max(top_k * 2, top_k + 4)  # over-fetch so de-duplication can't starve the result
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for doc, score in self.vectors.search(query, k=fetch_k):
            if score < score_threshold:
                continue
            key = " ".join(doc.page_content.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            meta = doc.metadata
            results.append(RetrievedChunk(
                content=doc.page_content,
                source=str(meta.get("source", "unknown")),
                page=meta.get("page"),
                rows=meta.get("rows"),
                chunk_index=int(meta.get("chunk_index", 0)),
                score=score,
            ))
            if len(results) >= top_k:
                break
        return results
