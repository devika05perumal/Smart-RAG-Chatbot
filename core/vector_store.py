"""FAISS vector store with persistence, per-source deletion and cosine scores."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.embeddings import EmbeddingError


class VectorStoreError(Exception):
    """Raised with a friendly message when the vector database fails."""


class VectorStoreManager:
    """Thin wrapper around LangChain's FAISS store.

    Embeddings are normalised, and the index uses inner product, so the score
    returned for each hit is the cosine similarity (higher = more similar).
    """

    INDEX_NAME = "index"

    def __init__(self, embeddings_provider: Callable[[], Embeddings], persist_dir: Path | str) -> None:
        self._embeddings_provider = embeddings_provider
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store: Optional[FAISS] = None

    # ------------------------------------------------------------ state
    def has_index(self) -> bool:
        return self._store is not None or (
            (self.persist_dir / f"{self.INDEX_NAME}.faiss").exists()
            and (self.persist_dir / f"{self.INDEX_NAME}.pkl").exists()
        )

    @property
    def store(self) -> Optional[FAISS]:
        if self._store is None and self.has_index():
            self._store = self._load()
        return self._store

    def _load(self) -> FAISS:
        try:
            # The index is a pickle written by this app into its own data folder.
            # Never point persist_dir at files from an untrusted source.
            return FAISS.load_local(
                str(self.persist_dir),
                self._embeddings_provider(),
                index_name=self.INDEX_NAME,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "The saved vector index could not be loaded (it may be corrupted or built with a different "
                "embedding model). Use 'Re-index all' in the sidebar to rebuild it."
            ) from exc

    def save(self) -> None:
        if self._store is None:
            return
        try:
            self._store.save_local(str(self.persist_dir), index_name=self.INDEX_NAME)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not save the vector index to disk: {exc}") from exc

    # --------------------------------------------------------- mutation
    def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        try:
            if self.store is None:
                self._store = FAISS.from_documents(
                    chunks,
                    self._embeddings_provider(),
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                )
            else:
                self._store.add_documents(chunks)
        except (EmbeddingError, VectorStoreError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not add documents to the vector index: {exc}") from exc
        self.save()
        return len(chunks)

    def ids_for_source(self, source: str) -> list[str]:
        store = self.store
        if store is None:
            return []
        return [i for i, d in store.docstore._dict.items() if d.metadata.get("source") == source]  # noqa: SLF001

    def delete_ids(self, ids: list[str]) -> None:
        if not ids or self.store is None:
            return
        try:
            self._store.delete(ids)
            if len(self._store.docstore._dict) == 0:  # noqa: SLF001
                self.clear()
            else:
                self.save()
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Could not remove documents from the vector index: {exc}") from exc

    def remove_source(self, source: str) -> int:
        ids = self.ids_for_source(source)
        self.delete_ids(ids)
        return len(ids)

    def clear(self) -> None:
        self._store = None
        for suffix in ("faiss", "pkl"):
            (self.persist_dir / f"{self.INDEX_NAME}.{suffix}").unlink(missing_ok=True)
        if self.persist_dir.exists() and not any(self.persist_dir.iterdir()):
            shutil.rmtree(self.persist_dir, ignore_errors=True)
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ query
    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        store = self.store
        if store is None:
            return []
        try:
            hits = store.similarity_search_with_score(query, k=k)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"The vector search failed: {exc}") from exc
        return [(doc, float(score)) for doc, score in hits]

    def count(self) -> int:
        store = self.store
        return 0 if store is None else len(store.docstore._dict)  # noqa: SLF001
