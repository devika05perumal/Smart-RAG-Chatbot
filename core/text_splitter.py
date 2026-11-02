"""Chunking: split documents into overlapping, retrieval-sized pieces."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

MIN_CHUNK_CHARS = 5
_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks, keeping source/page metadata.

    Each chunk gets a per-source `chunk_index`. Raises ValueError for invalid sizes.
    """
    if chunk_size < 100:
        raise ValueError("Chunk size must be at least 100 characters.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than the chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        add_start_index=True,
    )
    counters: dict[str, int] = defaultdict(int)
    chunks: list[Document] = []
    for chunk in splitter.split_documents(list(documents)):
        text = chunk.page_content.strip()
        if len(text) < MIN_CHUNK_CHARS:
            continue
        chunk.page_content = text
        source = str(chunk.metadata.get("source", "unknown"))
        chunk.metadata["chunk_index"] = counters[source]
        counters[source] += 1
        chunks.append(chunk)
    return chunks
