"""Embedding models.

Default: HuggingFace sentence-transformers (all-MiniLM-L6-v2), normalised so
that inner product == cosine similarity.

`HashingEmbeddings` is a small deterministic keyword embedder with no
downloads. It powers the automated tests and lets you try the app offline
(set EMBEDDING_BACKEND=hashing); it is lexical, not semantic.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import numpy as np
from langchain_core.embeddings import Embeddings

import config


class EmbeddingError(Exception):
    """Raised with a friendly message when embeddings cannot be created."""


_STOPWORDS = frozenset(
    "a an the of to is are was were be been in on at for and or but if as by with from "
    "what which who whom how do does did can could should would will it its this that these those "
    "i you he she we they my your our their me us them there here about into than then so".split()
)


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"\w+", text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words (+bigrams) hashing embedder, L2-normalised."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for feat in feats:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


_CACHE: dict[tuple[str, str], Embeddings] = {}


def get_embeddings(backend: Optional[str] = None, model_name: Optional[str] = None) -> Embeddings:
    """Return a (cached) embeddings object for the chosen backend."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    model_name = model_name or config.EMBEDDING_MODEL
    key = (backend, model_name)
    if key in _CACHE:
        return _CACHE[key]

    if backend == "hashing":
        emb: Embeddings = HashingEmbeddings()
    elif backend == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise EmbeddingError(
                "The embedding libraries are not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            emb = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:  # noqa: BLE001 - download / import / disk errors
            raise EmbeddingError(
                f"Could not load the embedding model '{model_name}'. The first run needs an internet "
                f"connection to download it (about 90 MB). Details: {exc}"
            ) from exc
    else:
        raise EmbeddingError(f"Unknown EMBEDDING_BACKEND '{backend}'. Use 'huggingface' or 'hashing'.")

    _CACHE[key] = emb
    return emb
