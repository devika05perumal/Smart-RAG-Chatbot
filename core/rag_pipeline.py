"""The RAG pipeline: ingestion, retrieval, prompt construction and answer generation.

Ingestion : validate -> load -> clean -> chunk -> embed -> FAISS
Answering : (condense follow-up) -> retrieve -> build prompt -> stream LLM answer
"""

from __future__ import annotations

import os
import tempfile
import threading
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
from core.document_loader import DocumentLoadError, load_document, validate_upload
from core.embeddings import EmbeddingError, get_embeddings
from core.llm import LLMError, build_llm, invoke_text, stream_text
from core.retriever import RetrievedChunk, Retriever
from core.text_splitter import split_documents
from core.vector_store import VectorStoreError, VectorStoreManager
from utils.helpers import classify_small_talk, get_extension, sanitize_filename, validate_question
from utils.prompts import (
    CONDENSE_PROMPT, GREETING_REPLY, HELP_REPLY, NO_ANSWER_MESSAGE,
    NO_DOCUMENTS_MESSAGE, SYSTEM_PROMPT, THANKS_REPLY,
)

log = logging.getLogger(__name__)

# Prevent overlapping manifest writes within this Python process.
_manifest_lock = threading.RLock()

INGEST_STAGES = ["Validating", "Extracting text", "Splitting into chunks", "Generating embeddings"]


class RAGError(Exception):
    """Friendly pipeline-level error (invalid question, missing index...)."""


@dataclass
class RAGSettings:
    model: str = config.DEFAULT_MODEL
    temperature: float = config.DEFAULT_TEMPERATURE
    top_k: int = config.DEFAULT_TOP_K
    score_threshold: float = config.DEFAULT_SCORE_THRESHOLD
    history_turns: int = config.DEFAULT_HISTORY_TURNS
    max_tokens: int = config.MAX_OUTPUT_TOKENS
    api_key: str = ""


@dataclass
class IngestResult:
    filename: str
    success: bool
    chunks: int = 0
    characters: int = 0
    error: Optional[str] = None


@dataclass
class PreparedQuery:
    question: str
    standalone_question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    messages: list = field(default_factory=list)
    direct_answer: Optional[str] = None
    reason: str = "ok"  # ok | no_documents | no_context | small_talk


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    prepared: PreparedQuery


class RAGPipeline:
    def __init__(
        self,
        embeddings_factory: Optional[Callable] = None,
        upload_dir: Path | str | None = None,
        vector_dir: Path | str | None = None,
        llm_factory: Optional[Callable] = None,
    ) -> None:
        self.upload_dir = Path(upload_dir or config.UPLOAD_DIR)
        self.vector_dir = Path(vector_dir or config.VECTOR_DB_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_factory = embeddings_factory or get_embeddings
        self._embeddings = None
        self.vectors = VectorStoreManager(self._get_embeddings, self.vector_dir)
        self.retriever = Retriever(self.vectors)
        self.llm_factory = llm_factory or build_llm
        self.manifest_path = self.vector_dir / "manifest.json"
        self.manifest: dict[str, dict] = self._load_manifest()

    # -------------------------------------------------------- embeddings
    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._embeddings_factory()
        return self._embeddings

    @property
    def embeddings_loaded(self) -> bool:
        return self._embeddings is not None

    # ---------------------------------------------------------- manifest
    def _load_manifest(self) -> dict[str, dict]:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_manifest(self) -> None:
        """Atomically save the manifest using a unique temporary file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with _manifest_lock:
            data = json.dumps(self.manifest, indent=2)
            fd, temp_name = tempfile.mkstemp(
                prefix="manifest_",
                suffix=".tmp",
                dir=str(self.manifest_path.parent),
            )

            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    tmp_file.write(data)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())

                os.replace(temp_name, self.manifest_path)
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)

    def list_documents(self) -> list[dict]:
        return [{"name": name, **info} for name, info in sorted(self.manifest.items())]

    def total_chunks(self) -> int:
        return sum(int(info.get("chunks", 0)) for info in self.manifest.values())

    def needs_reindex(self) -> bool:
        """True when documents are listed but their vector index is missing."""
        return bool(self.manifest) and not self.vectors.has_index()

    # --------------------------------------------------------- ingestion
    def ingest_file(
        self,
        filename: str,
        data: bytes,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> IngestResult:
        """Validate, extract, chunk, embed and index one uploaded file. Never raises for bad files."""
        stage = on_stage or (lambda _s: None)
        name = sanitize_filename(filename)
        tmp = self.upload_dir / f".incoming_{uuid.uuid4().hex}{get_extension(name)}"
        try:
            stage(INGEST_STAGES[0])
            validate_upload(name, data)
            tmp.write_bytes(data)

            stage(INGEST_STAGES[1])
            docs = load_document(tmp, display_name=name)

            stage(INGEST_STAGES[2])
            chunks = split_documents(docs, chunk_size, chunk_overlap)
            if not chunks:
                raise DocumentLoadError(f"No readable text was found in '{name}'.")

            stage(INGEST_STAGES[3])
            old_ids = self.vectors.ids_for_source(name)   # re-uploading replaces the old version
            self.vectors.add_documents(chunks)
            if old_ids:
                self.vectors.delete_ids(old_ids)

            tmp.replace(self.upload_dir / name)
            characters = sum(len(d.page_content) for d in docs)
            self.manifest[name] = {
                "chunks": len(chunks),
                "characters": characters,
                "size": len(data),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "added": datetime.now().isoformat(timespec="seconds"),
            }
            self._save_manifest()
            return IngestResult(name, True, chunks=len(chunks), characters=characters)
        except (DocumentLoadError, EmbeddingError, VectorStoreError, ValueError) as exc:
            return IngestResult(name, False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - never let one bad file crash the UI
            log.exception("Unexpected ingestion failure for %s", name)
            return IngestResult(name, False, error=f"Unexpected error while processing '{name}': {exc}")
        finally:
            tmp.unlink(missing_ok=True)

    def remove_document(self, name: str) -> bool:
        if name not in self.manifest:
            return False
        self.vectors.remove_source(name)
        (self.upload_dir / name).unlink(missing_ok=True)
        self.manifest.pop(name, None)
        if not self.manifest:
            self.vectors.clear()
        self._save_manifest()
        return True

    def reindex_all(
        self,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[IngestResult]:
        """Rebuild the whole index from the files stored in the uploads folder."""
        files = sorted(
            p for p in self.upload_dir.iterdir()
            if p.is_file() and not p.name.startswith(".") and get_extension(p.name) in config.SUPPORTED_EXTENSIONS
        )
        self.vectors.clear()
        self.manifest = {}
        results: list[IngestResult] = []
        for i, path in enumerate(files):
            if on_progress:
                on_progress(i, len(files), path.name)
            results.append(self.ingest_file(path.name, path.read_bytes(), chunk_size, chunk_overlap))
        self._save_manifest()
        if on_progress:
            on_progress(len(files), len(files), "")
        return results

    # --------------------------------------------------------- answering
    def _make_llm(self, settings: RAGSettings, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        return self.llm_factory(
            model=settings.model,
            temperature=settings.temperature if temperature is None else temperature,
            api_key=settings.api_key,
            max_tokens=max_tokens or settings.max_tokens,
        )

    def _condense(self, question: str, history: list[dict], settings: RAGSettings) -> str:
        """Turn a follow-up ("what about its price?") into a standalone search query."""
        if not history or len(question.split()) > 30:
            return question
        recent = history[-4:]
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}" for m in recent
        )
        prompt = CONDENSE_PROMPT.format(history=transcript, question=question)
        try:
            rewritten = invoke_text(self._make_llm(settings, temperature=0.0, max_tokens=128),
                                    [HumanMessage(content=prompt)])
        except LLMError as exc:
            if exc.kind != "other":
                raise          # missing/invalid key, rate limit, offline: fail fast with a clear message
            return question
        rewritten = rewritten.strip().strip('"').strip()
        return rewritten if 3 <= len(rewritten) <= 500 else question

    def _build_messages(self, question: str, history: list[dict], chunks: list[RetrievedChunk],
                        settings: RAGSettings) -> list:
        context = "\n\n".join(f"[Source {i}: {c.label}]\n{c.content}" for i, c in enumerate(chunks, start=1))
        messages: list = [SystemMessage(content=SYSTEM_PROMPT.format(context=context, no_answer=NO_ANSWER_MESSAGE))]
        for m in history[-(settings.history_turns * 2):]:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            messages.append(cls(content=m["content"]))
        messages.append(HumanMessage(content=question))
        return messages

    def prepare(self, question: str, history: Optional[list[dict]], settings: RAGSettings) -> PreparedQuery:
        """Everything up to (not including) the final LLM answer."""
        history = history or []
        ok, message = validate_question(question)
        if not ok:
            raise RAGError(message)
        question = question.strip()

        small = classify_small_talk(question)
        if small:
            reply = {"greeting": GREETING_REPLY, "thanks": THANKS_REPLY, "help": HELP_REPLY}[small]
            return PreparedQuery(question, question, direct_answer=reply, reason="small_talk")
        if not self.manifest:
            return PreparedQuery(question, question, direct_answer=NO_DOCUMENTS_MESSAGE, reason="no_documents")
        if self.needs_reindex():
            raise RAGError("The vector index is missing. Click 'Re-index all' in the sidebar to rebuild it.")

        standalone = self._condense(question, history, settings)
        chunks = self.retriever.retrieve(standalone, settings.top_k, settings.score_threshold)
        if not chunks:
            return PreparedQuery(question, standalone, direct_answer=NO_ANSWER_MESSAGE, reason="no_context")
        return PreparedQuery(
            question, standalone, chunks=chunks,
            messages=self._build_messages(question, history, chunks, settings),
        )

    def stream(self, prepared: PreparedQuery, settings: RAGSettings,
               temperature: Optional[float] = None) -> Iterator[str]:
        """Yield the answer text. Pre-written replies are yielded as-is without calling the LLM."""
        if prepared.direct_answer is not None:
            yield prepared.direct_answer
            return
        llm = self._make_llm(settings, temperature=temperature)
        yield from stream_text(llm, prepared.messages)

    def answer(self, question: str, history: Optional[list[dict]], settings: RAGSettings) -> RAGResult:
        """Non-streaming convenience wrapper (used by the tests and scripts)."""
        prepared = self.prepare(question, history, settings)
        if prepared.direct_answer is not None:
            text = prepared.direct_answer
        else:
            text = invoke_text(self._make_llm(settings), prepared.messages)
        return RAGResult(text, [c.to_dict() for c in prepared.chunks], prepared)
