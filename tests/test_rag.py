"""Test-suite for the Smart RAG Assistant.

Run from the project root:   pytest -q
Everything runs offline: embeddings use the deterministic HashingEmbeddings and
the LLM is replaced by a small fake, so no API key or model download is needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core.document_loader import DocumentLoadError, load_document, validate_upload  # noqa: E402
from core.embeddings import HashingEmbeddings  # noqa: E402
from core.llm import LLMError, build_llm, translate_exception  # noqa: E402
from core.rag_pipeline import RAGError, RAGPipeline, RAGSettings  # noqa: E402
from core.text_splitter import split_documents  # noqa: E402
from core.vector_store import VectorStoreManager  # noqa: E402
from database.chat_history import ChatHistoryDB  # noqa: E402
from utils.helpers import (  # noqa: E402
    classify_small_talk, clean_text, conversation_to_markdown, sanitize_filename, validate_question,
)
from utils.prompts import NO_ANSWER_MESSAGE, NO_DOCUMENTS_MESSAGE  # noqa: E402

SAMPLES = ROOT / "data" / "sample_docs"
SETTINGS = RAGSettings(api_key="gsk_test_key", score_threshold=0.1, top_k=4)


class FakeLLM:
    """Stands in for ChatGroq: records prompts, returns canned text."""

    def __init__(self, reply="Employees receive 24 days [Source 1].", condensed="standalone question"):
        self.reply, self.condensed = reply, condensed
        self.stream_calls, self.invoke_calls = [], []

    def stream(self, messages):
        self.stream_calls.append(messages)
        for word in self.reply.split(" "):
            yield SimpleNamespace(content=word + " ")

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        is_condense = "standalone question" in messages[0].content.lower() and len(messages) == 1
        return SimpleNamespace(content=self.condensed if is_condense else self.reply)


@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def pipeline(tmp_path, fake_llm):
    return RAGPipeline(
        embeddings_factory=HashingEmbeddings,
        upload_dir=tmp_path / "uploads",
        vector_dir=tmp_path / "vector_db",
        llm_factory=lambda **kwargs: fake_llm,
    )


@pytest.fixture()
def loaded(pipeline):
    for path in sorted(SAMPLES.iterdir()):
        result = pipeline.ingest_file(path.name, path.read_bytes(), 600, 100)
        assert result.success, result.error
    return pipeline


# ------------------------------------------------------------ 1. Upload
class TestDocumentUpload:
    def test_valid_files_pass_validation(self):
        for path in SAMPLES.iterdir():
            validate_upload(path.name, path.read_bytes())

    def test_unsupported_extension_rejected(self):
        with pytest.raises(DocumentLoadError, match="not a supported file type"):
            validate_upload("malware.exe", b"MZ....")

    def test_empty_file_rejected(self):
        with pytest.raises(DocumentLoadError, match="empty"):
            validate_upload("notes.txt", b"")

    def test_oversized_file_rejected(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_FILE_SIZE_MB", 1)
        with pytest.raises(DocumentLoadError, match="exceeds"):
            validate_upload("big.txt", b"a" * (1024 * 1024 + 1))

    def test_fake_pdf_and_docx_signatures_rejected(self):
        with pytest.raises(DocumentLoadError, match="valid PDF"):
            validate_upload("a.pdf", b"just text pretending to be a pdf")
        with pytest.raises(DocumentLoadError, match="valid DOCX"):
            validate_upload("a.docx", b"just text pretending to be a docx")

    def test_binary_masquerading_as_text_rejected(self):
        with pytest.raises(DocumentLoadError, match="binary"):
            validate_upload("a.txt", b"abc\x00\x01\x02def")

    def test_filename_is_sanitised(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("my report (final)?.pdf") == "my report _final__.pdf"

    def test_ingest_reports_failure_instead_of_raising(self, pipeline):
        result = pipeline.ingest_file("bad.pdf", b"%PDF-1.4 this is not really a pdf", 600, 100)
        assert not result.success and "corrupted" in result.error
        assert pipeline.list_documents() == []

    def test_reuploading_replaces_old_chunks(self, pipeline):
        data = (SAMPLES / "rag_basics.txt").read_bytes()
        first = pipeline.ingest_file("rag_basics.txt", data, 600, 100)
        second = pipeline.ingest_file("rag_basics.txt", data, 600, 100)
        assert first.success and second.success
        assert pipeline.vectors.count() == second.chunks == pipeline.manifest["rag_basics.txt"]["chunks"]

    def test_remove_document(self, loaded):
        assert loaded.remove_document("sales_data.csv")
        assert "sales_data.csv" not in loaded.manifest
        assert not (loaded.upload_dir / "sales_data.csv").exists()
        assert all(d.metadata["source"] != "sales_data.csv" for d in loaded.vectors.store.docstore._dict.values())

    def test_remove_last_document_clears_index(self, pipeline):
        pipeline.ingest_file("a.txt", b"Alpha beta gamma delta epsilon.", 600, 100)
        pipeline.remove_document("a.txt")
        assert not pipeline.vectors.has_index() and pipeline.manifest == {}

    def test_reindex_rebuilds_from_disk(self, loaded):
        before = loaded.total_chunks()
        (loaded.vectors.persist_dir / "index.faiss").unlink()
        loaded.vectors._store = None  # as after an app restart: nothing in memory, files gone
        assert loaded.needs_reindex()
        results = loaded.reindex_all(600, 100)
        assert all(r.success for r in results)
        assert loaded.total_chunks() == before and not loaded.needs_reindex()


# ------------------------------------------------------ 2. Text extraction
class TestTextExtraction:
    def test_pdf_pages_and_content(self):
        docs = load_document(SAMPLES / "company_handbook.pdf")
        assert [d.metadata["page"] for d in docs] == [1, 2]
        assert "24 days of paid annual leave" in docs[0].page_content

    def test_docx_includes_tables_in_order(self):
        text = load_document(SAMPLES / "product_faq.docx")[0].page_content
        assert "Plus | 5 euros | 100 GB" in text
        assert text.index("Pricing") < text.index("Plus | 5 euros") < text.index("Features")

    def test_txt_extraction(self):
        assert "Retrieval-Augmented Generation" in load_document(SAMPLES / "rag_basics.txt")[0].page_content

    def test_csv_rows_become_readable_text(self):
        docs = load_document(SAMPLES / "sales_data.csv")
        assert docs[0].metadata["rows"] == "1-20" and docs[1].metadata["rows"] == "21-24"
        assert "region: North; product: Nimbus Notes Plus" in docs[0].page_content

    def test_latin1_text_is_decoded(self, tmp_path):
        f = tmp_path / "cafe.txt"
        f.write_bytes("Caf\xe9 men\xfc".encode("latin-1"))
        assert "Café menü" in load_document(f)[0].page_content

    def test_corrupted_pdf(self, tmp_path):
        f = tmp_path / "broken.pdf"
        f.write_bytes(b"%PDF-1.7\n1 0 obj\n<< garbage")
        with pytest.raises(DocumentLoadError, match="corrupted"):
            load_document(f)

    def test_corrupted_docx(self, tmp_path):
        f = tmp_path / "broken.docx"
        f.write_bytes(b"PK\x03\x04 not a real zip")
        with pytest.raises(DocumentLoadError, match="corrupted|valid DOCX|could not be read"):
            load_document(f)

    @pytest.mark.parametrize("name,content", [("empty.txt", b"   \n\n  "), ("empty.csv", b"a,b,c\n")])
    def test_documents_without_text_are_rejected(self, tmp_path, name, content):
        f = tmp_path / name
        f.write_bytes(content)
        with pytest.raises(DocumentLoadError):
            load_document(f)

    def test_clean_text(self):
        raw = "Hel-\nlo   world\r\n\r\n\r\n\r\nNext\x00 line  "
        assert clean_text(raw) == "Hello world\n\nNext line"


# ---------------------------------------------------------- 3. Chunking
class TestChunking:
    def test_chunks_respect_size_and_carry_metadata(self):
        docs = load_document(SAMPLES / "rag_basics.txt")
        chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 300 for c in chunks)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert all(c.metadata["source"] == "rag_basics.txt" for c in chunks)

    def test_overlap_repeats_text_between_neighbours(self):
        text = " ".join(f"word{i}" for i in range(400))
        from langchain_core.documents import Document
        chunks = split_documents([Document(page_content=text, metadata={"source": "x.txt"})], 200, 60)
        assert set(chunks[0].page_content.split()[-3:]) & set(chunks[1].page_content.split()[:12])

    def test_smaller_chunk_size_gives_more_chunks(self):
        docs = load_document(SAMPLES / "rag_basics.txt")
        assert len(split_documents(docs, 200, 20)) > len(split_documents(docs, 1000, 20))

    @pytest.mark.parametrize("size,overlap", [(50, 0), (500, 500), (500, 900), (500, -1)])
    def test_invalid_settings_raise(self, size, overlap):
        with pytest.raises(ValueError):
            split_documents(load_document(SAMPLES / "rag_basics.txt"), size, overlap)


# ------------------------------------------------------ 4. Embeddings
class TestEmbeddings:
    def test_dimension_and_normalisation(self):
        emb = HashingEmbeddings(dim=256)
        vecs = emb.embed_documents(["annual leave policy", "sales by region"])
        assert len(vecs) == 2 and all(len(v) == 256 for v in vecs)
        assert all(abs(sum(x * x for x in v) - 1.0) < 1e-5 for v in vecs)

    def test_deterministic(self):
        emb = HashingEmbeddings()
        assert emb.embed_query("remote work policy") == emb.embed_query("remote work policy")

    def test_related_text_scores_higher_than_unrelated(self):
        emb = HashingEmbeddings()
        q, near, far = (emb.embed_query(t) for t in (
            "how many days of annual leave", "employees get 24 days of annual leave", "refund policy for plus plan"))
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
        assert dot(q, near) > dot(q, far)

    def test_empty_text_does_not_crash(self):
        assert len(HashingEmbeddings().embed_query("")) == HashingEmbeddings().dim


# ------------------------------------------------------- 5. Vector search
class TestVectorSearch:
    def test_most_relevant_chunk_is_first(self, loaded):
        hits = loaded.retriever.retrieve("How many days of annual leave do employees get?", top_k=3, score_threshold=0.05)
        assert hits and hits[0].source == "company_handbook.pdf"          # right document first
        assert any("24 days" in h.content for h in hits)                  # answer-bearing chunk retrieved
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)  # best first
        assert not any(h.source in {"sales_data.csv", "product_faq.docx"} for h in hits[:1])

    def test_scores_are_cosine_similarities(self, loaded):
        hits = loaded.retriever.retrieve("annual leave", top_k=2)
        assert all(-1.0001 <= h.score <= 1.0001 for h in hits)

    def test_top_k_limits_results(self, loaded):
        assert len(loaded.retriever.retrieve("employee leave refund sales", top_k=2)) <= 2

    def test_threshold_filters_everything_for_unrelated_query(self, loaded):
        assert loaded.retriever.retrieve("zebra volcano trombone", top_k=4, score_threshold=0.1) == []

    def test_index_persists_to_disk(self, loaded):
        fresh = VectorStoreManager(HashingEmbeddings, loaded.vector_dir)
        hits = fresh.search("remote work days per week", k=1)
        assert hits and hits[0][0].metadata["source"] == "company_handbook.pdf"

    def test_source_page_metadata_is_kept(self, loaded):
        hit = loaded.retriever.retrieve("expense reimbursement receipts", top_k=1, score_threshold=0.05)[0]
        assert hit.source == "company_handbook.pdf" and hit.page == 2 and "p. 2" in hit.label

    def test_empty_store_returns_nothing(self, pipeline):
        assert pipeline.retriever.retrieve("anything") == []


# ------------------------------------------------------ 6. Question answering
class TestQuestionAnswering:
    def test_answer_uses_retrieved_context_and_returns_sources(self, loaded, fake_llm):
        result = loaded.answer("How many days of annual leave do employees get?", [], SETTINGS)
        assert result.answer == "Employees receive 24 days [Source 1]."
        assert result.sources[0]["source"] == "company_handbook.pdf"
        system_prompt = fake_llm.invoke_calls[-1][0].content
        assert "24 days of paid annual leave" in system_prompt and "[Source 1:" in system_prompt
        assert NO_ANSWER_MESSAGE in system_prompt  # instructs the model what to say when context is missing

    def test_streaming_yields_pieces(self, loaded):
        prepared = loaded.prepare("What is the home office allowance for remote work?", [], SETTINGS)
        pieces = list(loaded.stream(prepared, SETTINGS))
        assert len(pieces) > 1 and "".join(pieces).startswith("Employees receive")

    def test_follow_up_is_rewritten_with_history(self, loaded, fake_llm):
        fake_llm.condensed = "How many days of annual leave do employees get?"
        history = [{"role": "user", "content": "Tell me about annual leave"},
                   {"role": "assistant", "content": "Employees get paid annual leave."}]
        prepared = loaded.prepare("How many days is it?", history, SETTINGS)
        assert prepared.standalone_question == fake_llm.condensed
        assert prepared.chunks and prepared.chunks[0].source == "company_handbook.pdf"
        roles = [type(m).__name__ for m in prepared.messages]
        assert roles == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]

    def test_history_is_limited(self, loaded):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(40)]
        settings = RAGSettings(api_key="k", score_threshold=0.05, history_turns=2)
        prepared = loaded.prepare("annual leave days", history, settings)
        assert len(prepared.messages) == 1 + 4 + 1

    def test_system_prompt_braces_in_documents_do_not_break_formatting(self, pipeline, fake_llm):
        pipeline.ingest_file("code.txt", b"Use the template {name} and {{literal}} braces in the config file.", 600, 100)
        result = pipeline.answer("Which template braces are used in the config file?", [], SETTINGS)
        assert "{name}" in fake_llm.invoke_calls[-1][0].content and result.sources


# ---------------------------------------------- 7. Missing context handling
class TestMissingContext:
    def test_unrelated_question_gets_the_exact_fallback_without_calling_the_llm(self, loaded, fake_llm):
        result = loaded.answer("What is the capital of Australia, volcano trombone?", [], SETTINGS)
        assert result.answer == NO_ANSWER_MESSAGE
        assert result.sources == [] and result.prepared.reason == "no_context"
        assert not fake_llm.invoke_calls and not fake_llm.stream_calls

    def test_no_documents_uploaded(self, pipeline, fake_llm):
        result = pipeline.answer("What is the leave policy?", [], SETTINGS)
        assert result.answer == NO_DOCUMENTS_MESSAGE and result.prepared.reason == "no_documents"
        assert not fake_llm.invoke_calls

    @pytest.mark.parametrize("text,kind", [("Hello!", "greeting"), ("hi there", "greeting"),
                                           ("thanks", "thanks"), ("what can you do?", "help")])
    def test_small_talk_skips_retrieval(self, loaded, fake_llm, text, kind):
        assert classify_small_talk(text) == kind
        result = loaded.answer(text, [], SETTINGS)
        assert result.prepared.reason == "small_talk" and not fake_llm.invoke_calls

    def test_real_question_is_not_small_talk(self):
        assert classify_small_talk("hello, how many leave days do I get?") is None

    @pytest.mark.parametrize("bad", ["", "   ", "?!?!", "x" * 5000])
    def test_invalid_questions_are_rejected(self, loaded, bad):
        assert validate_question(bad)[0] is False
        with pytest.raises(RAGError):
            loaded.prepare(bad, [], SETTINGS)

    def test_lost_index_asks_for_reindex(self, loaded):
        (loaded.vectors.persist_dir / "index.pkl").unlink()
        loaded.vectors._store = None
        with pytest.raises(RAGError, match="Re-index"):
            loaded.prepare("annual leave", [], SETTINGS)


# --------------------------------------------------------- LLM errors
class TestLLMErrors:
    def test_missing_api_key(self):
        with pytest.raises(LLMError) as exc:
            build_llm("llama-3.3-70b-versatile", 0.2, "")
        assert exc.value.kind == "missing_key" and "GROQ_API_KEY" in str(exc.value)

    @pytest.mark.parametrize("status,kind", [(401, "auth"), (404, "model"), (429, "rate_limit"), (413, "context"), (503, "server")])
    def test_http_errors_become_friendly_messages(self, status, kind):
        err = translate_exception(type("APIStatusError", (Exception,), {"status_code": status})("boom"))
        assert err.kind == kind

    def test_connection_failures(self):
        assert translate_exception(type("APIConnectionError", (Exception,), {})("down")).kind == "connection"

    def test_pipeline_fails_fast_on_missing_key_for_follow_ups(self, loaded):
        pipe = RAGPipeline(embeddings_factory=HashingEmbeddings, upload_dir=loaded.upload_dir, vector_dir=loaded.vector_dir)
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        with pytest.raises(LLMError) as exc:
            pipe.prepare("annual leave days?", history, RAGSettings(api_key="", score_threshold=0.05))
        assert exc.value.kind == "missing_key"


# ------------------------------------------------------ chat history DB
class TestChatHistory:
    def test_messages_sources_and_titles(self, tmp_path):
        db = ChatHistoryDB(tmp_path / "chat.db")
        cid = db.create_conversation()
        db.add_message(cid, "user", "Hi?")
        db.add_message(cid, "assistant", "Hello", [{"source": "a.pdf", "page": 2, "score": 0.9, "content": "x"}])
        msgs = db.get_messages(cid)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["sources"][0]["source"] == "a.pdf" and msgs[0]["created_at"]
        db.rename_conversation(cid, "Leave policy")
        assert db.get_title(cid) == "Leave policy"
        assert "a.pdf, p. 2" in conversation_to_markdown("T", msgs)

    def test_regenerate_removes_only_last_assistant_message(self, tmp_path):
        db = ChatHistoryDB(tmp_path / "chat.db")
        cid = db.create_conversation()
        db.add_message(cid, "user", "q")
        db.add_message(cid, "assistant", "a")
        assert db.delete_last_assistant_message(cid) is True
        assert db.delete_last_assistant_message(cid) is False
        assert [m["role"] for m in db.get_messages(cid)] == ["user"]

    def test_clear_and_delete(self, tmp_path):
        db = ChatHistoryDB(tmp_path / "chat.db")
        a, b = db.create_conversation("A"), db.create_conversation("B")
        db.add_message(a, "user", "q")
        db.clear_conversation(a)
        assert db.get_messages(a) == []
        db.delete_conversation(b)
        assert [c["id"] for c in db.list_conversations()] == [a]


# --------------------------------------------------------- app smoke test
def test_streamlit_app_end_to_end(tmp_path, monkeypatch):
    """Boots the real UI, loads sample docs, asks a question with no API key."""
    pytest.importorskip("streamlit.testing.v1")
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "VECTOR_DB_DIR", tmp_path / "vector_db")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "chat.db")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    monkeypatch.setattr(config, "DEFAULT_SCORE_THRESHOLD", 0.1)
    monkeypatch.setenv("GROQ_API_KEY", "")
    st.cache_resource.clear()

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Start by adding a document" in m.value for m in at.markdown)

    next(b for b in at.sidebar.button if b.key == "load_samples").click().run()
    assert not at.exception, at.exception
    assert any("Uploaded files (4)" in e.label for e in at.sidebar.expander)

    at.chat_input[0].set_value("How many days of annual leave do employees get?").run()
    assert not at.exception, at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error), [e.value for e in at.error]

    next(b for b in at.sidebar.button if b.key == "new_chat").click().run()
    at.chat_input[0].set_value("what is the weather on volcano trombone").run()
    assert not at.exception, at.exception
    assert any(NO_ANSWER_MESSAGE in m.value for m in at.chat_message[-1].markdown)
    st.cache_resource.clear()
