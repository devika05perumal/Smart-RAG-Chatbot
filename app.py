"""Smart RAG Assistant - Streamlit front end.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

import config
from core.document_loader import DocumentLoadError
from core.embeddings import EmbeddingError, get_embeddings
from core.llm import LLMError, check_connection
from core.rag_pipeline import INGEST_STAGES, RAGError, RAGPipeline, RAGSettings
from core.vector_store import VectorStoreError
from database.chat_history import ChatHistoryDB, ChatHistoryError
from utils.helpers import (
    conversation_to_markdown, dedupe_keep_order, escape, format_file_size, format_timestamp,
    get_extension, is_fallback_answer, source_label, truncate, validate_question,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("smart_rag")

USER_AVATAR, AI_AVATAR = "👤", "🧠"

st.set_page_config(page_title=config.APP_NAME, page_icon="🧠", layout="wide", initial_sidebar_state="expanded")


# ------------------------------------------------------------------ setup
@st.cache_resource(show_spinner=False)
def get_pipeline(backend: str, model_name: str) -> RAGPipeline:
    return RAGPipeline(
        embeddings_factory=lambda: get_embeddings(backend, model_name),
        upload_dir=config.UPLOAD_DIR,
        vector_dir=config.VECTOR_DB_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_db(path: str) -> ChatHistoryDB:
    return ChatHistoryDB(path)


def load_css() -> None:
    if config.STYLE_PATH.exists():
        st.markdown(f"<style>{config.STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state(db: ChatHistoryDB) -> None:
    ss = st.session_state
    defaults = {
        "uploader_key": 0, "ingest_report": [], "api_status": None, "regen": False,
        "model": config.DEFAULT_MODEL, "temperature": config.DEFAULT_TEMPERATURE,
        "chunk_size": config.DEFAULT_CHUNK_SIZE, "chunk_overlap": config.DEFAULT_CHUNK_OVERLAP,
        "top_k": config.DEFAULT_TOP_K, "score_threshold": config.DEFAULT_SCORE_THRESHOLD,
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)
    if "conv_id" not in ss:
        existing = db.list_conversations(limit=1)
        ss.conv_id = existing[0]["id"] if existing else db.create_conversation()


def current_settings() -> RAGSettings:
    ss = st.session_state
    return RAGSettings(
        model=ss.model, temperature=float(ss.temperature), top_k=int(ss.top_k),
        score_threshold=float(ss.score_threshold), api_key=config.get_api_key(),
    )


def refresh_api_status(force: bool = False) -> None:
    if force or st.session_state.api_status is None:
        with st.spinner("Checking Groq connection..."):
            st.session_state.api_status = check_connection(config.get_api_key())


# -------------------------------------------------------------- ingestion
def process_files(pipeline: RAGPipeline, files: list[tuple[str, bytes]]) -> None:
    """Ingest files with a live progress bar, then store a per-file report."""
    total = len(files)
    bar = st.sidebar.progress(0.0, text="Preparing...")
    results = []
    for i, (name, data) in enumerate(files):
        def on_stage(stage: str, i: int = i, name: str = name) -> None:
            step = INGEST_STAGES.index(stage) if stage in INGEST_STAGES else 0
            extra = " (first run downloads the model)" if stage == INGEST_STAGES[-1] and not pipeline.embeddings_loaded else ""
            bar.progress(min((i + (step + 0.5) / len(INGEST_STAGES)) / total, 0.99),
                         text=f"{truncate(name, 28)}: {stage.lower()}{extra}...")
        results.append(pipeline.ingest_file(
            name, data, st.session_state.chunk_size, st.session_state.chunk_overlap, on_stage=on_stage))
    bar.progress(1.0, text="Done")
    st.session_state.ingest_report = [
        {"name": r.filename, "ok": r.success, "chunks": r.chunks, "error": r.error} for r in results]
    st.session_state.uploader_key += 1
    st.rerun()


def load_sample_docs(pipeline: RAGPipeline) -> None:
    files = [(p.name, p.read_bytes()) for p in sorted(config.SAMPLE_DOCS_DIR.glob("*"))
             if p.is_file() and get_extension(p.name) in config.SUPPORTED_EXTENSIONS]
    if not files:
        st.warning("No sample documents were found in data/sample_docs.")
        return
    process_files(pipeline, files)


# ---------------------------------------------------------------- sidebar
def status_pill(state: str, text: str) -> str:
    css = {"connected": "ok", "invalid_key": "bad", "error": "bad", "offline": "warn", "missing_key": "warn"}.get(state, "warn")
    return f'<span class="status-pill {css}"><i></i>{escape(text)}</span>'


def render_sidebar(pipeline: RAGPipeline, db: ChatHistoryDB) -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><b>{escape(config.APP_NAME)}</b>'
            f'<span>Answers grounded in your documents</span></div>', unsafe_allow_html=True)

        docs = pipeline.list_documents()

        # ---- 1. Upload documents
        with st.expander("Upload documents", expanded=not docs):
            files = st.file_uploader(
                "Upload documents", type=[e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS],
                accept_multiple_files=True, key=f"uploader_{ss.uploader_key}", label_visibility="collapsed",
                help=f"PDF, DOCX, TXT or CSV. Up to {config.MAX_FILE_SIZE_MB} MB each.")
            if st.button("Process documents", type="primary", use_container_width=True,
                         disabled=not files, key="process_btn"):
                process_files(pipeline, [(f.name, f.getvalue()) for f in files])
            if not docs and config.SAMPLE_DOCS_DIR.exists():
                if st.button("Load sample documents", use_container_width=True, key="load_samples"):
                    load_sample_docs(pipeline)
            for item in ss.ingest_report:
                if item["ok"]:
                    st.success(f"{item['name']}: {item['chunks']} chunks indexed", icon="✅")
                else:
                    st.error(item["error"], icon="🚫")

        # ---- 2. Uploaded files
        with st.expander(f"Uploaded files ({len(docs)})", expanded=True):
            if not docs:
                st.caption("Nothing indexed yet.")
            for d in docs:
                c1, c2 = st.columns([5, 1])
                c1.markdown(
                    f'<div class="file-row"><div class="file-ext">{escape(get_extension(d["name"]).lstrip(".").upper())}</div>'
                    f'<div class="file-meta"><b title="{escape(d["name"])}">{escape(d["name"])}</b>'
                    f'<small>{d.get("chunks", 0)} chunks · {format_file_size(d.get("size", 0))}</small></div></div>',
                    unsafe_allow_html=True)
                if c2.button("✕", key=f"rm_{d['name']}", help=f"Remove {d['name']}"):
                    try:
                        pipeline.remove_document(d["name"])
                        ss.ingest_report = []
                    except (VectorStoreError, EmbeddingError) as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
            if pipeline.needs_reindex():
                st.warning("The search index is missing. Re-index to rebuild it from your uploaded files.")
            if docs or pipeline.needs_reindex():
                if st.button("Re-index all", use_container_width=True, key="reindex_btn",
                             help="Rebuild the index with the current chunk settings."):
                    bar = st.progress(0.0, text="Re-indexing...")
                    results = pipeline.reindex_all(
                        ss.chunk_size, ss.chunk_overlap,
                        on_progress=lambda i, n, name: bar.progress(min(i / max(n, 1), 1.0), text=f"Re-indexing {truncate(name, 24)}..."))
                    ss.ingest_report = [{"name": r.filename, "ok": r.success, "chunks": r.chunks, "error": r.error}
                                        for r in results if not r.success]
                    st.rerun()

        # ---- 3. Chat history
        with st.expander("Chat history", expanded=True):
            if st.button("New chat", use_container_width=True, key="new_chat"):
                if db.get_messages(ss.conv_id):
                    ss.conv_id = db.create_conversation()
                st.rerun()
            for conv in db.list_conversations(limit=10):
                active = conv["id"] == ss.conv_id
                c1, c2 = st.columns([6, 1])
                if c1.button(truncate(conv["title"], 26), key=f"conv_{conv['id']}", use_container_width=True,
                             type="primary" if active else "secondary", help=format_timestamp(conv["updated_at"])):
                    ss.conv_id = conv["id"]
                    st.rerun()
                if c2.button("✕", key=f"del_{conv['id']}", help="Delete this chat"):
                    db.delete_conversation(conv["id"])
                    if active:
                        remaining = db.list_conversations(limit=1)
                        ss.conv_id = remaining[0]["id"] if remaining else db.create_conversation()
                    st.rerun()

        # ---- 4-6. LLM, temperature, RAG, top-k
        status = ss.api_status
        models = list(config.configured_models())
        if status and status.models:
            models += [m for m in status.models if m not in models]
        if ss.model not in models:
            models.insert(0, ss.model)
        with st.expander("LLM model settings"):
            st.selectbox("Groq model", models, key="model",
                         help="Models are listed from your Groq account once the connection check succeeds.")
            st.slider("Temperature", 0.0, 1.0, step=0.05, key="temperature",
                      help="Low = factual and consistent. High = more varied wording. 0.0-0.3 is best for document Q&A.")
        with st.expander("RAG settings"):
            st.slider("Chunk size (characters)", 200, 3000, step=50, key="chunk_size",
                      help="Applies to newly processed files. Use 'Re-index all' to apply to existing ones.")
            ss.chunk_overlap = min(ss.chunk_overlap, ss.chunk_size // 2)
            st.slider("Chunk overlap", 0, ss.chunk_size // 2, step=25, key="chunk_overlap")
            st.slider("Top-K retrieval", 1, 10, key="top_k", help="How many passages are sent to the model per question.")
            st.slider("Minimum similarity", 0.0, 0.9, step=0.05, key="score_threshold",
                      help="Passages scoring below this are ignored. Raise it for stricter answers.")

        # ---- Model info
        with st.expander("Model information"):
            st.markdown(
                '<table class="mini-table">'
                f'<tr><td>LLM</td><td>{escape(ss.model)}</td></tr>'
                '<tr><td>Provider</td><td>Groq</td></tr>'
                f'<tr><td>Embeddings</td><td>{escape(config.EMBEDDING_MODEL if config.EMBEDDING_BACKEND == "huggingface" else "keyword hashing (offline)")}</td></tr>'
                '<tr><td>Vector DB</td><td>FAISS (cosine)</td></tr>'
                f'<tr><td>Indexed</td><td>{len(docs)} files · {pipeline.total_chunks()} chunks</td></tr>'
                '</table>', unsafe_allow_html=True)

        # ---- Conversation actions
        messages = db.get_messages(ss.conv_id)
        a1, a2 = st.columns(2)
        if a1.button("Clear chat", use_container_width=True, disabled=not messages, key="clear_chat"):
            db.clear_conversation(ss.conv_id)
            st.rerun()
        a2.download_button(
            "Download", data=conversation_to_markdown(db.get_title(ss.conv_id) or "Conversation", messages),
            file_name=f"conversation_{ss.conv_id}.md", mime="text/markdown",
            use_container_width=True, disabled=not messages, key="download_chat")

        # ---- API status
        st.markdown("**API connection**")
        status = ss.api_status
        st.markdown(status_pill(status.state, {"connected": "Connected to Groq", "missing_key": "No API key",
                                               "invalid_key": "Invalid API key", "offline": "Can't reach Groq"}
                                .get(status.state, "Connection problem")), unsafe_allow_html=True)
        if not status.ok:
            st.caption(status.message)
        elif status.models and ss.model not in status.models:
            st.caption("The selected model isn't in your Groq model list. Pick another one above.")
        if st.button("Test connection", use_container_width=True, key="test_conn"):
            refresh_api_status(force=True)
            st.rerun()


# ------------------------------------------------------------ main panels
def render_header(pipeline: RAGPipeline) -> None:
    ss = st.session_state
    st.markdown(
        f'<div class="app-header"><h1>{escape(config.APP_NAME)}</h1><p>{escape(config.APP_TAGLINE)}</p></div>'
        f'<div class="chip-row"><span class="chip">Documents <b>{len(pipeline.manifest)}</b></span>'
        f'<span class="chip">Chunks <b>{pipeline.total_chunks()}</b></span>'
        f'<span class="chip">Model <b>{escape(ss.model)}</b></span>'
        f'<span class="chip">Top-K <b>{ss.top_k}</b></span></div>', unsafe_allow_html=True)


def render_empty_state(pipeline: RAGPipeline) -> None:
    if not pipeline.manifest:
        st.markdown(
            '<div class="empty"><h3>Start by adding a document</h3>'
            '<p>Upload PDF, DOCX, TXT or CSV files in the sidebar. Questions are answered only from what '
            'those files contain, and each answer shows the passages it came from.</p>'
            '<div class="steps">'
            '<div class="step"><b>Upload</b><span>Add one or more files, then choose Process documents.</span></div>'
            '<div class="step"><b>Ask</b><span>Type a question. Follow-ups like “and what about refunds?” work too.</span></div>'
            '<div class="step"><b>Verify</b><span>Open the retrieved snippets under any answer to check the source.</span></div>'
            '</div></div>', unsafe_allow_html=True)
        return
    st.markdown('<div class="empty"><h3>Ask about your documents</h3>'
                '<p>Try one of these, or type your own question below.</p></div>', unsafe_allow_html=True)
    examples = ["Give me a short summary of the uploaded documents.",
                "What are the main topics covered?",
                "List the key dates, numbers or names mentioned."]
    cols = st.columns(len(examples))
    for col, text in zip(cols, examples):
        if col.button(text, key=f"ex_{text}", use_container_width=True):
            st.session_state.queued_prompt = text
            st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    labels = dedupe_keep_order(source_label(s) for s in sources)
    chips = "".join(f'<span class="src-chip">{escape(label)}</span>' for label in labels)
    st.markdown(f'<div class="src-row"><span class="src-title">Sources</span>{chips}</div>', unsafe_allow_html=True)
    with st.expander(f"Retrieved snippets ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            pct = int(max(0.0, min(1.0, float(s.get("score", 0)))) * 100)
            text = escape(truncate(s.get("content", ""), 700)).replace("\n", "<br>")
            st.markdown(
                f'<div class="snip-head">[{i}] {escape(source_label(s))}'
                f'<span class="score-bar"><i style="width:{pct}%"></i></span><span class="score-num">{pct}% match</span></div>'
                f'<div class="snippet">{text}</div>', unsafe_allow_html=True)


def render_messages(messages: list[dict], show_actions: bool) -> None:
    last = len(messages) - 1
    for i, msg in enumerate(messages):
        is_user = msg["role"] == "user"
        with st.chat_message(msg["role"], avatar=USER_AVATAR if is_user else AI_AVATAR):
            st.markdown(msg["content"])
            if not is_user:
                render_sources(msg.get("sources", []))
                c1, c2, _ = st.columns([1.3, 1.1, 5])
                with c2.popover("Copy"):
                    st.caption("Use the copy icon at the top right of the box.")
                    st.code(msg["content"], language=None, wrap_lines=True)
                if show_actions and i == last and c1.button("Regenerate", key=f"regen_{msg['id']}"):
                    st.session_state.regen = True
                    st.rerun()
    if show_actions and messages and messages[-1]["role"] == "user":
        if st.button("Retry answer", key="retry_btn", type="primary"):
            st.session_state.regen = True
            st.rerun()


def show_error(exc: Exception) -> None:
    st.error(str(exc), icon="🚫")
    if isinstance(exc, LLMError) and exc.kind in {"missing_key", "auth"}:
        st.caption("After editing `.env`, restart the app (Ctrl+C, then `streamlit run app.py`).")


def generate_reply(pipeline: RAGPipeline, db: ChatHistoryDB, question: str, history: list[dict], regenerate: bool) -> None:
    ss = st.session_state
    settings = current_settings()
    temperature = min(1.0, settings.temperature + 0.2) if regenerate else None  # vary wording on regenerate
    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            wait = "Searching your documents..." if pipeline.embeddings_loaded else \
                   "Loading the embedding model (the first run downloads ~90 MB)..."
            with st.spinner(wait):
                prepared = pipeline.prepare(question, history, settings)
            if prepared.direct_answer is not None:
                answer = prepared.direct_answer
                st.markdown(answer)
            else:
                answer = st.write_stream(pipeline.stream(prepared, settings, temperature=temperature))
        except (LLMError, RAGError, EmbeddingError, VectorStoreError, DocumentLoadError) as exc:
            show_error(exc)
            return
        except ChatHistoryError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error while answering")
            st.error(f"Something unexpected went wrong: {exc}", icon="🚫")
            return
        if not str(answer).strip():
            st.warning("The model returned an empty reply. Try again, or pick another model.")
            return
        sources = [] if is_fallback_answer(answer) else [c.to_dict() for c in prepared.chunks]
    db.add_message(ss.conv_id, "assistant", str(answer), sources)
    st.rerun()


# ------------------------------------------------------------------- main
def main() -> None:
    load_css()
    try:
        db = get_db(str(config.DB_PATH))
    except ChatHistoryError as exc:
        st.error(str(exc))
        st.stop()
    pipeline = get_pipeline(config.EMBEDDING_BACKEND, config.EMBEDDING_MODEL)
    init_state(db)
    refresh_api_status()
    ss = st.session_state

    prompt = st.chat_input("Ask a question about your documents...") or ss.pop("queued_prompt", None)
    regen = ss.pop("regen", False)
    if regen:
        db.delete_last_assistant_message(ss.conv_id)

    question, valid_prompt = None, False
    if prompt:
        ok, message = validate_question(prompt)
        if ok:
            db.add_message(ss.conv_id, "user", prompt.strip())
            if db.get_title(ss.conv_id) == ChatHistoryDB.DEFAULT_TITLE:
                db.rename_conversation(ss.conv_id, truncate(prompt, 48))
            question, valid_prompt = prompt.strip(), True
        else:
            st.toast(message, icon="⚠️")

    messages = db.get_messages(ss.conv_id)
    if regen and messages and messages[-1]["role"] == "user":
        question = messages[-1]["content"]
    history = messages[:-1] if question else messages

    render_sidebar(pipeline, db)
    render_header(pipeline)
    if not messages:
        render_empty_state(pipeline)
    render_messages(messages, show_actions=question is None)
    if question:
        generate_reply(pipeline, db, question, history, regenerate=regen and not valid_prompt)


main()
