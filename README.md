# Smart RAG Assistant

A document Q&A chatbot built on Retrieval-Augmented Generation. Upload PDF, DOCX, TXT or CSV files, ask questions in plain language, and get answers grounded in your documents, with the source file, page and matching passage shown under every answer. If the answer isn't in your files, the assistant says so instead of guessing.

## Features

- **Documents:** multi-file upload, PDF / DOCX / TXT / CSV, per-file status and progress bar, remove a file, re-index everything, friendly errors for empty, corrupted, oversized or unsupported files.
- **RAG pipeline:** load, clean, chunk, embed, store in FAISS, retrieve by cosine similarity, build a prompt, stream the answer. Chunk size, overlap, Top-K and a minimum-similarity cut-off are all adjustable from the sidebar.
- **Conversation:** follow-up questions work ("and what about refunds?") because follow-ups are rewritten into standalone search queries using the chat history. Regenerate, copy, clear, new chat, delete chats, download a conversation as Markdown.
- **No hallucination by design:** if nothing relevant is retrieved, the app replies with the fixed sentence *"I could not find sufficient information in the uploaded documents to answer this question."* without calling the LLM at all. When context is found, the system prompt forbids outside knowledge.
- **Sources:** file names with page numbers (or CSV row ranges) as chips, plus an expandable list of the retrieved snippets with similarity scores.
- **Persistence:** chat history in SQLite; the vector index and file list survive restarts.
- **Safety:** API key only from `.env`, file extension / size / signature checks, uploads are parsed and never executed.

## Tech stack

| Layer | Choice |
|---|---|
| UI | Streamlit + custom CSS (`assets/style.css`) |
| Orchestration | LangChain components (`langchain-core`, `-text-splitters`, `-community`, `-groq`, `-huggingface`) |
| LLM | Groq API, Llama models (default `llama-3.3-70b-versatile`), streaming |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace, normalised |
| Vector store | FAISS (inner product on normalised vectors = cosine similarity) |
| Parsing | `pypdf`, `python-docx`, `pandas` |
| History | SQLite (`data/chat_history.db`) |
| Config | `python-dotenv` |

## Architecture

```
 Upload ──► validate ──► extract ──► clean ──► chunk ──► embed ──► FAISS index
                                                                      │
 Question ─► (rewrite follow-up with chat history) ─► embed ─► top-K search
                                                                      │
        answer + sources  ◄── Groq Llama (streaming) ◄── prompt = rules + context + history + question
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI: sidebar, chat, sources, error display |
| `config.py` | Paths, limits, defaults, env handling |
| `core/document_loader.py` | Validation and text extraction for the four formats |
| `core/text_splitter.py` | Recursive character chunking with metadata |
| `core/embeddings.py` | HuggingFace embeddings (plus an offline keyword embedder for tests) |
| `core/vector_store.py` | FAISS wrapper: add, delete by source, save/load, search |
| `core/retriever.py` | Top-K retrieval, similarity threshold, de-duplication |
| `core/llm.py` | Groq client, streaming, connection check, error translation |
| `core/rag_pipeline.py` | Ties everything together (ingest, re-index, prepare, stream) |
| `database/chat_history.py` | SQLite conversations and messages |
| `utils/prompts.py`, `utils/helpers.py` | Prompts, fixed messages, text cleaning, validation |
| `tests/test_rag.py` | 69 offline tests |

## Installation

Requires **Python 3.11+**.

```powershell
# 1. create and activate a virtual environment (Windows PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# 2. install packages (the first install downloads PyTorch, so it is large)
pip install -r requirements.txt
```

macOS / Linux: `source venv/bin/activate` instead of the Activate.ps1 line.

If PowerShell says scripts are disabled, run this once in the same window and activate again:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## API key

1. Create a free key at <https://console.groq.com/keys>.
2. Copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY=gsk_...
   ```
3. `.env` is already listed in `.gitignore`. Never commit it.

## Run

```powershell
streamlit run app.py
```

The first question (or first processed document) downloads the embedding model, about 90 MB, once. After that it works from the local cache.

**Quick try:** open the sidebar, press **Load sample documents**, and ask the example questions below.

## Example questions (sample documents)

The `data/sample_docs/` folder holds a fictional company handbook (PDF), a product FAQ (DOCX), a RAG primer (TXT) and a sales table (CSV).

- How many days of annual leave do employees get? *(then follow up: "Can unused days be carried over?")*
- What is the home office allowance for remote work?
- How much does the Team plan cost, and what support does it include?
- Which region sold the most units in March? *(CSV rows are retrieved as text, so treat totals across many rows with care: RAG retrieves passages, it does not run calculations.)*
- What is chunk overlap? 
- What is the capital of France? *(not in the documents, so you get the "could not find" reply)*

## Settings

| Setting | Meaning | Tip |
|---|---|---|
| Temperature | Randomness of wording | 0.0 to 0.3 for factual Q&A |
| Chunk size / overlap | Passage length and shared text between neighbours | Applies to newly processed files; press **Re-index all** to apply to existing ones |
| Top-K | Passages sent to the model | 3 to 6 is typical |
| Minimum similarity | Passages scoring below are ignored | Raise for stricter answers, lower if good questions return "could not find" |

Optional environment variables are listed in `.env.example` (model, embedding model, defaults, size limit).

## Testing

```powershell
pytest -q
```

The suite covers upload validation, extraction for all four formats (including corrupted and empty files), chunking, embeddings, vector search and persistence, question answering with follow-ups, missing-context handling, LLM error translation, the SQLite history, and a scripted run of the real Streamlit UI. It runs offline: embeddings use a small deterministic keyword embedder and the LLM is a fake, so no API key or model download is needed.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No Groq API key found" | Create `.env` from `.env.example`, set `GROQ_API_KEY`, restart the app |
| "Groq rejected your API key" | The key is wrong or revoked; create a new one and restart |
| "Could not reach the Groq API" | Check internet, VPN or proxy; use **Test connection** in the sidebar |
| "Model not found" / "model retired" | Groq retires models. Pick another in the sidebar, or set `GROQ_MODEL` in `.env` |
| Rate limit message | Wait a few seconds, lower Top-K, or choose `llama-3.1-8b-instant` |
| "Could not load the embedding model" | The first run needs internet to download it; check connection or proxy |
| "No readable text" for a PDF | It is probably a scanned image. OCR is not supported; use a text-based PDF |
| Good questions return "could not find" | Lower **Minimum similarity** or raise **Top-K** in RAG settings |
| "The search index is missing" | Press **Re-index all** in the sidebar |
| Changed `.env` but nothing happened | Restart Streamlit; `.env` is read at startup |
| `faiss-cpu` fails to install | Upgrade pip first (`python -m pip install -U pip`); if it still fails, try Python 3.11 or 3.12 |

## Notes and limits

- Single-user, local app. The FAISS index is stored with pickle metadata, so only load an index this app created yourself.
- Scanned PDFs (images of text) are not supported, as OCR is outside the requested scope.
- Regenerate raises the temperature slightly so the wording changes.
- Copy opens a small popover with a copy icon (browsers restrict clipboard writes from embedded components, so this is the most reliable approach).
- `langchain-community` prints a deprecation notice at import; the FAISS integration it provides still works and is pinned in `requirements.txt`.
- `data/uploads/`, `data/vector_db/` and `data/chat_history.db` are git-ignored; `data/sample_docs/` is committed.
- `.streamlit/config.toml` sets the theme and a 20 MB upload limit. If you change `MAX_FILE_SIZE_MB`, change `maxUploadSize` to match.
