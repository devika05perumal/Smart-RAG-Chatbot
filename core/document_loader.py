"""Document loading and text extraction (PDF, DOCX, TXT, CSV).

Files are only ever *parsed* - never executed - and are validated (extension,
size, content signature) before any parser touches them.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.documents import Document

import config
from utils.helpers import clean_text, format_file_size, get_extension


class DocumentLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be used."""


# ------------------------------------------------------------- validation
def validate_upload(filename: str, data: bytes) -> None:
    """Validate an uploaded file's name, size and content signature."""
    ext = get_extension(filename)
    if ext not in config.SUPPORTED_EXTENSIONS:
        supported = ", ".join(e.lstrip(".").upper() for e in config.SUPPORTED_EXTENSIONS)
        raise DocumentLoadError(
            f"'{filename}' is not a supported file type ({ext or 'no extension'}). Supported formats: {supported}."
        )
    if not data:
        raise DocumentLoadError(f"'{filename}' is empty.")
    limit = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data) > limit:
        raise DocumentLoadError(
            f"'{filename}' is {format_file_size(len(data))}, which exceeds the {config.MAX_FILE_SIZE_MB} MB limit."
        )
    head = data[:8]
    if ext == ".pdf" and b"%PDF" not in data[:1024]:
        raise DocumentLoadError(f"'{filename}' is not a valid PDF (missing PDF header).")
    if ext == ".docx" and not head.startswith(b"PK"):
        raise DocumentLoadError(f"'{filename}' is not a valid DOCX file (it is not a Word document archive).")
    if ext in (".txt", ".csv"):
        is_utf16 = data.startswith((b"\xff\xfe", b"\xfe\xff"))
        if not is_utf16 and b"\x00" in data[:4096]:
            raise DocumentLoadError(f"'{filename}' looks like a binary file, not plain text.")


# --------------------------------------------------------------- loaders
def _load_pdf(path: Path, name: str) -> list[Document]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:  # noqa: BLE001 - any decrypt failure means we can't read it
                unlocked = 0
            if not unlocked:
                raise DocumentLoadError(f"'{name}' is password-protected. Remove the password and upload it again.")
        docs: list[Document] = []
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = clean_text(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - skip a single unreadable page, keep the rest
                text = ""
            if text:
                docs.append(Document(page_content=text, metadata={"source": name, "page": page_no, "type": "pdf"}))
        return docs
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - pypdf raises many different error types
        raise DocumentLoadError(f"'{name}' appears to be corrupted or is not a readable PDF.") from exc


def _load_docx(path: Path, name: str) -> list[Document]:
    from docx import Document as DocxDocument
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = DocxDocument(str(path))
    except (zipfile.BadZipFile, KeyError, ValueError, OSError) as exc:
        raise DocumentLoadError(f"'{name}' appears to be corrupted or is not a valid DOCX file.") from exc
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"'{name}' could not be read as a Word document.") from exc

    parts: list[str] = []
    # Walk the body in document order so tables stay next to the text around them.
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells: list[str] = []
                for cell in row.cells:
                    value = cell.text.strip().replace("\n", " ")
                    if value and (not cells or cells[-1] != value):  # merged cells repeat their text
                        cells.append(value)
                if cells:
                    parts.append(" | ".join(cells))
    text = clean_text("\n".join(parts))
    return [Document(page_content=text, metadata={"source": name, "page": None, "type": "docx"})] if text else []


def _read_text_file(path: Path, name: str) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    else:
        encodings = ["utf-8-sig", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(f"'{name}' uses a text encoding that could not be decoded.")


def _load_txt(path: Path, name: str) -> list[Document]:
    text = clean_text(_read_text_file(path, name))
    return [Document(page_content=text, metadata={"source": name, "page": None, "type": "txt"})] if text else []


def _load_csv(path: Path, name: str) -> list[Document]:
    df: Optional[pd.DataFrame] = None
    last_error: Optional[Exception] = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                path, dtype=str, keep_default_na=False, encoding=enc,
                sep=None, engine="python", on_bad_lines="skip",
            )
            break
        except pd.errors.EmptyDataError:
            raise DocumentLoadError(f"'{name}' contains no data.") from None
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError, csv.Error) as exc:
            last_error = exc
    if df is None:
        raise DocumentLoadError(f"'{name}' could not be parsed as a CSV file.") from last_error
    if df.empty:
        raise DocumentLoadError(f"'{name}' has headers but no data rows.")

    columns = [str(c).strip() for c in df.columns]
    header = "Columns: " + ", ".join(columns)
    block = config.CSV_ROWS_PER_BLOCK
    docs: list[Document] = []
    for start in range(0, len(df), block):
        chunk_df = df.iloc[start:start + block]
        lines = [header]
        for offset, (_, row) in enumerate(chunk_df.iterrows(), start=start + 1):
            pairs = [f"{col}: {str(val).strip()}" for col, val in zip(columns, row) if str(val).strip()]
            if pairs:
                lines.append(f"Row {offset}: " + "; ".join(pairs))
        if len(lines) > 1:
            docs.append(Document(
                page_content=clean_text("\n".join(lines)),
                metadata={"source": name, "page": None, "rows": f"{start + 1}-{start + len(chunk_df)}", "type": "csv"},
            ))
    return docs


_LOADERS = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt, ".csv": _load_csv}


def load_document(path: Path | str, display_name: Optional[str] = None) -> list[Document]:
    """Extract cleaned text from a file as LangChain Documents.

    Raises DocumentLoadError (with a friendly message) for unsupported,
    corrupted or empty files.
    """
    path = Path(path)
    name = display_name or path.name
    ext = get_extension(name)
    loader = _LOADERS.get(ext)
    if loader is None:
        raise DocumentLoadError(f"'{name}' is not a supported file type.")
    if not path.exists():
        raise DocumentLoadError(f"'{name}' could not be found on disk.")
    try:
        docs = loader(path, name)
    except DocumentLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"'{name}' could not be read: {exc}") from exc
    if not docs:
        hint = " It may be a scanned/image-only PDF (OCR is not supported)." if ext == ".pdf" else ""
        raise DocumentLoadError(f"No readable text was found in '{name}'.{hint}")
    return docs
