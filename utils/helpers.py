"""Small, dependency-light helpers shared across the project."""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import config
from utils.prompts import NO_ANSWER_MESSAGE

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ------------------------------------------------------------ text cleaning
def clean_text(text: str) -> str:
    """Normalise text extracted from documents (step 3 of the RAG pipeline)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)      # re-join words hyphenated at line ends
    text = re.sub(r"[ \t\u00a0]+", " ", text)              # collapse runs of spaces
    text = re.sub(r" *\n *", "\n", text)                   # strip spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)                 # at most one blank line
    return text.strip()


# ------------------------------------------------------------ file helpers
def sanitize_filename(name: str) -> str:
    """Return a safe file name (no paths, no odd characters)."""
    name = Path(str(name).replace("\\", "/")).name
    name = re.sub(r"[^\w.\- ]", "_", name, flags=re.UNICODE).strip(" .")
    name = re.sub(r"\s+", " ", name)
    if len(name) > 120:
        stem, suffix = Path(name).stem[:100], Path(name).suffix[:15]
        name = stem + suffix
    return name or "document"


def get_extension(name: str) -> str:
    return Path(name).suffix.lower()


def format_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


# ------------------------------------------------------- question handling
def validate_question(question: Optional[str]) -> tuple[bool, str]:
    """Return (is_valid, message). Message explains what to fix when invalid."""
    q = (question or "").strip()
    if not q:
        return False, "Type a question first."
    if len(q) > config.MAX_QUESTION_CHARS:
        return False, f"That question is too long ({len(q)} characters). Keep it under {config.MAX_QUESTION_CHARS}."
    if not re.search(r"\w", q, flags=re.UNICODE):
        return False, "That doesn't look like a question. Try using words."
    return True, ""


_GREETING = re.compile(
    r"^(hi|hello|hey|hiya|howdy|greetings|good (morning|afternoon|evening))"
    r"( there| everyone| bot| assistant)?[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(
    r"^(thanks|thank you|thx|ty|many thanks|thank you so much)( a lot| so much)?[\s!.,?]*$",
    re.IGNORECASE,
)
_HELP = re.compile(r"^(who are you|what are you|what can you do|help)[\s!.?]*$", re.IGNORECASE)


def classify_small_talk(question: str) -> Optional[str]:
    """Detect greetings / thanks / 'what can you do' so they don't hit retrieval."""
    q = question.strip()
    if _GREETING.match(q):
        return "greeting"
    if _THANKS.match(q):
        return "thanks"
    if _HELP.match(q):
        return "help"
    return None


def is_fallback_answer(text: str) -> bool:
    return NO_ANSWER_MESSAGE.lower() in (text or "").lower()


# ---------------------------------------------------------------- sources
def source_label(meta: Mapping[str, Any]) -> str:
    """Human-readable reference like 'handbook.pdf, p. 3' or 'sales.csv, rows 1-20'."""
    name = str(meta.get("source", "unknown"))
    if meta.get("page"):
        return f"{name}, p. {meta['page']}"
    if meta.get("rows"):
        return f"{name}, rows {meta['rows']}"
    return name


def truncate(text: str, limit: int = 40) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def format_timestamp(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %H:%M")
    except (TypeError, ValueError):
        return iso or ""


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ----------------------------------------------------------------- export
def conversation_to_markdown(title: str, messages: list[dict]) -> str:
    lines = [f"# {title}", "", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}_", ""]
    for msg in messages:
        who = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {who}  \n<sub>{format_timestamp(msg.get('created_at', ''))}</sub>")
        lines.append("")
        lines.append(msg["content"])
        sources = msg.get("sources") or []
        if sources:
            lines.append("")
            lines.append("**Sources:**")
            for label in dedupe_keep_order(source_label(s) for s in sources):
                lines.append(f"- {label}")
        lines.append("")
    return "\n".join(lines)
