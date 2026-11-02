"""Groq LLM integration: connection check, model building, streaming, error translation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import config


class LLMError(Exception):
    """A friendly, user-facing LLM problem. `kind` lets callers react to the category."""

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


MISSING_KEY_MESSAGE = (
    "No Groq API key found. Create a free key at https://console.groq.com/keys, copy `.env.example` "
    "to `.env`, set `GROQ_API_KEY=...`, then restart the app."
)


@dataclass
class ConnectionStatus:
    state: str  # connected | missing_key | invalid_key | offline | error
    ok: bool
    message: str
    models: list[str] = field(default_factory=list)


# Error translation
def translate_exception(exc: BaseException) -> LLMError:
    """Map SDK / network exceptions to a friendly LLMError."""
    if isinstance(exc, LLMError):
        return exc
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(exc, "status_code", None)

    if status == 401 or name == "AuthenticationError" or "invalid_api_key" in low:
        return LLMError(
            "Groq rejected your API key (invalid or revoked). Check `GROQ_API_KEY` in your `.env` file "
            "and restart the app.", "auth")
    if status == 403 or name == "PermissionDeniedError":
        return LLMError(
            "Groq denied access to this request (permission or region restriction). "
            "Check your Groq account and the selected model.", "auth")
    if status == 404 or name == "NotFoundError":
        return LLMError(
            "The selected model was not found on Groq. Pick another model in the sidebar "
            "(Groq retires models from time to time).", "model")
    if status == 429 or name == "RateLimitError" or "rate limit" in low:
        return LLMError(
            "Groq rate limit reached. Wait a few seconds and try again, lower Top-K, "
            "or switch to a smaller model.", "rate_limit")
    if status == 413 or "request too large" in low or "context_length" in low or "too large" in low:
        return LLMError(
            "The request was too large for this model. Lower Top-K or the chunk size and try again.", "context")
    if "decommissioned" in low or "model_not_found" in low:
        return LLMError(
            "This model has been retired by Groq. Choose a different model in the sidebar.", "model")
    if status == 400 or name == "BadRequestError":
        return LLMError(f"Groq could not process the request: {text[:300]}", "other")
    if name in {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout"} or isinstance(
        exc, (ConnectionError, TimeoutError)
    ):
        return LLMError(
            "Could not reach the Groq API. Check your internet connection (or proxy/firewall) and try again.",
            "connection")
    if isinstance(status, int) and status >= 500:
        return LLMError("Groq is having temporary problems. Please try again in a moment.", "server")
    return LLMError(f"Unexpected LLM error ({name}): {text[:300]}", "other")


# Building
def build_llm(
    model: str,
    temperature: float,
    api_key: str,
    max_tokens: Optional[int] = None,
) -> Any:
    """Create a ChatGroq client. Raises LLMError if the key is missing."""
    if not api_key:
        raise LLMError(MISSING_KEY_MESSAGE, "missing_key")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise LLMError("`langchain-groq` is not installed. Run: pip install -r requirements.txt") from exc
    try:
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key,
            max_tokens=max_tokens or config.MAX_OUTPUT_TOKENS,
            request_timeout=config.REQUEST_TIMEOUT_S,
            max_retries=2,
        )
    except Exception as exc:  # noqa: BLE001
        raise translate_exception(exc) from exc


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", "")) for part in content
            if isinstance(part, (str, dict))
        )
    return "" if content is None else str(content)


def stream_text(llm: Any, messages: list) -> Iterator[str]:
    """Yield answer text pieces as they arrive from the model."""
    try:
        for chunk in llm.stream(messages):
            text = _content_to_text(getattr(chunk, "content", ""))
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001
        raise translate_exception(exc) from exc


def invoke_text(llm: Any, messages: list) -> str:
    """Single, non-streaming call that returns the full text."""
    try:
        return _content_to_text(getattr(llm.invoke(messages), "content", "")).strip()
    except Exception as exc:  # noqa: BLE001
        raise translate_exception(exc) from exc


# Connection check
_SKIP_MODEL_WORDS = ("guard", "whisper", "tts", "embed", "transcribe", "speech")


def check_connection(api_key: str, timeout: float = 8.0) -> ConnectionStatus:
    """Check credentials and return available text/chat model IDs."""
    if not api_key:
        return ConnectionStatus("missing_key", False, MISSING_KEY_MESSAGE)
    try:
        from groq import Groq

        client = Groq(api_key=api_key, timeout=timeout, max_retries=0)
        ids = sorted({m.id for m in client.models.list().data})

        # Do not restrict this to model IDs containing 'llama': Groq's current
        # recommended GPT-OSS models do not contain that word.
        models = [
            model_id for model_id in ids
            if not any(word in model_id.lower() for word in _SKIP_MODEL_WORDS)
        ]
        return ConnectionStatus("connected", True, "Connected to Groq", models)
    except Exception as exc:  # noqa: BLE001
        err = translate_exception(exc)
        state = {"auth": "invalid_key", "connection": "offline"}.get(err.kind, "error")
        return ConnectionStatus(state, False, str(err))
