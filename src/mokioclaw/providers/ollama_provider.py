from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


_ENV_LOADED = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dotenv_candidates() -> list[Path]:
    candidates = [Path.cwd() / ".env", _project_root() / ".env"]
    unique: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(item)
    return unique


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for path in _dotenv_candidates():
        load_dotenv(dotenv_path=path, override=False)
    _ENV_LOADED = True


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.path.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _provider_kwargs() -> dict[str, str]:
    _ensure_env_loaded()
    kwargs: dict[str, str] = {}

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    # BASE_URL keeps compatibility with local OpenAI-compatible endpoints
    # such as Ollama; OPENAI_BASE_URL is accepted as a fallback alias.
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = _normalize_base_url(base_url)

    return kwargs


def default_model() -> str:
    _ensure_env_loaded()
    return os.getenv("MODEL") or os.getenv("OLLAMA_MODEL") or "gpt-4o-mini"


def build_chat_model(model: str) -> ChatOpenAI:
    # Kept in a dedicated provider module so we can replace this
    # with a true Ollama backend without touching loop/cli code.
    return ChatOpenAI(model=model, temperature=0, **_provider_kwargs())


def invoke_chat(
    messages: Sequence[tuple[str, str]], model: str = "gpt-4o-mini"
) -> str:
    llm = build_chat_model(model)
    message = llm.invoke(list(messages))

    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        return "\n".join(str(item) for item in message.content)
    return str(message.content)
