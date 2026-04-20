from __future__ import annotations

from pathlib import Path

from mokioclaw.providers import ollama_provider


def test_provider_kwargs_loaded_from_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENAI_API_KEY="ollama"\nBASE_URL="http://localhost:11434"\nMODEL="qwen3.5:cloud"\n',
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setattr(
        ollama_provider,
        "_dotenv_candidates",
        lambda: [Path(env_file)],
    )
    monkeypatch.setattr(ollama_provider, "_ENV_LOADED", False)

    kwargs = ollama_provider._provider_kwargs()
    assert kwargs["api_key"] == "ollama"
    assert kwargs["base_url"] == "http://localhost:11434/v1"
    assert ollama_provider.default_model() == "qwen3.5:cloud"
