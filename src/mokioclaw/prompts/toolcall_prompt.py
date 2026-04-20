from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_NAME = "toolcall.jinja2"


def _prompt_dir() -> Path:
    return Path(__file__).resolve().parent


def _build_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_prompt_dir())),
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )


PROMPT_ENV = _build_environment()


def load_toolcall_template() -> str:
    path = _prompt_dir() / TEMPLATE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_toolcall_system_prompt(tools: list[dict[str, Any]]) -> str:
    template = PROMPT_ENV.get_template(TEMPLATE_NAME)
    return template.render(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
    ).strip()
