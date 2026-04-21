from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

REACT_TEMPLATE_NAME = "react_system.jinja2"
PLANNER_TEMPLATE_NAME = "planner_system.jinja2"
FINALIZER_TEMPLATE_NAME = "finalizer_system.jinja2"


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


def load_react_template() -> str:
    path = _prompt_dir() / REACT_TEMPLATE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_react_system_prompt(tools: list[dict[str, Any]]) -> str:
    template = PROMPT_ENV.get_template(REACT_TEMPLATE_NAME)
    return template.render(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
        plan_markdown="(planner not provided)",
        completed_steps_markdown="(none)",
        current_step="当前任务",
    ).strip()


def build_planner_system_prompt(tools: list[dict[str, Any]]) -> str:
    template = PROMPT_ENV.get_template(PLANNER_TEMPLATE_NAME)
    return template.render(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
    ).strip()


def build_executor_system_prompt(
    *,
    tools: list[dict[str, Any]],
    plan: list[str],
    completed_steps: list[str],
    current_step: str,
) -> str:
    template = PROMPT_ENV.get_template(REACT_TEMPLATE_NAME)
    return template.render(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2),
        plan_markdown=_render_plan_markdown(plan),
        completed_steps_markdown=_render_completed_steps_markdown(completed_steps),
        current_step=current_step,
    ).strip()


def build_finalizer_system_prompt(
    *,
    user_input: str,
    plan: list[str],
    completed_steps: list[str],
) -> str:
    template = PROMPT_ENV.get_template(FINALIZER_TEMPLATE_NAME)
    return template.render(
        user_input=user_input,
        plan_markdown=_render_plan_markdown(plan),
        completed_steps_markdown=_render_completed_steps_markdown(completed_steps),
    ).strip()


def _render_plan_markdown(plan: list[str]) -> str:
    if not plan:
        return "(no plan)"
    return "\n".join(f"{index}. {step}" for index, step in enumerate(plan, start=1))


def _render_completed_steps_markdown(completed_steps: list[str]) -> str:
    if not completed_steps:
        return "(none)"
    return "\n".join(
        f"{index}. {step}" for index, step in enumerate(completed_steps, start=1)
    )
