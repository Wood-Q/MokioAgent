from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field

from mokioclaw.core.memory import render_todo_panel
from mokioclaw.core.state import MokioclawState, TodoItem

VERIFICATION_KEYWORDS = (
    "verify",
    "verification",
    "validate",
    "test",
    "tests",
    "check",
    "review",
    "confirm",
    "验证",
    "测试",
    "检查",
    "复查",
    "核对",
    "确认",
)


class TodoWriteEntry(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="Short standalone todo item text.",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        ...,
        description="Current todo status.",
    )


def _build_tool_command(
    *,
    tool_call_id: str,
    content: str,
    state_update: dict[str, object] | None = None,
) -> Command:
    update = dict(state_update or {})
    update["messages"] = [ToolMessage(content=content, tool_call_id=tool_call_id)]
    return Command(update=update)


def _require_tool_call_id(runtime: ToolRuntime) -> str:
    if runtime.tool_call_id is None:
        raise ValueError("Tool runtime is missing a tool_call_id.")
    return runtime.tool_call_id


def _normalize_todos(entries: list[TodoWriteEntry]) -> list[TodoItem]:
    normalized: list[TodoItem] = []
    seen: set[str] = set()

    in_progress_count = 0
    for entry in entries:
        content = entry.content.strip()
        if not content:
            raise ValueError("Todo content must not be empty.")
        lowered = content.casefold()
        if lowered in seen:
            raise ValueError("Todo items must be unique.")
        seen.add(lowered)

        if entry.status == "in_progress":
            in_progress_count += 1

        normalized.append(
            TodoItem(
                content=content,
                status=entry.status,
            )
        )

    if in_progress_count > 1:
        raise ValueError("At most one todo item may be in_progress.")
    if len(normalized) > 8:
        raise ValueError("Todo panel is too large. Keep it to 8 items or fewer.")
    return normalized


def _needs_verification_nudge(todos: list[TodoItem]) -> bool:
    if len(todos) < 3:
        return False
    for todo in todos:
        if any(
            keyword in todo["content"].casefold()
            for keyword in VERIFICATION_KEYWORDS
        ):
            return False
    return True


@tool("todo_write")
def todo_write(
    todos: list[TodoWriteEntry],
    runtime: ToolRuntime,
) -> Command:
    """Create or replace the current todo panel for the active task."""

    normalized = _normalize_todos(todos)
    all_completed = bool(normalized) and all(
        todo["status"] == "completed" for todo in normalized
    )
    verification_nudge = ""
    if all_completed and _needs_verification_nudge(normalized):
        verification_nudge = (
            "Complex task completed without an explicit verification todo. "
            "Consider adding a final verification step before closing."
        )

    panel_state = [] if all_completed else normalized
    summary_lines = [
        "todo_write completed",
        f"items: {len(normalized)}",
        f"panel_cleared: {'yes' if all_completed else 'no'}",
        "panel:",
        render_todo_panel(normalized),
    ]
    if verification_nudge:
        summary_lines.extend(["verification_nudge:", verification_nudge])

    return _build_tool_command(
        tool_call_id=_require_tool_call_id(runtime),
        content="\n".join(summary_lines),
        state_update={
            "todos": panel_state,
            "todo_snapshot": normalized,
            "verification_nudge": verification_nudge,
        },
    )


@tool("notepad_write")
def notepad_write(
    note: str,
    runtime: ToolRuntime,
    replace: bool = False,
) -> Command:
    """Append a durable note for intermediate findings, or replace the notepad."""

    cleaned_note = note.strip()
    if not cleaned_note:
        raise ValueError("note must not be empty.")

    state = cast(MokioclawState, runtime.state or MokioclawState(messages=[]))
    existing_notes = list(state.get("notepad", []))
    updated_notes = [cleaned_note] if replace else [*existing_notes, cleaned_note]

    summary_lines = [
        "notepad_write completed",
        f"mode: {'replace' if replace else 'append'}",
        f"total_notes: {len(updated_notes)}",
        f"latest_note: {cleaned_note}",
    ]

    return _build_tool_command(
        tool_call_id=_require_tool_call_id(runtime),
        content="\n".join(summary_lines),
        state_update={"notepad": updated_notes},
    )


todo_write_impl = cast(Any, todo_write).func
notepad_write_impl = cast(Any, notepad_write).func
