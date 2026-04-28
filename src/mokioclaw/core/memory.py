from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from mokioclaw.core.state import MokioclawState, TodoItem
from mokioclaw.core.types import TodoSnapshot, ToolExecution


def build_initial_state(user_input: str) -> MokioclawState:
    return MokioclawState(
        messages=[HumanMessage(content=user_input)],
        user_input=user_input,
        short_term_memory=[f"User request: {user_input}"],
        file_snapshots={},
        plan=[],
        completed_steps=[],
        current_step_index=0,
        todos=[],
        todo_snapshot=[],
        notepad=[],
        clarification_attempts=0,
        last_clarification_signature="",
        compaction_summary="",
        compaction_count=0,
        last_compaction_focus="",
        final_response="",
        verification_nudge="",
        pending_approval=None,
        approved_tool_call_ids=[],
        turn_events=[],
    )


def collect_tool_executions(messages: Sequence[BaseMessage]) -> list[ToolExecution]:
    executions_by_id: dict[str, ToolExecution] = {}
    ordered_ids: list[str] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                call_id = str(tool_call.get("id", f"call_{len(ordered_ids)}"))
                args = tool_call.get("args")
                arguments = args if isinstance(args, dict) else None
                executions_by_id[call_id] = ToolExecution(
                    name=str(tool_call.get("name", "unknown_tool")),
                    arguments=arguments,
                )
                ordered_ids.append(call_id)
        elif isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", None)
            if call_id and call_id in executions_by_id:
                executions_by_id[call_id].result = _stringify_content(message.content)

    return [executions_by_id[call_id] for call_id in ordered_ids]


def build_short_term_memory(
    user_input: str, messages: Sequence[BaseMessage]
) -> list[str]:
    memory = [f"User request: {user_input}"]
    tool_executions = collect_tool_executions(messages)

    for execution in tool_executions:
        memory.append(f"Tool used: {execution.name}")

    final_response = extract_final_response(messages)
    if final_response:
        memory.append(f"Final answer: {final_response}")

    return memory


def extract_final_response(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            content = _stringify_content(message.content)
            if content:
                return content
    return ""


def render_message_trace(messages: Sequence[BaseMessage]) -> str:
    lines: list[str] = []

    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"Human: {_stringify_content(message.content)}")
            continue

        if isinstance(message, AIMessage):
            tool_calls = message.tool_calls or []
            content = _stringify_content(message.content)

            if tool_calls:
                lines.append("AI: planning next action")
                for tool_call in tool_calls:
                    lines.append(
                        "Tool Call: "
                        f"{tool_call.get('name', 'unknown_tool')}("
                        f"{json.dumps(tool_call.get('args', {}), ensure_ascii=False)})"
                    )
            elif content:
                lines.append(f"AI: {content}")
            continue

        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", None) or "unknown_tool"
            lines.append(
                f"Tool Result [{tool_name}]: {_stringify_content(message.content)}"
            )

    return "\n".join(lines)


def render_turn_trace(events: Sequence[str], messages: Sequence[BaseMessage]) -> str:
    lines = [*events]
    message_trace = render_message_trace(messages)
    if message_trace:
        lines.append(message_trace)
    return "\n".join(lines)


def render_todo_panel(todos: Sequence[TodoItem | TodoSnapshot]) -> str:
    if not todos:
        return "(empty)"
    return "\n".join(
        f"{_todo_marker(_todo_status(todo))} {_todo_content(todo)}" for todo in todos
    )


def render_notepad(notes: Sequence[str]) -> str:
    if not notes:
        return "(empty)"
    return "\n".join(f"- {note}" for note in notes)


def coerce_todo_snapshots(
    todos: Sequence[TodoItem | TodoSnapshot] | None,
) -> list[TodoSnapshot]:
    snapshots: list[TodoSnapshot] = []
    for todo in todos or []:
        snapshots.append(
            TodoSnapshot(
                content=_todo_content(todo),
                status=_todo_status(todo),
            )
        )
    return snapshots


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            parts.append(str(item))
        return "\n".join(parts)
    return str(cast(object, content))


def _todo_marker(status: str) -> str:
    if status == "completed":
        return "[x]"
    if status == "in_progress":
        return "[-]"
    return "[ ]"


def _todo_content(todo: TodoItem | TodoSnapshot) -> str:
    if isinstance(todo, TodoSnapshot):
        return todo.content
    return todo["content"]


def _todo_status(todo: TodoItem | TodoSnapshot) -> str:
    if isinstance(todo, TodoSnapshot):
        return todo.status
    return todo["status"]
