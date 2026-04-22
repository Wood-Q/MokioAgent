from __future__ import annotations

from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolRuntime

import mokioclaw.tools.session_tools as session_tools


def _runtime(state: dict | None = None, *, tool_call_id: str = "call_1") -> ToolRuntime:
    return ToolRuntime(
        state=state or {},
        context=None,
        config={},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def _message_from_command(command) -> ToolMessage:
    messages = command.update["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    return messages[0]


def test_todo_write_creates_panel_snapshot():
    command = session_tools.todo_write_impl(
        todos=[
            session_tools.TodoWriteEntry(
                content="读取目录",
                status="in_progress",
            ),
            session_tools.TodoWriteEntry(
                content="整理文件",
                status="pending",
            ),
        ],
        runtime=_runtime(),
    )

    message = _message_from_command(command)
    assert "todo_write completed" in str(message.content)
    assert len(command.update["todos"]) == 2
    assert len(command.update["todo_snapshot"]) == 2
    assert command.update["verification_nudge"] == ""


def test_todo_write_clears_completed_panel_and_emits_verification_nudge():
    command = session_tools.todo_write_impl(
        todos=[
            session_tools.TodoWriteEntry(
                content="读取目录",
                status="completed",
            ),
            session_tools.TodoWriteEntry(
                content="整理文件",
                status="completed",
            ),
            session_tools.TodoWriteEntry(
                content="生成总结",
                status="completed",
            ),
        ],
        runtime=_runtime(),
    )

    message = _message_from_command(command)
    assert "panel_cleared: yes" in str(message.content)
    assert command.update["todos"] == []
    assert len(command.update["todo_snapshot"]) == 3
    assert "verification" in command.update["verification_nudge"]


def test_notepad_write_appends_and_replaces():
    append_command = session_tools.notepad_write_impl(
        note="先记录第一条发现",
        runtime=_runtime(),
    )
    append_message = _message_from_command(append_command)
    assert "total_notes: 1" in str(append_message.content)
    assert append_command.update["notepad"] == ["先记录第一条发现"]

    replace_command = session_tools.notepad_write_impl(
        note="重新整理后的唯一结论",
        replace=True,
        runtime=_runtime({"notepad": ["旧结论"]}, tool_call_id="call_2"),
    )
    replace_message = _message_from_command(replace_command)
    assert "mode: replace" in str(replace_message.content)
    assert replace_command.update["notepad"] == ["重新整理后的唯一结论"]


def test_session_tools_can_be_converted_to_openai_tool_schema():
    for tool in (
        session_tools.todo_write,
        session_tools.notepad_write,
    ):
        schema = convert_to_openai_tool(tool)
        assert schema["type"] == "function"
        assert "function" in schema
