from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt import ToolRuntime

import mokioclaw.tools.workspace_tools as workspace_tools


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


def test_file_write_creates_file_and_returns_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)

    command = workspace_tools.file_write_impl(
        path="notes/summary.md",
        content="# Summary\n",
        runtime=_runtime(),
    )

    target = tmp_path / "notes" / "summary.md"
    message = _message_from_command(command)

    assert target.read_text(encoding="utf-8") == "# Summary\n"
    assert "mode: create" in str(message.content)
    assert str(target) in command.update["file_snapshots"]


def test_bash_read_tracks_file_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    target = tmp_path / "demo" / "a.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello\nworld\n", encoding="utf-8")

    command = workspace_tools.bash_impl(
        command="cat demo/a.txt",
        runtime=_runtime(),
    )

    message = _message_from_command(command)
    assert "tracked_reads: demo/a.txt" in str(message.content)
    assert str(target) in command.update["file_snapshots"]


def test_file_edit_requires_fresh_read_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    target = tmp_path / "demo" / "a.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello world\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fresh read snapshot"):
        workspace_tools.file_edit_impl(
            path="demo/a.txt",
            old_string="hello",
            new_string="hi",
            runtime=_runtime({"file_snapshots": {}}),
        )


def test_file_edit_rejects_stale_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    target = tmp_path / "demo" / "a.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello world\n", encoding="utf-8")

    read_command = workspace_tools.bash_impl(
        command="cat demo/a.txt",
        runtime=_runtime(),
    )
    state = {"file_snapshots": read_command.update["file_snapshots"]}
    target.write_text("hello changed world\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after it was read"):
        workspace_tools.file_edit_impl(
            path="demo/a.txt",
            old_string="hello changed",
            new_string="hi",
            runtime=_runtime(state),
        )


def test_file_edit_updates_file_after_read(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    target = tmp_path / "demo" / "a.txt"
    target.parent.mkdir(parents=True)
    target.write_text("hello world\n", encoding="utf-8")

    read_command = workspace_tools.bash_impl(
        command="cat demo/a.txt",
        runtime=_runtime(),
    )
    state = {"file_snapshots": read_command.update["file_snapshots"]}

    edit_command = workspace_tools.file_edit_impl(
        path="demo/a.txt",
        old_string="hello",
        new_string="hi",
        runtime=_runtime(state),
    )
    message = _message_from_command(edit_command)

    assert target.read_text(encoding="utf-8") == "hi world\n"
    assert "file unchanged since last read: ok" in str(message.content)
    assert str(target) in edit_command.update["file_snapshots"]


def test_workspace_tools_can_be_converted_to_openai_tool_schema():
    for tool in (
        workspace_tools.file_write,
        workspace_tools.file_edit,
        workspace_tools.bash,
    ):
        schema = convert_to_openai_tool(tool)
        assert schema["type"] == "function"
        assert "function" in schema
