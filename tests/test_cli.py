from __future__ import annotations

from typer.testing import CliRunner

import mokioclaw.cli.app as cli_app
from mokioclaw.core.loop import MokioclawSession
from mokioclaw.core.types import LoopOutcome, TodoSnapshot, ToolExecution

runner = CliRunner()


def test_cli_prints_assistant_response(monkeypatch):
    monkeypatch.setattr(
        cli_app,
        "run_single_step",
        lambda message, model: LoopOutcome(
            need_tool=False,
            raw="Human: 你好\nAI: 你好",
            response="你好",
            memory=["User request: 你好", "Final answer: 你好"],
        ),
    )

    result = runner.invoke(cli_app.app, ["你好", "--model", "demo-model"])

    assert result.exit_code == 0
    assert "=== Agent Trace ===" in result.output
    assert "=== Final Response ===" in result.output
    assert "你好" in result.output


def test_cli_prints_tool_steps(monkeypatch):
    monkeypatch.setattr(
        cli_app,
        "run_single_step",
        lambda message, model: LoopOutcome(
            need_tool=True,
            raw="Human: 移动文件\nAI: planning next action",
            response="文件已移动",
            tool_calls=[
                ToolExecution(
                    name="move_file",
                    arguments={"src": "a", "dst": "b"},
                    result="Moved file from 'a' to 'b'.",
                )
            ],
            memory=["User request: 移动文件", "Tool used: move_file"],
        ),
    )

    result = runner.invoke(cli_app.app, ["移动文件"])

    assert result.exit_code == 0
    assert "=== Tool Steps ===" in result.output
    assert "move_file" in result.output
    assert "Moved file from 'a' to 'b'." in result.output


def test_cli_prints_todo_notepad_and_verification_sections(monkeypatch):
    monkeypatch.setattr(
        cli_app,
        "run_single_step",
        lambda message, model: LoopOutcome(
            need_tool=True,
            raw="Planner: generated execution plan",
            response="整理完成。",
            todos=[
                TodoSnapshot(content="读取目录", status="completed"),
                TodoSnapshot(content="生成总结", status="completed"),
            ],
            notepad=["目录中包含 2 个文本文件"],
            verification_nudge="Consider adding a verification step.",
        ),
    )

    result = runner.invoke(cli_app.app, ["整理目录"])

    assert result.exit_code == 0
    assert "=== Todo Panel ===" in result.output
    assert "[x] 读取目录" in result.output
    assert "=== NotePad ===" in result.output
    assert "目录中包含 2 个文本文件" in result.output
    assert "=== Verification Nudge ===" in result.output


def test_cli_prints_runtime_error(monkeypatch):
    def _raise_error(message, model):
        raise RuntimeError("404 page not found")

    monkeypatch.setattr(cli_app, "run_single_step", _raise_error)

    result = runner.invoke(cli_app.app, ["你好"])

    assert result.exit_code == 1
    assert "=== Runtime Error ===" in result.output
    assert "404 page not found" in result.output
    assert "--model qwen3.5:cloud" in result.output


def test_cli_uses_textual_ui_by_default_in_interactive_mode(monkeypatch):
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)
    monkeypatch.setattr(
        cli_app,
        "run_textual_chat",
        lambda message, model: 0,
    )

    result = runner.invoke(cli_app.app, ["帮我整理一下 archive 和 demo"])

    assert result.exit_code == 0


def test_cli_runs_interactive_chat_until_exit(monkeypatch):
    class FakeSession(MokioclawSession):
        def __init__(self, model: str):
            self.model = model
            self.turns: list[str] = []

        def reset(self) -> None:
            self.turns.clear()

        def run_turn(self, user_input: str) -> LoopOutcome:
            self.turns.append(user_input)
            if len(self.turns) == 1:
                return LoopOutcome(
                    need_tool=False,
                    raw="Human: 帮我整理\nAI: 你想按什么方式整理？",
                    response="你想按主题分类，还是统一格式？",
                    todos=[
                        TodoSnapshot(
                            content="确认整理方式",
                            status="in_progress",
                        )
                    ],
                )
            return LoopOutcome(
                need_tool=False,
                raw="Human: 按主题分类\nAI: 好的",
                response="好的，我会按主题分类整理。",
                todos=[
                    TodoSnapshot(
                        content="确认整理方式",
                        status="completed",
                    ),
                    TodoSnapshot(
                        content="按主题完成整理",
                        status="in_progress",
                    ),
                ],
            )

    monkeypatch.setattr(cli_app, "MokioclawSession", FakeSession)
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(
        cli_app.app,
        ["帮我整理一下 archive 和 demo", "--ui", "plain"],
        input="按主题分类\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Mokioclaw chat mode" in result.output
    assert "Todo>" not in result.output
    assert "Assistant> 你想按主题分类，还是统一格式？" in result.output
    assert "Assistant> 好的，我会按主题分类整理。" in result.output
    assert "Session ended." in result.output


def test_cli_plain_chat_supports_todo_and_notepad_commands(monkeypatch):
    class FakeSession(MokioclawSession):
        def __init__(self, model: str):
            self.model = model
            self.state = {
                "todos": [],
                "todo_snapshot": [
                    {"content": "读取目录", "status": "completed"},
                    {"content": "整理文件", "status": "in_progress"},
                ],
                "notepad": ["目录中包含 2 个文本文件"],
            }

        def reset(self) -> None:
            return None

        def run_turn(self, user_input: str) -> LoopOutcome:
            raise AssertionError("run_turn should not be used for state commands")

    monkeypatch.setattr(cli_app, "MokioclawSession", FakeSession)
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(
        cli_app.app,
        ["--ui", "plain"],
        input="/todo\n/notepad\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Todo>" in result.output
    assert "[x] 读取目录" in result.output
    assert "NotePad>" in result.output
    assert "目录中包含 2 个文本文件" in result.output


def test_cli_plain_chat_supports_approval_commands(monkeypatch):
    class FakeSession(MokioclawSession):
        def __init__(self, model: str):
            self.model = model
            self.decisions: list[bool] = []

        def reset(self) -> None:
            return None

        def run_turn(self, user_input: str) -> LoopOutcome:
            return LoopOutcome(
                need_tool=True,
                raw="Approval: waiting",
                response="Human approval required before executing this action.",
                pending_approval={
                    "id": "call_1",
                    "message": "Human approval required before executing this action.",
                    "tool_calls": [],
                },
            )

        def resolve_pending_approval(self, approved: bool) -> LoopOutcome:
            self.decisions.append(approved)
            return LoopOutcome(
                need_tool=False,
                raw="Approval resolved",
                response="approval resolved",
            )

    monkeypatch.setattr(cli_app, "MokioclawSession", FakeSession)
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(
        cli_app.app,
        ["--ui", "plain"],
        input="move file\n/approve\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Approval Required>" in result.output
    assert "Human approval required" in result.output
    assert "Assistant> approval resolved" in result.output


def test_cli_plain_chat_supports_compact_command(monkeypatch):
    class FakeSession(MokioclawSession):
        def __init__(self, model: str):
            self.model = model
            self.compact_focuses: list[str | None] = []

        def reset(self) -> None:
            return None

        def run_turn(self, user_input: str) -> LoopOutcome:
            raise AssertionError("run_turn should not be used for /compact")

        def compact_session(self, focus: str | None = None) -> LoopOutcome:
            self.compact_focuses.append(focus)
            return LoopOutcome(
                need_tool=False,
                raw="Compaction: manual context compaction completed.",
                response="已完成手动上下文压缩。",
            )

    monkeypatch.setattr(cli_app, "MokioclawSession", FakeSession)
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(
        cli_app.app,
        ["--ui", "plain"],
        input="/compact 保留文件修改\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Assistant> 已完成手动上下文压缩。" in result.output
    assert "Session ended." in result.output
