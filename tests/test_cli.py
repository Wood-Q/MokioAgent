from __future__ import annotations

from typer.testing import CliRunner

import mokioclaw.cli.app as cli_app
from mokioclaw.core.loop import MokioclawSession
from mokioclaw.core.types import LoopOutcome, ToolExecution

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


def test_cli_prints_runtime_error(monkeypatch):
    def _raise_error(message, model):
        raise RuntimeError("404 page not found")

    monkeypatch.setattr(cli_app, "run_single_step", _raise_error)

    result = runner.invoke(cli_app.app, ["你好"])

    assert result.exit_code == 1
    assert "=== Runtime Error ===" in result.output
    assert "404 page not found" in result.output
    assert "--model qwen3.5:cloud" in result.output


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
                )
            return LoopOutcome(
                need_tool=False,
                raw="Human: 按主题分类\nAI: 好的",
                response="好的，我会按主题分类整理。",
            )

    monkeypatch.setattr(cli_app, "MokioclawSession", FakeSession)
    monkeypatch.setattr(cli_app, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(
        cli_app.app,
        ["帮我整理一下 archive 和 demo"],
        input="按主题分类\n/exit\n",
    )

    assert result.exit_code == 0
    assert "Mokioclaw chat mode" in result.output
    assert "Assistant> 你想按主题分类，还是统一格式？" in result.output
    assert "Assistant> 好的，我会按主题分类整理。" in result.output
    assert "Session ended." in result.output
