from __future__ import annotations

from typer.testing import CliRunner

import mokioclaw.cli.app as cli_app
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
