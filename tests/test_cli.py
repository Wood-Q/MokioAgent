from __future__ import annotations

from typer.testing import CliRunner

import mokioclaw.cli.app as cli_app
from mokioclaw.core.types import LoopOutcome


runner = CliRunner()


def test_cli_prints_assistant_response(monkeypatch):
    monkeypatch.setattr(
        cli_app,
        "run_single_step",
        lambda message, model: LoopOutcome(
            need_tool=False,
            raw='{"need_tool": false, "response": "你好"}',
            response="你好",
        ),
    )

    result = runner.invoke(cli_app.app, ["你好", "--model", "demo-model"])

    assert result.exit_code == 0
    assert "=== Model ToolCall (raw) ===" in result.output
    assert "=== Assistant Response ===" in result.output
    assert "你好" in result.output


def test_cli_prints_tool_error(monkeypatch):
    monkeypatch.setattr(
        cli_app,
        "run_single_step",
        lambda message, model: LoopOutcome(
            need_tool=True,
            raw='{"need_tool": true, "tool": "move_file"}',
            tool="move_file",
            arguments={"src": "a", "dst": "b"},
            tool_error="boom",
        ),
    )

    result = runner.invoke(cli_app.app, ["移动文件"])

    assert result.exit_code == 1
    assert "=== Tool Error ===" in result.output
    assert "boom" in result.output


def test_cli_prints_runtime_error(monkeypatch):
    def _raise_error(message, model):
        raise RuntimeError("404 page not found")

    monkeypatch.setattr(cli_app, "run_single_step", _raise_error)

    result = runner.invoke(cli_app.app, ["你好"])

    assert result.exit_code == 1
    assert "=== Runtime Error ===" in result.output
    assert "404 page not found" in result.output
    assert "--model qwen3.5:cloud" in result.output
