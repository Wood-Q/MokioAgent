from __future__ import annotations

import sys
from enum import StrEnum
from typing import Annotated

import typer

from mokioclaw.core.loop import MokioclawSession, run_single_step
from mokioclaw.core.memory import (
    coerce_todo_snapshots,
    render_notepad,
    render_todo_panel,
)
from mokioclaw.core.state import PendingApprovalState
from mokioclaw.core.types import LoopOutcome
from mokioclaw.providers.ollama_provider import default_model
from mokioclaw.tui.app import run_textual_chat

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Plan & Execute agent demo for Agent fundamentals.",
    no_args_is_help=False,
)

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMANDS = {"/help", "help"}
CLEAR_COMMANDS = {"/clear"}
COMPACT_COMMAND = "/compact"
APPROVE_COMMANDS = {"/approve", "approve"}
DENY_COMMANDS = {"/deny", "deny"}
TODO_COMMANDS = {"/todo", "todo"}
NOTEPAD_COMMANDS = {"/notepad", "notepad"}


class UIChoice(StrEnum):
    AUTO = "auto"
    PLAIN = "plain"
    TEXTUAL = "textual"


def _render_verbose_outcome(outcome: LoopOutcome) -> int:
    typer.echo("=== Agent Trace ===")
    typer.echo(outcome.raw)

    if outcome.tool_error:
        typer.echo("\n=== Tool Error ===")
        typer.echo(outcome.tool_error)
        return 1

    if outcome.tool_calls:
        typer.echo("\n=== Tool Steps ===")
        for index, tool_call in enumerate(outcome.tool_calls, start=1):
            typer.echo(f"{index}. tool={tool_call.name}")
            typer.echo(f"   arguments={tool_call.arguments}")
            typer.echo(f"   result={tool_call.result}")

    if outcome.memory:
        typer.echo("\n=== Memory Snapshot ===")
        for item in outcome.memory:
            typer.echo(f"- {item}")

    if outcome.todos:
        typer.echo("\n=== Todo Panel ===")
        typer.echo(render_todo_panel(outcome.todos))

    if outcome.notepad:
        typer.echo("\n=== NotePad ===")
        typer.echo(render_notepad(outcome.notepad))

    if outcome.verification_nudge:
        typer.echo("\n=== Verification Nudge ===")
        typer.echo(outcome.verification_nudge)

    typer.echo("\n=== Final Response ===")
    typer.echo(outcome.response)
    return 0


def _render_runtime_error(exc: Exception) -> int:
    typer.echo("=== Runtime Error ===")
    typer.echo(str(exc))
    typer.echo(
        "\nHint: for Ollama, set BASE_URL=http://localhost:11434 and "
        "use a local model name such as --model qwen3.5:cloud or "
        "MODEL=qwen3.5:cloud."
    )
    return 1


def _render_outcome(message: str, model: str) -> int:
    try:
        outcome = run_single_step(message, model=model)
    except Exception as exc:
        return _render_runtime_error(exc)
    return _render_verbose_outcome(outcome)


def _run_textual(message: str | None, model: str) -> int:
    try:
        return run_textual_chat(message=message, model=model)
    except Exception as exc:
        return _render_runtime_error(exc)


def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _render_chat_turn(outcome: LoopOutcome) -> None:
    if outcome.pending_approval:
        _render_pending_approval(outcome.pending_approval)
    if outcome.verification_nudge:
        typer.echo("\nVerifier>")
        typer.echo(outcome.verification_nudge)
    response = outcome.response or "(no response)"
    typer.echo(f"\nAssistant> {response}")


def _render_pending_approval(pending_approval: PendingApprovalState) -> None:
    typer.echo("\nApproval Required>")
    typer.echo(str(pending_approval.get("message", "")))


def _render_chat_help() -> None:
    typer.echo(
        "Commands: /help, /todo, /notepad, /clear, "
        "/compact [focus], /approve, /deny, /exit"
    )


def _render_session_todos(session: MokioclawSession) -> None:
    state = getattr(session, "state", None)
    todos = []
    if state:
        todos = coerce_todo_snapshots(state.get("todos") or state.get("todo_snapshot"))
    typer.echo("\nTodo>")
    typer.echo(render_todo_panel(todos) if todos else "No todo items yet.")


def _render_session_notepad(session: MokioclawSession) -> None:
    state = getattr(session, "state", None)
    notes = list(state.get("notepad", [])) if state else []
    typer.echo("\nNotePad>")
    typer.echo(render_notepad(notes) if notes else "No saved notes yet.")


def _parse_compact_command(user_input: str) -> str | None:
    if user_input == COMPACT_COMMAND:
        return ""
    if not user_input.startswith(f"{COMPACT_COMMAND} "):
        return None
    focus = user_input[len(COMPACT_COMMAND) :].strip()
    return focus or ""


def _read_user_input() -> str | None:
    try:
        return input("\nYou> ")
    except EOFError:
        typer.echo("")
        return None


def _run_chat_session(message: str | None, model: str) -> int:
    session = MokioclawSession(model=model)
    typer.echo("Mokioclaw chat mode. Type /help for commands, /exit to quit.")

    pending_message = message
    while True:
        if pending_message is None:
            pending_message = _read_user_input()
        if pending_message is None:
            return 0

        user_input = pending_message.strip()
        pending_message = None

        if not user_input:
            continue
        if user_input in HELP_COMMANDS:
            _render_chat_help()
            continue
        if user_input in CLEAR_COMMANDS:
            session.reset()
            typer.echo("\nSession cleared.")
            continue
        if user_input in APPROVE_COMMANDS:
            outcome = session.resolve_pending_approval(approved=True)
            _render_chat_turn(outcome)
            continue
        if user_input in DENY_COMMANDS:
            outcome = session.resolve_pending_approval(approved=False)
            _render_chat_turn(outcome)
            continue
        if user_input in TODO_COMMANDS:
            _render_session_todos(session)
            continue
        if user_input in NOTEPAD_COMMANDS:
            _render_session_notepad(session)
            continue
        compact_focus = _parse_compact_command(user_input)
        if compact_focus is not None:
            try:
                outcome = session.compact_session(compact_focus or None)
            except Exception as exc:
                _render_runtime_error(exc)
                continue
            _render_chat_turn(outcome)
            continue
        if user_input in EXIT_COMMANDS:
            typer.echo("\nSession ended.")
            return 0

        try:
            outcome = session.run_turn(user_input)
        except Exception as exc:
            _render_runtime_error(exc)
            continue

        _render_chat_turn(outcome)


@app.command()
def main(
    message: Annotated[
        str | None,
        typer.Argument(help="Optional initial natural language request"),
    ] = None,
    model: Annotated[str, typer.Option(help="LLM model name")] = default_model(),
    chat: Annotated[
        bool,
        typer.Option(
            "--chat/--one-shot",
            help="Run as a persistent chat session or a single-turn command.",
        ),
    ] = True,
    ui: Annotated[
        UIChoice,
        typer.Option(
            "--ui",
            help="Interactive interface mode: auto, textual, or plain.",
        ),
    ] = UIChoice.AUTO,
) -> None:
    """Run Mokioclaw as a one-shot command or an interactive chat session."""

    if chat and _stdin_is_interactive():
        if ui in {UIChoice.AUTO, UIChoice.TEXTUAL}:
            raise typer.Exit(code=_run_textual(message=message, model=model))
        raise typer.Exit(code=_run_chat_session(message=message, model=model))
    if message is None:
        typer.echo("A message is required in one-shot mode.")
        raise typer.Exit(code=1)
    raise typer.Exit(code=_render_outcome(message=message, model=model))
