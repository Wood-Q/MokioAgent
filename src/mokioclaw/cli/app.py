from __future__ import annotations

import sys
from typing import Annotated

import typer

from mokioclaw.core.loop import MokioclawSession, run_single_step
from mokioclaw.core.types import LoopOutcome
from mokioclaw.providers.ollama_provider import default_model

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Minimal ReAct agent demo for Agent fundamentals.",
    no_args_is_help=False,
)

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMANDS = {"/help", "help"}
CLEAR_COMMANDS = {"/clear"}


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


def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _render_chat_turn(outcome: LoopOutcome) -> None:
    response = outcome.response or "(no response)"
    typer.echo(f"\nAssistant> {response}")


def _render_chat_help() -> None:
    typer.echo("Commands: /help, /clear, /exit")


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
) -> None:
    """Run Mokioclaw as a one-shot command or an interactive chat session."""

    if chat and _stdin_is_interactive():
        raise typer.Exit(code=_run_chat_session(message=message, model=model))
    if message is None:
        typer.echo("A message is required in one-shot mode.")
        raise typer.Exit(code=1)
    raise typer.Exit(code=_render_outcome(message=message, model=model))
