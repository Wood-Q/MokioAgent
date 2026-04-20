from __future__ import annotations

from typing import Annotated

import typer

from mokioclaw.core.loop import run_single_step
from mokioclaw.providers.ollama_provider import default_model

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Minimal ReAct agent demo for Agent fundamentals.",
    no_args_is_help=True,
)


def _render_outcome(message: str, model: str) -> int:
    try:
        outcome = run_single_step(message, model=model)
    except Exception as exc:
        typer.echo("=== Runtime Error ===")
        typer.echo(str(exc))
        typer.echo(
            "\nHint: for Ollama, set BASE_URL=http://localhost:11434 and "
            "use a local model name such as --model qwen3.5:cloud or "
            "MODEL=qwen3.5:cloud."
        )
        return 1

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


@app.command()
def main(
    message: Annotated[str, typer.Argument(help="Natural language request")],
    model: Annotated[str, typer.Option(help="LLM model name")] = default_model(),
) -> None:
    """Run a minimal ReAct agent loop from a natural language message."""

    raise typer.Exit(code=_render_outcome(message=message, model=model))
