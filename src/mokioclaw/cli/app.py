from __future__ import annotations

from typing import Annotated

import typer

from mokioclaw.core.loop import run_single_step
from mokioclaw.providers.ollama_provider import default_model

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Single-step tool caller demo for Agent fundamentals.",
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

    typer.echo("=== Model ToolCall (raw) ===")
    typer.echo(outcome.raw)

    if not outcome.need_tool:
        typer.echo("\n=== Assistant Response ===")
        typer.echo(outcome.response)
        return 0

    typer.echo("\n=== Execute Tool ===")
    typer.echo(f"tool={outcome.tool}")
    typer.echo(f"arguments={outcome.arguments}")

    if outcome.tool_error:
        typer.echo("\n=== Tool Error ===")
        typer.echo(outcome.tool_error)
        return 1

    typer.echo("\n=== Tool Result ===")
    typer.echo(outcome.tool_result)
    return 0


@app.command()
def main(
    message: Annotated[str, typer.Argument(help="Natural language request")],
    model: Annotated[str, typer.Option(help="LLM model name")] = default_model(),
) -> None:
    """Run a single-step tool-call loop from a natural language message."""

    raise typer.Exit(code=_render_outcome(message=message, model=model))
