from __future__ import annotations

from mokioclaw.core.context import RunContext
from mokioclaw.core.toolcall_decider import decide_tool_call
from mokioclaw.core.types import LoopOutcome
from mokioclaw.tools.registry import execute_tool_call, tools_for_prompt


def run_single_step(
    user_input: str,
    model: str = "gpt-4o-mini",
    context: RunContext | None = None,
) -> LoopOutcome:
    ctx = context or RunContext(user_input=user_input, model=model)
    decision = decide_tool_call(ctx.user_input, tools_for_prompt(), model=ctx.model)

    if not decision.need_tool:
        return LoopOutcome(
            need_tool=False,
            raw=decision.raw,
            response=decision.response,
        )

    try:
        result = execute_tool_call(decision.tool or "", decision.arguments or {})
    except Exception as exc:
        return LoopOutcome(
            need_tool=True,
            raw=decision.raw,
            tool=decision.tool,
            arguments=decision.arguments,
            tool_error=str(exc),
        )

    return LoopOutcome(
        need_tool=True,
        raw=decision.raw,
        tool=decision.tool,
        arguments=decision.arguments,
        tool_result=result,
    )
