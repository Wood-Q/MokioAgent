from __future__ import annotations

from langchain.agents import create_agent

from mokioclaw.core.context import RunContext
from mokioclaw.core.memory import (
    build_initial_state,
    build_short_term_memory,
    collect_tool_executions,
    extract_final_response,
    render_message_trace,
)
from mokioclaw.core.state import MokioclawState
from mokioclaw.core.types import LoopOutcome
from mokioclaw.prompts.react_prompt import build_react_system_prompt
from mokioclaw.providers.ollama_provider import build_chat_model
from mokioclaw.tools.registry import tools_for_agent, tools_for_prompt


def run_single_step(
    user_input: str,
    model: str = "gpt-4o-mini",
    context: RunContext | None = None,
) -> LoopOutcome:
    ctx = context or RunContext(user_input=user_input, model=model)
    agent = create_agent(
        model=build_chat_model(ctx.model),
        tools=tools_for_agent(),
        system_prompt=build_react_system_prompt(tools_for_prompt()),
        state_schema=MokioclawState,
        name="mokioclaw-react-agent",
    )
    result = agent.invoke(build_initial_state(ctx.user_input))
    messages = result["messages"]
    tool_calls = collect_tool_executions(messages)

    return LoopOutcome(
        need_tool=bool(tool_calls),
        raw=render_message_trace(messages),
        response=extract_final_response(messages),
        tool_calls=tool_calls,
        memory=build_short_term_memory(ctx.user_input, messages),
    )
