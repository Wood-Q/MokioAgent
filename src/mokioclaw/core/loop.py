from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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


def build_react_graph(model: str):
    tools = tools_for_agent()
    builder = StateGraph(cast(Any, MokioclawState))
    builder.add_node(
        "agent",
        cast(Any, _build_agent_node(model=model, tools=tools)),
    )
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )
    builder.add_edge("tools", "agent")
    return builder.compile(name="mokioclaw-react-graph")


def _build_agent_node(
    *, model: str, tools: Sequence[BaseTool]
) -> Callable[[MokioclawState], dict[str, list[BaseMessage]]]:
    llm = build_chat_model(model).bind_tools(tools)
    system_message = SystemMessage(
        content=build_react_system_prompt(tools_for_prompt())
    )

    def call_model(state: MokioclawState) -> dict[str, list[BaseMessage]]:
        response = llm.invoke([system_message, *state["messages"]])
        return {"messages": [_coerce_ai_message(response)]}

    return call_model


def _coerce_ai_message(message: BaseMessage) -> AIMessage:
    if isinstance(message, AIMessage):
        return message
    raise TypeError(f"Expected AIMessage from chat model, got {type(message).__name__}")


def run_single_step(
    user_input: str,
    model: str = "gpt-4o-mini",
    context: RunContext | None = None,
) -> LoopOutcome:
    ctx = context or RunContext(user_input=user_input, model=model)
    graph = build_react_graph(ctx.model)
    result = graph.invoke(build_initial_state(ctx.user_input))
    messages = result["messages"]
    tool_calls = collect_tool_executions(messages)

    return LoopOutcome(
        need_tool=bool(tool_calls),
        raw=render_message_trace(messages),
        response=extract_final_response(messages),
        tool_calls=tool_calls,
        memory=build_short_term_memory(ctx.user_input, messages),
    )
