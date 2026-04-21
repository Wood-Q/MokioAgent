from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mokioclaw.core.context import RunContext
from mokioclaw.core.memory import (
    build_initial_state,
    build_short_term_memory,
    collect_tool_executions,
    extract_final_response,
    render_turn_trace,
)
from mokioclaw.core.state import MokioclawState
from mokioclaw.core.types import LoopOutcome
from mokioclaw.prompts.react_prompt import (
    build_executor_system_prompt,
    build_finalizer_system_prompt,
    build_planner_system_prompt,
)
from mokioclaw.providers.ollama_provider import build_chat_model
from mokioclaw.tools.registry import tools_for_agent, tools_for_prompt


@dataclass(frozen=True)
class PlannerDecision:
    steps: list[str]
    final_response: str = ""


def build_plan_execute_graph(model: str):
    tools = tools_for_agent()
    builder = StateGraph(cast(Any, MokioclawState))
    builder.add_node("planner", cast(Any, _build_planner_node(model=model)))
    builder.add_node(
        "executor",
        cast(Any, _build_executor_node(model=model, tools=tools)),
    )
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("advance", cast(Any, _build_advance_node()))
    builder.add_node("finalizer", cast(Any, _build_finalizer_node(model=model)))
    builder.add_edge(START, "planner")
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "executor": "executor",
            "finalizer": "finalizer",
        },
    )
    builder.add_conditional_edges(
        "executor",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "advance",
        },
    )
    builder.add_edge("tools", "executor")
    builder.add_conditional_edges(
        "advance",
        _route_after_advance,
        {
            "executor": "executor",
            "finalizer": "finalizer",
        },
    )
    builder.add_edge("finalizer", END)
    return builder.compile(name="mokioclaw-plan-execute-graph")


def _build_planner_node(
    *,
    model: str,
) -> Callable[[MokioclawState], dict[str, object]]:
    llm = build_chat_model(model)
    system_message = SystemMessage(
        content=build_planner_system_prompt(tools_for_prompt())
    )

    def plan(state: MokioclawState) -> dict[str, object]:
        response = llm.invoke([system_message, *state["messages"]])
        decision = _parse_planner_response(_coerce_ai_message(response))
        if decision.steps:
            events = [
                "Planner: generated execution plan",
                *[
                    f"Plan Step {index}: {step}"
                    for index, step in enumerate(decision.steps, start=1)
                ],
            ]
            return {
                "plan": decision.steps,
                "completed_steps": [],
                "current_step_index": 0,
                "final_response": "",
                "turn_events": events,
            }

        direct_response = decision.final_response or "我需要更多信息才能继续。"
        return {
            "plan": [],
            "completed_steps": [],
            "current_step_index": 0,
            "final_response": direct_response,
            "turn_events": ["Planner: returned a direct response without execution."],
        }

    return plan


def _build_executor_node(
    *,
    model: str,
    tools: Sequence[BaseTool],
) -> Callable[[MokioclawState], dict[str, list[BaseMessage]]]:
    llm = build_chat_model(model).bind_tools(tools)

    def execute(state: MokioclawState) -> dict[str, list[BaseMessage]]:
        current_step = _current_step(state)
        system_message = SystemMessage(
            content=build_executor_system_prompt(
                tools=tools_for_prompt(),
                plan=state.get("plan", []),
                completed_steps=state.get("completed_steps", []),
                current_step=current_step,
            )
        )
        response = llm.invoke([system_message, *state["messages"]])
        return {"messages": [_coerce_ai_message(response)]}

    return execute


def _build_advance_node() -> Callable[[MokioclawState], dict[str, object]]:
    def advance(state: MokioclawState) -> dict[str, object]:
        plan = state.get("plan", [])
        current_index = state.get("current_step_index", 0)
        if current_index >= len(plan):
            return {}

        completed_steps = [*state.get("completed_steps", []), plan[current_index]]
        return {
            "completed_steps": completed_steps,
            "current_step_index": current_index + 1,
            "turn_events": [
                f"Completed Step {current_index + 1}/{len(plan)}: {plan[current_index]}"
            ],
        }

    return advance


def _build_finalizer_node(
    *,
    model: str,
) -> Callable[[MokioclawState], dict[str, object]]:
    llm = build_chat_model(model)

    def finalize(state: MokioclawState) -> dict[str, object]:
        direct_response = state.get("final_response", "").strip()
        if direct_response and not state.get("plan"):
            message = AIMessage(content=direct_response)
            return {
                "messages": [message],
                "final_response": direct_response,
                "turn_events": ["Finalizer: returned planner response to the user."],
            }

        system_message = SystemMessage(
            content=build_finalizer_system_prompt(
                user_input=state.get("user_input", ""),
                plan=state.get("plan", []),
                completed_steps=state.get("completed_steps", []),
            )
        )
        response = _coerce_ai_message(llm.invoke([system_message, *state["messages"]]))
        return {
            "messages": [response],
            "final_response": _stringify_content(response.content),
            "turn_events": ["Finalizer: summarized completed plan execution."],
        }

    return finalize


def _route_after_planner(state: MokioclawState) -> str:
    return "executor" if state.get("plan") else "finalizer"


def _route_after_advance(state: MokioclawState) -> str:
    if state.get("current_step_index", 0) < len(state.get("plan", [])):
        return "executor"
    return "finalizer"


def _current_step(state: MokioclawState) -> str:
    plan = state.get("plan", [])
    current_index = state.get("current_step_index", 0)
    if current_index >= len(plan):
        return "完成当前任务"
    return plan[current_index]


def _parse_planner_response(message: AIMessage) -> PlannerDecision:
    text = _stringify_content(message.content).strip()
    if not text:
        return PlannerDecision(steps=[], final_response="我需要更多信息才能继续。")

    payload = _extract_json_object(text)
    if payload is None:
        return PlannerDecision(steps=[], final_response=text)

    steps_raw = payload.get("steps", [])
    final_response_raw = payload.get("final_response", "")

    steps: list[str] = []
    if isinstance(steps_raw, list):
        steps = [
            step.strip() for step in steps_raw if isinstance(step, str) and step.strip()
        ]
    elif isinstance(steps_raw, str) and steps_raw.strip():
        steps = [steps_raw.strip()]

    final_response = (
        final_response_raw.strip() if isinstance(final_response_raw, str) else ""
    )

    if steps:
        return PlannerDecision(steps=steps, final_response="")
    if final_response:
        return PlannerDecision(steps=[], final_response=final_response)
    return PlannerDecision(steps=[], final_response=text)


def _extract_json_object(text: str) -> dict[str, object] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None
    return cast(dict[str, object], payload)


def _coerce_ai_message(message: BaseMessage) -> AIMessage:
    if isinstance(message, AIMessage):
        return message
    raise TypeError(f"Expected AIMessage from chat model, got {type(message).__name__}")


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            parts.append(str(item))
        return "\n".join(parts)
    return str(cast(object, content))


@dataclass
class MokioclawSession:
    model: str = "gpt-4o-mini"
    graph: Any = field(init=False)
    state: MokioclawState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.graph = build_plan_execute_graph(self.model)

    def reset(self) -> None:
        self.state = None

    def run_turn(self, user_input: str) -> LoopOutcome:
        input_state, prior_message_count = self._prepare_turn_state(user_input)
        result = cast(MokioclawState, self.graph.invoke(input_state))
        self.state = result
        current_turn_messages = result["messages"][prior_message_count:]
        tool_calls = collect_tool_executions(current_turn_messages)

        return LoopOutcome(
            need_tool=bool(tool_calls),
            raw=render_turn_trace(result.get("turn_events", []), current_turn_messages),
            response=(
                result.get("final_response")
                or extract_final_response(current_turn_messages)
            ),
            tool_calls=tool_calls,
            memory=build_short_term_memory(user_input, current_turn_messages),
        )

    def _prepare_turn_state(
        self,
        user_input: str,
    ) -> tuple[MokioclawState, int]:
        if self.state is None:
            return build_initial_state(user_input), 0

        messages = [*self.state["messages"], HumanMessage(content=user_input)]
        short_term_memory = [
            *self.state.get("short_term_memory", []),
            f"User request: {user_input}",
        ]
        next_state = cast(
            MokioclawState,
            {
                **self.state,
                "messages": messages,
                "user_input": user_input,
                "short_term_memory": short_term_memory,
                "plan": [],
                "completed_steps": [],
                "current_step_index": 0,
                "final_response": "",
                "turn_events": [],
            },
        )
        return next_state, len(self.state["messages"])


def run_single_step(
    user_input: str,
    model: str = "gpt-4o-mini",
    context: RunContext | None = None,
) -> LoopOutcome:
    ctx = context or RunContext(user_input=user_input, model=model)
    return MokioclawSession(model=ctx.model).run_turn(ctx.user_input)
