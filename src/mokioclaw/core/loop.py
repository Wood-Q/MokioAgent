from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from mokioclaw.core.context import RunContext
from mokioclaw.core.memory import (
    build_initial_state,
    build_short_term_memory,
    coerce_todo_snapshots,
    collect_tool_executions,
    extract_final_response,
    render_todo_panel,
    render_turn_trace,
)
from mokioclaw.core.state import MokioclawState, TodoItem
from mokioclaw.core.types import LoopOutcome
from mokioclaw.prompts.react_prompt import (
    build_executor_system_prompt,
    build_finalizer_system_prompt,
    build_planner_system_prompt,
)
from mokioclaw.providers.ollama_provider import build_chat_model
from mokioclaw.tools.registry import tools_for_agent, tools_for_prompt

DEFAULT_RECURSION_LIMIT = 40
MAX_REPEATED_CLARIFICATION_ATTEMPTS = 2


@dataclass(frozen=True)
class PlannerDecision:
    steps: list[str]
    final_response: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    missing_information: list[str] = field(default_factory=list)
    suggested_user_replies: list[str] = field(default_factory=list)
    assumption_if_user_unsure: str = ""


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
                "todos": [],
                "todo_snapshot": [],
                "notepad": [],
                "clarification_attempts": 0,
                "last_clarification_signature": "",
                "final_response": "",
                "verification_nudge": "",
                "turn_events": events,
            }

        direct_response = decision.final_response or "我需要更多信息才能继续。"
        if decision.needs_clarification or _looks_like_clarification(direct_response):
            clarification_result = _build_clarification_state_update(
                decision=decision,
                state=state,
            )
            return {
                "plan": [],
                "completed_steps": [],
                "current_step_index": 0,
                "todos": [],
                "todo_snapshot": [],
                "notepad": [],
                **clarification_result,
            }

        return {
            "plan": [],
            "completed_steps": [],
            "current_step_index": 0,
            "todos": [],
            "todo_snapshot": [],
            "notepad": [],
            "clarification_attempts": 0,
            "last_clarification_signature": "",
            "final_response": direct_response,
            "verification_nudge": "",
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
                todos=cast(list[dict[str, str]], state.get("todos", [])),
                notepad=state.get("notepad", []),
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
        todo_snapshot, active_todos = _sync_todos_after_step(
            state=state,
            completed_index=current_index,
        )
        verification_nudge = _build_verification_nudge(todo_snapshot, active_todos)
        events = [
            f"Completed Step {current_index + 1}/{len(plan)}: {plan[current_index]}"
        ]
        if todo_snapshot:
            events.append(f"Todo Panel:\n{render_todo_panel(todo_snapshot)}")
        if not active_todos and todo_snapshot:
            events.append("Todo Panel cleared after all items completed.")
        if verification_nudge:
            events.append(f"Verification Nudge: {verification_nudge}")

        return {
            "completed_steps": completed_steps,
            "current_step_index": current_index + 1,
            "todos": active_todos,
            "todo_snapshot": todo_snapshot,
            "verification_nudge": verification_nudge,
            "turn_events": events,
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
                todos=cast(
                    list[dict[str, str]],
                    state.get("todo_snapshot") or state.get("todos", []),
                ),
                notepad=state.get("notepad", []),
                verification_nudge=state.get("verification_nudge", ""),
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
        return PlannerDecision(
            steps=[],
            needs_clarification=True,
            clarification_question="请告诉我当前最关键的目标或限制是什么？",
            missing_information=["缺少任务目标或关键限制"],
        )

    payload = _extract_json_object(text)
    if payload is None:
        if _looks_like_clarification(text):
            return PlannerDecision(
                steps=[],
                final_response="",
                needs_clarification=True,
                clarification_question=text,
                missing_information=_infer_missing_information(text),
            )
        return PlannerDecision(steps=[], final_response=text)

    steps_raw = payload.get("steps", [])
    final_response_raw = payload.get("final_response", "")
    needs_clarification_raw = payload.get("needs_clarification", False)
    clarification_question_raw = payload.get("clarification_question", "")
    missing_information_raw = payload.get("missing_information", [])
    suggested_user_replies_raw = payload.get("suggested_user_replies", [])
    assumption_if_user_unsure_raw = payload.get("assumption_if_user_unsure", "")

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
    needs_clarification = bool(needs_clarification_raw)
    clarification_question = (
        clarification_question_raw.strip()
        if isinstance(clarification_question_raw, str)
        else ""
    )
    missing_information = _coerce_string_list(missing_information_raw)
    suggested_user_replies = _coerce_string_list(suggested_user_replies_raw)
    assumption_if_user_unsure = (
        assumption_if_user_unsure_raw.strip()
        if isinstance(assumption_if_user_unsure_raw, str)
        else ""
    )

    if steps:
        return PlannerDecision(
            steps=steps,
            final_response="",
            needs_clarification=False,
            clarification_question="",
            missing_information=[],
            suggested_user_replies=[],
            assumption_if_user_unsure="",
        )
    if needs_clarification or clarification_question or missing_information:
        return PlannerDecision(
            steps=[],
            final_response="",
            needs_clarification=True,
            clarification_question=clarification_question or final_response,
            missing_information=missing_information or _infer_missing_information(text),
            suggested_user_replies=suggested_user_replies,
            assumption_if_user_unsure=assumption_if_user_unsure,
        )
    if final_response:
        return PlannerDecision(steps=[], final_response=final_response)
    if _looks_like_clarification(text):
        return PlannerDecision(
            steps=[],
            final_response="",
            needs_clarification=True,
            clarification_question=text,
            missing_information=_infer_missing_information(text),
        )
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


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _looks_like_clarification(text: str) -> bool:
    lowered = text.casefold()
    clarification_markers = (
        "需要更多信息",
        "需要更多上下文",
        "需要先确认",
        "请提供",
        "请确认",
        "请告诉我",
        "缺少",
        "不明确",
        "?",
        "？",
    )
    return any(marker in lowered for marker in clarification_markers)


def _infer_missing_information(text: str) -> list[str]:
    lowered = text.casefold()
    if "主题" in text or "分类" in text:
        return ["缺少整理方式或分类标准"]
    if "路径" in text or "目录" in text:
        return ["缺少目标路径或工作范围"]
    if "格式" in text:
        return ["缺少期望的输出格式"]
    if "优先" in text or "范围" in text:
        return ["缺少任务范围或优先级"]
    if "need more information" in lowered or "需要更多信息" in text:
        return ["缺少关键任务约束、目标或期望输出"]
    return ["缺少继续执行所需的关键任务信息"]


def _clarification_signature(decision: PlannerDecision) -> str:
    parts = [
        decision.clarification_question.strip().casefold(),
        *[item.strip().casefold() for item in decision.missing_information],
    ]
    return "|".join(part for part in parts if part)


def _build_clarification_state_update(
    *,
    decision: PlannerDecision,
    state: MokioclawState,
) -> dict[str, object]:
    signature = _clarification_signature(decision)
    previous_signature = state.get("last_clarification_signature", "")
    repeated = bool(signature) and signature == previous_signature
    attempts = (
        state.get("clarification_attempts", 0) + 1
        if repeated
        else 1
    )

    message = _format_clarification_message(
        decision,
        repeated=repeated,
        attempts=attempts,
    )
    events = ["Planner: requested clarification with concrete missing information."]
    if repeated:
        events.append(
            f"Loop Guard: repeated clarification detected ({attempts} attempts)."
        )

    return {
        "clarification_attempts": attempts,
        "last_clarification_signature": signature,
        "final_response": message,
        "verification_nudge": "",
        "turn_events": events,
    }


def _format_clarification_message(
    decision: PlannerDecision,
    *,
    repeated: bool,
    attempts: int,
) -> str:
    question = (
        decision.clarification_question.strip()
        or "请告诉我当前最关键的目标、范围或限制是什么？"
    )
    missing_information = (
        decision.missing_information
        or _infer_missing_information(question)
    )

    lines = ["我还缺少以下信息才能继续："]
    lines.extend(
        f"{index}. {item}" for index, item in enumerate(missing_information, start=1)
    )
    lines.extend(["", f"请直接回答这个问题：{question}"])

    if decision.suggested_user_replies:
        lines.append("")
        lines.append("你可以直接回复以下任一项：")
        lines.extend(f"- {option}" for option in decision.suggested_user_replies[:4])

    if decision.assumption_if_user_unsure:
        lines.append("")
        lines.append(
            "如果你不想细化，我也可以按这个默认假设继续："
            f"{decision.assumption_if_user_unsure}"
        )

    if repeated and attempts >= MAX_REPEATED_CLARIFICATION_ATTEMPTS:
        lines.append("")
        lines.append(
            "为避免重复卡住，我先暂停自动推进。请尽量直接回答上面的问题；"
            "如果你愿意，我也可以按默认假设继续。"
        )

    return "\n".join(lines)


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


def _sync_todos_after_step(
    *,
    state: MokioclawState,
    completed_index: int,
) -> tuple[list[TodoItem], list[TodoItem]]:
    plan = state.get("plan", [])
    base_todos = (
        state.get("todo_snapshot")
        or state.get("todos")
        or _todos_from_plan(plan)
    )
    synchronized: list[TodoItem] = []

    for index, todo in enumerate(base_todos):
        status = "pending"
        if index <= completed_index:
            status = "completed"
        elif index == completed_index + 1:
            status = "in_progress"
        synchronized.append(
            TodoItem(
                content=todo["content"],
                status=status,
            )
        )

    if not synchronized and plan:
        synchronized = _todos_from_plan(plan)

    active_todos = (
        []
        if synchronized and all(todo["status"] == "completed" for todo in synchronized)
        else synchronized
    )
    return synchronized, active_todos


def _todos_from_plan(plan: Sequence[str]) -> list[TodoItem]:
    todos: list[TodoItem] = []
    for index, step in enumerate(plan):
        todos.append(
            TodoItem(
                content=step,
                status="in_progress" if index == 0 else "pending",
            )
        )
    return todos


def _build_verification_nudge(
    todo_snapshot: Sequence[TodoItem],
    active_todos: Sequence[TodoItem],
) -> str:
    if active_todos or len(todo_snapshot) < 3:
        return ""

    keywords = ("verify", "verification", "validate", "test", "check", "review")
    chinese_keywords = ("验证", "测试", "检查", "复查", "核对", "确认")
    for todo in todo_snapshot:
        content = todo["content"].casefold()
        if any(keyword in content for keyword in keywords) or any(
            keyword in todo["content"] for keyword in chinese_keywords
        ):
            return ""

    return (
        "Consider adding a dedicated verification step for complex work before "
        "fully closing the task."
    )


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
        try:
            result = cast(
                MokioclawState,
                self.graph.invoke(
                    input_state,
                    config={"recursion_limit": DEFAULT_RECURSION_LIMIT},
                ),
            )
        except GraphRecursionError:
            return LoopOutcome(
                need_tool=True,
                raw=(
                    "Loop Guard: stopped this turn after hitting the graph recursion "
                    "limit."
                ),
                response=(
                    "我停止了本轮执行，因为同一类步骤反复循环、没有形成有效进展。"
                    "请换一种表达方式，或直接告诉我你希望我先确认什么、先执行什么。"
                ),
                tool_calls=[],
                memory=[f"User request: {user_input}"],
            )
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
            todos=coerce_todo_snapshots(
                result.get("todos") or result.get("todo_snapshot")
            ),
            notepad=list(result.get("notepad", [])) or None,
            memory=build_short_term_memory(user_input, current_turn_messages),
            verification_nudge=result.get("verification_nudge") or None,
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
                "todos": [],
                "todo_snapshot": [],
                "notepad": [],
                "final_response": "",
                "verification_nudge": "",
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
