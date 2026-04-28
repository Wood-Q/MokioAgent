from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

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
from mokioclaw.core.project_rules import load_project_rule_messages
from mokioclaw.core.state import MokioclawState, TodoItem
from mokioclaw.core.types import LoopOutcome
from mokioclaw.harness.approvals import (
    approval_from_state,
    approval_to_state,
    collect_pending_approvals,
)
from mokioclaw.prompts.react_prompt import (
    build_compact_system_prompt,
    build_executor_system_prompt,
    build_finalizer_system_prompt,
    build_planner_system_prompt,
)
from mokioclaw.providers.ollama_provider import build_chat_model
from mokioclaw.tools.selector import (
    select_agent_tools_for_executor,
    select_prompt_tools_for_executor,
    select_prompt_tools_for_planner,
)

DEFAULT_RECURSION_LIMIT = 40
MAX_REPEATED_CLARIFICATION_ATTEMPTS = 2
DEFAULT_CONTEXT_CHAR_LIMIT = 24000
DEFAULT_COMPACT_TAIL_MESSAGES = 4
DEFAULT_COMPACTION_FOCUS = (
    "保留当前用户目标、已确认决定、关键文件路径、代码或文件改动、"
    "todo 进度、notepad 发现，以及仍未解决的问题。"
)
COMPACTION_SYSTEM_PREFIX = "[Compacted Session Summary]"
CASUAL_CHAT_SYSTEM_PROMPT = """
你是 Mokioclaw。

当前这轮消息是普通聊天，不是文件整理或工具执行任务。
请自然、简洁地用和用户相同的语言回复：
- 不要把问候、感谢或自我介绍请求转成任务澄清
- 不要追问任务范围、路径或优先级
- 不要主动提到计划、工具、工作流或权限检查
- 如果用户在问你是谁或能做什么，可以简短说明你既能聊天，也能帮助处理代码和工作区任务
""".strip()
CASUAL_CHAT_EXACT_INPUTS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thankyou",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "上午好",
    "中午好",
    "下午好",
    "晚上好",
    "谢谢",
    "多谢",
    "谢了",
    "辛苦了",
    "在吗",
    "你是谁",
    "你会什么",
    "你能做什么",
    "介绍一下你自己",
    "介绍下你自己",
}
TASK_INTENT_MARKERS = (
    "./",
    "../",
    ".txt",
    ".md",
    ".py",
    "/",
    "\\",
    "archive",
    "demo",
    "todo",
    "notepad",
    "tool",
    "bash",
    "文件",
    "目录",
    "文件夹",
    "路径",
    "重命名",
    "归类",
    "整理",
    "移动",
    "新建",
    "创建",
    "修改",
    "编辑",
    "写入",
    "读取",
    "查看",
    "检查",
    "搜索",
    "执行",
    "规划",
    "总结",
    "rename",
    "move",
    "edit",
    "write",
    "read",
    "organize",
    "classify",
    "search",
)


@dataclass(frozen=True)
class PlannerDecision:
    steps: list[str]
    final_response: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    missing_information: list[str] = field(default_factory=list)
    suggested_user_replies: list[str] = field(default_factory=list)
    assumption_if_user_unsure: str = ""


@dataclass(frozen=True)
class CompactionStats:
    before_chars: int
    after_chars: int
    focus: str
    summary: str
    count: int
    automatic: bool = False


def build_plan_execute_graph(model: str):
    builder = StateGraph(cast(Any, MokioclawState))
    builder.add_node("entry", cast(Any, _build_entry_node()))
    builder.add_node("planner", cast(Any, _build_planner_node(model=model)))
    builder.add_node("executor", cast(Any, _build_executor_node(model=model)))
    builder.add_node("approval", cast(Any, _build_approval_node()))
    builder.add_node("tools", cast(Any, _build_dynamic_tools_node()))
    builder.add_node("advance", cast(Any, _build_advance_node()))
    builder.add_node("finalizer", cast(Any, _build_finalizer_node(model=model)))
    builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry",
        _route_after_entry,
        {
            "planner": "planner",
            "tools": "tools",
        },
    )
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
        _route_after_executor,
        {
            "approval": "approval",
            "tools": "tools",
            "advance": "advance",
        },
    )
    builder.add_edge("approval", END)
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


def _build_entry_node() -> Callable[[MokioclawState], dict[str, object]]:
    def enter(state: MokioclawState) -> dict[str, object]:
        return {}

    return enter


def _build_planner_node(
    *,
    model: str,
) -> Callable[[MokioclawState], dict[str, object]]:
    llm = build_chat_model(model)
    def plan(state: MokioclawState) -> dict[str, object]:
        system_message = SystemMessage(
            content=build_planner_system_prompt(
                select_prompt_tools_for_planner(state.get("user_input", ""))
            )
        )
        response = llm.invoke(
            [system_message, *_project_rule_messages(), *state["messages"]]
        )
        decision = _parse_planner_response(_coerce_ai_message(response))
        deterministic_decision = _deterministic_file_placement_plan(
            state.get("user_input", "")
        )
        if deterministic_decision is not None:
            decision = deterministic_decision
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
                "pending_approval": None,
                "approved_tool_call_ids": [],
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
                "pending_approval": None,
                "approved_tool_call_ids": [],
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
            "pending_approval": None,
            "approved_tool_call_ids": [],
            "turn_events": ["Planner: returned a direct response without execution."],
        }

    return plan


def _build_executor_node(
    *,
    model: str,
) -> Callable[[MokioclawState], dict[str, list[BaseMessage] | list[str]]]:
    base_llm = build_chat_model(model)

    def execute(state: MokioclawState) -> dict[str, list[BaseMessage] | list[str]]:
        current_step = _current_step(state)
        prompt_tools = select_prompt_tools_for_executor(state, current_step)
        agent_tools = select_agent_tools_for_executor(state, current_step)
        llm = base_llm.bind_tools(agent_tools)
        system_message = SystemMessage(
            content=build_executor_system_prompt(
                tools=prompt_tools,
                plan=state.get("plan", []),
                completed_steps=state.get("completed_steps", []),
                current_step=current_step,
                todos=cast(list[dict[str, str]], state.get("todos", [])),
                notepad=state.get("notepad", []),
            )
        )
        response = llm.invoke(
            [system_message, *_project_rule_messages(), *state["messages"]]
        )
        return {
            "messages": [_coerce_ai_message(response)],
            "turn_events": [
                "Tool Selection: executor tools -> "
                + ", ".join(tool.name for tool in agent_tools)
            ],
        }

    return execute


def _build_approval_node() -> Callable[[MokioclawState], dict[str, object]]:
    def request_approval(state: MokioclawState) -> dict[str, object]:
        approval = _pending_approval_for_state(state)
        if approval is None:
            return {"pending_approval": None}
        return {
            "pending_approval": approval_to_state(approval),
            "final_response": approval.message,
            "turn_events": [
                "Approval: waiting for human approval before executing tool calls."
            ],
        }

    return request_approval


def _build_dynamic_tools_node() -> Callable[[MokioclawState], dict[str, object]]:
    def run_tools(state: MokioclawState) -> dict[str, object]:
        current_step = _current_step(state)
        tools = select_agent_tools_for_executor(state, current_step)
        node = ToolNode(tools, handle_tool_errors=True)
        return cast(dict[str, object], node.invoke(state))

    return run_tools


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
        response = _coerce_ai_message(
            llm.invoke([system_message, *_project_rule_messages(), *state["messages"]])
        )
        return {
            "messages": [response],
            "final_response": _stringify_content(response.content),
            "turn_events": ["Finalizer: summarized completed plan execution."],
        }

    return finalize


def _route_after_entry(state: MokioclawState) -> str:
    return "tools" if state.get("approved_tool_call_ids") else "planner"


def _route_after_planner(state: MokioclawState) -> str:
    return "executor" if state.get("plan") else "finalizer"


def _route_after_executor(state: MokioclawState) -> str:
    if not _last_ai_tool_calls(state):
        return "advance"
    if _pending_approval_for_state(state) is not None:
        return "approval"
    return "tools"


def _route_after_advance(state: MokioclawState) -> str:
    if state.get("current_step_index", 0) < len(state.get("plan", [])):
        return "executor"
    return "finalizer"


def _last_ai_tool_calls(state: MokioclawState) -> list[dict[str, Any]]:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return cast(list[dict[str, Any]], list(message.tool_calls or []))
    return []


def _pending_approval_for_state(state: MokioclawState):
    return collect_pending_approvals(
        _last_ai_tool_calls(state),
        set(state.get("approved_tool_call_ids", [])),
    )


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


def _deterministic_file_placement_plan(user_input: str) -> PlannerDecision | None:
    folder = _extract_target_folder(user_input)
    file_name = _extract_file_name(user_input)
    if folder is None or file_name is None:
        return None

    target = f"{folder}/{file_name}"
    return PlannerDecision(
        steps=[f"将 {file_name} 移动到 {target}。"],
    )


def _extract_target_folder(text: str) -> str | None:
    patterns = (
        r"(?:建立|创建|新建)(?:一个)?(?P<folder>[A-Za-z0-9_.-]+)文件夹",
        r"(?P<folder>[A-Za-z0-9_.-]+)文件夹",
        r"(?:放进|放到|放入|移动到|移到)(?P<folder>[A-Za-z0-9_.-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group("folder")
    return None


def _extract_file_name(text: str) -> str | None:
    match = re.search(r"(?P<file>[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)", text)
    if match is None:
        return None
    return match.group("file")


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


def _looks_like_casual_chat(user_input: str) -> bool:
    stripped = user_input.strip()
    if not stripped:
        return False

    lowered = stripped.casefold()
    if any(marker in lowered or marker in stripped for marker in TASK_INTENT_MARKERS):
        return False

    normalized = re.sub(r"[\s\W_]+", "", lowered)
    if normalized in CASUAL_CHAT_EXACT_INPUTS:
        return True

    chinese_greetings = ("你好", "您好", "嗨", "哈喽")
    chinese_thanks = ("谢谢", "多谢", "谢了", "辛苦了")
    if any(normalized.startswith(item) for item in chinese_greetings) and len(
        normalized
    ) <= 4:
        return True
    if any(normalized.startswith(item) for item in chinese_thanks) and len(
        normalized
    ) <= 4:
        return True

    return False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _project_rule_messages() -> list[HumanMessage]:
    return load_project_rule_messages()


def _approx_context_chars(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for message in messages:
        total += len(_stringify_content(getattr(message, "content", "")))
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                total += len(json.dumps(tool_call, ensure_ascii=False))
        total += 12
    return total


def _non_summary_tail_messages(
    messages: Sequence[BaseMessage],
    limit: int,
) -> list[BaseMessage]:
    if limit <= 0:
        return []

    filtered: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, SystemMessage) and _stringify_content(
            message.content
        ).startswith(COMPACTION_SYSTEM_PREFIX):
            continue
        filtered.append(message)
    return filtered[-limit:]


def _render_compacted_summary_message(
    *,
    summary: str,
    focus: str,
    count: int,
) -> str:
    return "\n".join(
        [
            f"{COMPACTION_SYSTEM_PREFIX} #{count}",
            "",
            "Treat this as the authoritative summary of earlier conversation context.",
            f"Focus: {focus}",
            "",
            summary.strip(),
        ]
    ).strip()


def _truncate_compaction_summary(summary: str, limit: int) -> str:
    clean = summary.strip()
    if not clean:
        return "## Active Objective\n\n- No earlier context was available to preserve."

    summary_limit = max(1200, limit // 2) if limit > 0 else 1200
    if len(clean) <= summary_limit:
        return clean
    return clean[: summary_limit - 1].rstrip() + "…"


def _format_compaction_raw(stats: CompactionStats) -> str:
    mode = "automatic" if stats.automatic else "manual"
    return "\n".join(
        [
            f"Compaction: {mode} context compaction completed.",
            f"Context chars before: {stats.before_chars}",
            f"Context chars after: {stats.after_chars}",
            f"Compaction count: {stats.count}",
            f"Focus: {stats.focus}",
        ]
    )


def _format_compaction_response(stats: CompactionStats) -> str:
    mode = "自动" if stats.automatic else "手动"
    return "\n".join(
        [
            f"已完成{mode}上下文压缩。",
            f"- 近似上下文长度：{stats.before_chars} -> {stats.after_chars} 字符",
            "- 我会基于压缩后的会话摘要继续后续对话和执行。",
            f"- 压缩焦点：{stats.focus}",
        ]
    )


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
        return _clean_model_text(content)
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


def _clean_model_text(text: str) -> str:
    return text.replace("<|im_end|>", "").strip()


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
    context_char_limit: int | None = None
    compact_tail_messages: int | None = None
    graph: Any = field(init=False)
    state: MokioclawState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.graph = build_plan_execute_graph(self.model)
        if self.context_char_limit is None:
            self.context_char_limit = _env_int(
                "MOKIOCLAW_CONTEXT_CHAR_LIMIT",
                DEFAULT_CONTEXT_CHAR_LIMIT,
            )
        if self.compact_tail_messages is None:
            self.compact_tail_messages = _env_int(
                "MOKIOCLAW_COMPACT_TAIL_MESSAGES",
                DEFAULT_COMPACT_TAIL_MESSAGES,
            )

    def reset(self) -> None:
        self.state = None

    def has_pending_approval(self) -> bool:
        return bool(self.state and self.state.get("pending_approval"))

    def resolve_pending_approval(self, approved: bool) -> LoopOutcome:
        if self.state is None or not self.state.get("pending_approval"):
            return LoopOutcome(
                need_tool=False,
                raw="Approval: no pending approval request.",
                response="当前没有等待审批的工具调用。",
                tool_calls=[],
            )

        pending_approval_state = self.state["pending_approval"]
        assert pending_approval_state is not None
        pending_approval = approval_from_state(
            cast(dict[str, Any], pending_approval_state)
        )
        prior_message_count = len(self.state["messages"])
        if not approved:
            rejected_messages = [
                ToolMessage(
                    content=(
                        "Tool call rejected by human approval. "
                        "Do not retry this action unless the user asks again."
                    ),
                    name=tool_call.name,
                    tool_call_id=tool_call.id,
                )
                for tool_call in pending_approval.tool_calls
            ]
            self.state = cast(
                MokioclawState,
                {
                    **self.state,
                    "messages": [*self.state["messages"], *rejected_messages],
                    "pending_approval": None,
                    "approved_tool_call_ids": [],
                    "final_response": "已取消执行需要审批的工具调用。",
                    "turn_events": ["Approval: human denied pending tool calls."],
                },
            )
            return self._outcome_from_state(
                self.state,
                prior_message_count=prior_message_count,
            )

        approved_ids = [tool_call.id for tool_call in pending_approval.tool_calls]
        resume_state = cast(
            MokioclawState,
            {
                **self.state,
                "pending_approval": None,
                "approved_tool_call_ids": approved_ids,
                "final_response": "",
                "turn_events": ["Approval: human approved pending tool calls."],
            },
        )
        try:
            result = cast(
                MokioclawState,
                self.graph.invoke(
                    resume_state,
                    config={"recursion_limit": DEFAULT_RECURSION_LIMIT},
                ),
            )
        except GraphRecursionError:
            return LoopOutcome(
                need_tool=True,
                raw="Loop Guard: stopped approved execution after recursion limit.",
                response="审批后的执行仍然反复循环，我已停止本轮执行。",
                tool_calls=[],
            )
        result["approved_tool_call_ids"] = []
        self.state = result
        return self._outcome_from_state(result, prior_message_count=prior_message_count)

    def compact_session(self, focus: str | None = None) -> LoopOutcome:
        if self.state is None or not self.state.get("messages"):
            return LoopOutcome(
                need_tool=False,
                raw="Compaction skipped: session is empty.",
                response="当前会话还很短，没有可压缩的上下文。",
                tool_calls=[],
            )

        stats = self._compact_state(focus=focus, automatic=False)
        active_todos = coerce_todo_snapshots(self.state.get("todos"))
        return LoopOutcome(
            need_tool=False,
            raw=_format_compaction_raw(stats),
            response=_format_compaction_response(stats),
            tool_calls=[],
            todos=active_todos or None,
            notepad=list(self.state.get("notepad", [])) or None,
            verification_nudge=self.state.get("verification_nudge") or None,
            memory=[
                f"Compaction count: {stats.count}",
                f"Context chars before: {stats.before_chars}",
                f"Context chars after: {stats.after_chars}",
            ],
        )

    def run_turn(self, user_input: str) -> LoopOutcome:
        auto_compaction_note = self._maybe_auto_compact(user_input)
        if _looks_like_casual_chat(user_input):
            outcome = self._run_casual_turn(user_input)
            if auto_compaction_note:
                return LoopOutcome(
                    need_tool=outcome.need_tool,
                    raw="\n".join([auto_compaction_note, outcome.raw]),
                    response=outcome.response,
                    tool_calls=outcome.tool_calls,
                    todos=outcome.todos,
                    notepad=outcome.notepad,
                    memory=outcome.memory,
                    verification_nudge=outcome.verification_nudge,
                    tool_error=outcome.tool_error,
                )
            return outcome

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
        outcome = self._outcome_from_state(
            result,
            prior_message_count=prior_message_count,
        )
        if auto_compaction_note:
            return LoopOutcome(
                need_tool=outcome.need_tool,
                raw="\n".join([auto_compaction_note, outcome.raw]),
                response=outcome.response,
                tool_calls=outcome.tool_calls,
                todos=outcome.todos,
                notepad=outcome.notepad,
                memory=outcome.memory,
                verification_nudge=outcome.verification_nudge,
                tool_error=outcome.tool_error,
                pending_approval=outcome.pending_approval,
            )
        return outcome

    def _outcome_from_state(
        self,
        state: MokioclawState,
        *,
        prior_message_count: int,
    ) -> LoopOutcome:
        current_turn_messages = state["messages"][prior_message_count:]
        tool_calls = collect_tool_executions(current_turn_messages)
        raw = render_turn_trace(state.get("turn_events", []), current_turn_messages)
        active_todos = coerce_todo_snapshots(state.get("todos"))
        pending_approval = state.get("pending_approval")
        return LoopOutcome(
            need_tool=bool(tool_calls) or bool(pending_approval),
            raw=raw,
            response=(
                state.get("final_response")
                or extract_final_response(current_turn_messages)
            ),
            tool_calls=tool_calls,
            todos=active_todos or None,
            notepad=list(state.get("notepad", [])) or None,
            memory=build_short_term_memory(
                state.get("user_input", ""),
                current_turn_messages,
            ),
            verification_nudge=state.get("verification_nudge") or None,
            pending_approval=pending_approval,
        )

    def _run_casual_turn(self, user_input: str) -> LoopOutcome:
        human_message = HumanMessage(content=user_input)
        prior_messages = [*self.state["messages"]] if self.state else []
        llm = build_chat_model(self.model)
        ai_message = _coerce_ai_message(
            llm.invoke(
                [
                    SystemMessage(content=CASUAL_CHAT_SYSTEM_PROMPT),
                    *_project_rule_messages(),
                    *prior_messages,
                    human_message,
                ]
            )
        )
        response_text = _stringify_content(ai_message.content)
        ai_message = AIMessage(content=response_text)
        turn_messages = [human_message, ai_message]

        if self.state is None:
            next_state = build_initial_state(user_input)
            next_state["messages"].append(ai_message)
            next_state["turn_events"] = [
                "Chat: handled this turn as normal conversation."
            ]
            next_state["final_response"] = response_text
            self.state = next_state
        else:
            self.state = cast(
                MokioclawState,
                {
                    **self.state,
                    "messages": [*self.state["messages"], *turn_messages],
                    "user_input": user_input,
                    "short_term_memory": [
                        *self.state.get("short_term_memory", []),
                        f"User request: {user_input}",
                    ],
                    "plan": [],
                    "completed_steps": [],
                    "current_step_index": 0,
                    "todos": [],
                    "todo_snapshot": [],
                    "notepad": [],
                    "clarification_attempts": 0,
                    "last_clarification_signature": "",
                    "final_response": response_text,
                    "verification_nudge": "",
                    "pending_approval": None,
                    "approved_tool_call_ids": [],
                    "turn_events": [
                        "Chat: handled this turn as normal conversation."
                    ],
                },
            )

        return LoopOutcome(
            need_tool=False,
            raw=render_turn_trace(self.state.get("turn_events", []), turn_messages),
            response=response_text,
            tool_calls=[],
            todos=None,
            notepad=None,
            memory=build_short_term_memory(user_input, turn_messages),
            verification_nudge=None,
        )

    def _maybe_auto_compact(self, upcoming_input: str) -> str | None:
        if self.state is None or not self.context_char_limit:
            return None

        projected_chars = (
            _approx_context_chars(self.state.get("messages", [])) + len(upcoming_input)
        )
        if projected_chars <= self.context_char_limit:
            return None

        stats = self._compact_state(
            focus=(
                "自动压缩时优先保留当前会话的任务目标、最近的关键决策、"
                "文件改动、todo 进度、notepad 发现，以及未完成事项。"
            ),
            automatic=True,
        )
        return _format_compaction_raw(stats)

    def _compact_state(
        self,
        *,
        focus: str | None,
        automatic: bool,
    ) -> CompactionStats:
        assert self.state is not None

        resolved_focus = (
            focus.strip()
            if isinstance(focus, str) and focus.strip()
            else os.getenv("MOKIOCLAW_COMPACT_DEFAULT_FOCUS", "").strip()
            or DEFAULT_COMPACTION_FOCUS
        )
        before_chars = _approx_context_chars(self.state.get("messages", []))
        llm = build_chat_model(self.model)
        response = _coerce_ai_message(
            llm.invoke(
                [
                    SystemMessage(
                        content=build_compact_system_prompt(
                            plan=self.state.get("plan", []),
                            completed_steps=self.state.get("completed_steps", []),
                            todos=cast(
                                list[dict[str, str]],
                                self.state.get("todo_snapshot")
                                or self.state.get("todos", []),
                            ),
                            notepad=self.state.get("notepad", []),
                            verification_nudge=self.state.get(
                                "verification_nudge", ""
                            ),
                            focus=resolved_focus,
                        )
                    ),
                    *_project_rule_messages(),
                    *self.state["messages"],
                ]
            )
        )
        summary = _truncate_compaction_summary(
            _stringify_content(response.content),
            self.context_char_limit or DEFAULT_CONTEXT_CHAR_LIMIT,
        )
        compaction_count = self.state.get("compaction_count", 0) + 1
        summary_message = SystemMessage(
            content=_render_compacted_summary_message(
                summary=summary,
                focus=resolved_focus,
                count=compaction_count,
            )
        )
        tail_messages = _non_summary_tail_messages(
            self.state["messages"],
            self.compact_tail_messages or DEFAULT_COMPACT_TAIL_MESSAGES,
        )
        compacted_messages: list[BaseMessage] = [summary_message, *tail_messages]
        while (
            self.context_char_limit
            and len(compacted_messages) > 1
            and _approx_context_chars(compacted_messages) > self.context_char_limit
        ):
            compacted_messages.pop(1)

        after_chars = _approx_context_chars(compacted_messages)
        self.state = cast(
            MokioclawState,
            {
                **self.state,
                "messages": compacted_messages,
                "compaction_summary": summary,
                "compaction_count": compaction_count,
                "last_compaction_focus": resolved_focus,
                "turn_events": [
                    (
                        "Compaction: context summary refreshed automatically."
                        if automatic
                        else "Compaction: context summary refreshed manually."
                    )
                ],
            },
        )
        return CompactionStats(
            before_chars=before_chars,
            after_chars=after_chars,
            focus=resolved_focus,
            summary=summary,
            count=compaction_count,
            automatic=automatic,
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
                    "compaction_summary": self.state.get("compaction_summary", ""),
                    "compaction_count": self.state.get("compaction_count", 0),
                    "last_compaction_focus": self.state.get(
                        "last_compaction_focus", ""
                    ),
                    "final_response": "",
                    "verification_nudge": "",
                    "pending_approval": None,
                    "approved_tool_call_ids": [],
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
