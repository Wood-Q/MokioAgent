import re

import requests
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ========== 1️⃣ 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气情况"""
    lat, lon = 34.75, 113.62
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    res = requests.get(url, timeout=15).json()
    weather = res["current_weather"]
    return (
        f"{city}当前天气 -> 温度: {weather['temperature']}°C, "
        f"风速: {weather['windspeed']}km/h"
    )


@tool
def recommend_clothes(temperature: float) -> str:
    """根据温度给出简单穿衣建议"""
    if temperature >= 28:
        return "天气较热，建议短袖、薄裤，注意防晒和补水。"
    if temperature >= 20:
        return "天气温和，建议长袖或薄外套，穿着轻便即可。"
    if temperature >= 12:
        return "天气偏凉，建议长袖打底，外搭夹克、风衣或薄毛衣。"
    if temperature >= 5:
        return "天气较冷，建议毛衣加外套，必要时加围巾。"
    return "天气很冷，建议厚外套、保暖内层，并做好防风保暖。"


tools = [get_weather, recommend_clothes]
tools_map = {tool_.name: tool_ for tool_ in tools}


# ========== 2️⃣ 初始化 LLM ==========
llm = ChatOpenAI(
    model="qwen3.5:cloud",
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

planner_llm = llm
executor_llm = llm.bind_tools(tools)
replanner_llm = llm


# ========== 3️⃣ 定义 State ==========
class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    current_step: str
    last_result: str
    past_steps: list[str]
    final_answer: str
    planner_raw: str
    replanner_raw: str
    round_count: int


MAX_REPLAN_ROUNDS = 6

PLANNER_SYSTEM_PROMPT = """
你是 Planner（规划者）。
你接到用户目标后，不执行任何工具，只负责拆解任务，输出一个清晰的 Task List。

要求：
1. 只输出待执行步骤，不要自己开始回答问题。
2. 步骤要具体、可执行，尽量让 Executor 一次只做一件事。
3. 优先拆成 2-4 步，避免过粗或过细。
4. 输出格式必须是项目符号列表，每行以 "- " 开头。
""".strip()

EXECUTOR_SYSTEM_PROMPT = """
你是 Executor（执行者）。
你一次只执行当前这一个步骤，不负责规划后续。

要求：
1. 如果需要外部信息，就调用工具。
2. 如果信息已经足够，就给出“当前步骤的执行结果”。
3. 不要擅自改计划，不要总结全部任务，只处理当前步骤。
4. 输出应尽量基于工具结果，不要编造。
""".strip()

REPLANNER_SYSTEM_PROMPT = """
你是 Replanner / Reviewer（复盘者 / 重新规划者）。
你的职责是根据刚执行完的结果，更新剩余计划。

请遵守：
1. 已完成的步骤要从计划中划掉。
2. 如果执行结果暴露出问题或信息不足，要修改剩余步骤。
3. 如果整体目标已经完成，直接结束。
4. 你不能替 Executor 执行尚未完成的步骤，也不要提前生成本该由后续步骤产出的内容。
5. 只要还有合理的未完成步骤，一般都应该输出 CONTINUE。
6. 只有当剩余计划为空，或刚完成的结果已经明确覆盖全部剩余步骤时，才能输出 FINISH。

输出格式必须严格如下：
ACTION: CONTINUE 或 FINISH
REASON: 简短说明
UPDATED_PLAN:
- 步骤A
- 步骤B
FINAL_ANSWER:
如果还不能结束，这一段写 N/A。
""".strip()


def _parse_bullet_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^[-*]\s+", line):
            items.append(re.sub(r"^[-*]\s+", "", line).strip())
        elif re.match(r"^\d+\.\s+", line):
            items.append(re.sub(r"^\d+\.\s+", "", line).strip())
    return [item for item in items if item]


def _extract_temperature(text: str) -> float | None:
    match = re.search(r"温度[:：]\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    return None


def _extract_section(text: str, title: str, next_titles: list[str]) -> str:
    pattern = rf"{title}:\s*(.*)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""

    section = match.group(1)
    end_positions = []
    for next_title in next_titles:
        next_match = re.search(rf"\n{next_title}:", section, flags=re.DOTALL)
        if next_match:
            end_positions.append(next_match.start())

    if end_positions:
        section = section[: min(end_positions)]
    return section.strip()


def _clean_remaining_plan(items: list[str], completed_step: str) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        normalized = item.strip()
        if re.fullmatch(r"~~.*~~", normalized):
            continue

        normalized = normalized.replace("~~", "").strip()
        normalized = re.sub(r"^(已完成[:：]\s*)", "", normalized).strip()

        if not normalized:
            continue
        if normalized == completed_step:
            continue

        cleaned.append(normalized)
    return cleaned


def planner_node(state: PlanExecuteState) -> PlanExecuteState:
    response = planner_llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"用户目标：{state['input']}"),
        ]
    )
    planner_raw = response.content if isinstance(response.content, str) else str(response.content)
    plan = _parse_bullet_list(planner_raw)
    if not plan:
        plan = [planner_raw.strip()]

    return {
        "plan": plan,
        "current_step": plan[0],
        "planner_raw": planner_raw,
        "past_steps": [],
        "last_result": "",
        "final_answer": "",
        "replanner_raw": "",
        "round_count": 0,
    }


def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    current_step = state["plan"][0]
    completed = "\n".join(f"- {item}" for item in state["past_steps"]) or "暂无"

    local_messages = [
        SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"原始目标：{state['input']}\n"
                f"完整计划：\n" + "\n".join(f"- {item}" for item in state["plan"]) + "\n\n"
                f"已完成步骤：\n{completed}\n\n"
                f"当前只执行这一步：{current_step}\n"
                "请专注完成当前步骤。如果需要工具就调用；如果不需要，就直接给出该步骤结果。"
            )
        ),
    ]

    while True:
        response = executor_llm.invoke(local_messages)
        local_messages.append(response)

        if not response.tool_calls:
            step_result = response.content if isinstance(response.content, str) else str(response.content)
            return {
                "current_step": current_step,
                "last_result": step_result,
            }

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "recommend_clothes" and "temperature" not in tool_args:
                weather_text = " ".join(
                    msg.content for msg in local_messages if isinstance(msg, ToolMessage)
                )
                temperature = _extract_temperature(weather_text)
                if temperature is not None:
                    tool_args["temperature"] = temperature

            try:
                result = tools_map[tool_name].invoke(tool_args)
            except Exception as exc:
                result = f"工具 {tool_name} 执行失败: {exc}"

            local_messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )


def replanner_node(state: PlanExecuteState) -> PlanExecuteState:
    current_step = state["plan"][0]
    remaining_plan = state["plan"]
    fallback_remaining_plan = remaining_plan[1:]
    completed_steps = state["past_steps"] + [f"{current_step} -> {state['last_result']}"]

    response = replanner_llm.invoke(
        [
            SystemMessage(content=REPLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"原始目标：{state['input']}\n\n"
                    f"执行前的剩余计划：\n"
                    + "\n".join(f"- {item}" for item in remaining_plan)
                    + "\n\n"
                    f"刚完成的步骤：{current_step}\n"
                    f"该步骤执行结果：{state['last_result']}\n\n"
                    f"已完成记录：\n"
                    + "\n".join(f"- {item}" for item in completed_steps)
                )
            ),
        ]
    )

    replanner_raw = response.content if isinstance(response.content, str) else str(response.content)
    action_match = re.search(r"ACTION:\s*(CONTINUE|FINISH)", replanner_raw, flags=re.IGNORECASE)
    action = action_match.group(1).upper() if action_match else "CONTINUE"

    updated_plan_text = _extract_section(
        replanner_raw,
        "UPDATED_PLAN",
        ["FINAL_ANSWER"],
    )
    updated_plan = _clean_remaining_plan(
        _parse_bullet_list(updated_plan_text),
        current_step,
    )

    final_answer = _extract_section(replanner_raw, "FINAL_ANSWER", [])
    if final_answer.upper() == "N/A":
        final_answer = ""

    if fallback_remaining_plan and action == "FINISH":
        action = "CONTINUE"
        final_answer = ""

    if action == "CONTINUE" and not updated_plan:
        updated_plan = fallback_remaining_plan

    return {
        "past_steps": completed_steps,
        "plan": updated_plan,
        "current_step": updated_plan[0] if updated_plan else "",
        "final_answer": final_answer if action == "FINISH" else "",
        "replanner_raw": replanner_raw,
        "round_count": state["round_count"] + 1,
    }


def route_after_replanner(state: PlanExecuteState) -> str:
    if state["final_answer"]:
        return END
    if not state["plan"]:
        return END
    if state["round_count"] >= MAX_REPLAN_ROUNDS:
        return END
    return "executor"


# ========== 4️⃣ 构建 LangGraph ==========
workflow = StateGraph(PlanExecuteState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("replanner", replanner_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")
workflow.add_conditional_edges("replanner", route_after_replanner)

app = workflow.compile()


# ========== 5️⃣ 运行并展示 Plan & Execute ==========
inputs: PlanExecuteState = {
    "input": "请帮我查一下郑州当前天气，并给出穿衣建议，最后总结今天是否适合出门。",
    "plan": [],
    "current_step": "",
    "last_result": "",
    "past_steps": [],
    "final_answer": "",
    "planner_raw": "",
    "replanner_raw": "",
    "round_count": 0,
}

print("\n🚀 开始启动 Plan & Execute 循环...\n")

latest_plan: list[str] = []
execution_log: list[str] = []
final_answer = ""

for step in app.stream(inputs, stream_mode="updates"):
    for node, state in step.items():
        print(f"================== 🟡 当前节点: {node.upper()} ==================")

        if node == "planner":
            latest_plan = state["plan"]
            print("[Planner (规划者)]")
            print("🧭 规划出的任务清单：")
            for index, item in enumerate(state["plan"], start=1):
                print(f"   Step {index}: {item}")

        elif node == "executor":
            print("[Executor (执行者)]")
            print(f"🎯 当前执行步骤: {state['current_step']}")
            print("🛠️ 执行结果：")
            print(state["last_result"])

        elif node == "replanner":
            latest_plan = state["plan"]
            execution_log = state["past_steps"]
            final_answer = state["final_answer"]

            print("[Replanner / Reviewer (复盘者 / 重新规划者)]")
            print(state["replanner_raw"])

            if state["final_answer"]:
                print("✅ 复盘者判断：整体目标已经完成。")
            elif state["plan"]:
                print("🔄 更新后的剩余计划：")
                for index, item in enumerate(state["plan"], start=1):
                    print(f"   Next {index}: {item}")
            else:
                print("⚠️ 计划已空，流程即将结束。")

        print("\n")

print("📌 已完成步骤记录：")
for item in execution_log:
    print(f"- {item}")

if final_answer:
    print("\n📌 最终答案：")
    print(final_answer)
else:
    print("\n📌 未生成显式 final answer，说明流程在计划耗尽或轮次上限处停止。")

print("✅ Plan & Execute 演示结束！")
