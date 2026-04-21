import re
from typing import Any

import requests
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict


# This file demonstrates two classic multi-agent orchestration patterns
# implemented with vanilla LangGraph primitives:
# 1. Supervisor: a central router decides which specialist should work next.
# 2. Swarm: each agent hands the task off to the next agent by itself.


# ========== 1) Tools ==========
CITY_COORDS = {
    "郑州": (34.75, 113.62),
    "北京": (39.90, 116.40),
    "上海": (31.23, 121.47),
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    lat, lon = CITY_COORDS.get(city, CITY_COORDS["郑州"])
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    res = requests.get(url, timeout=15).json()
    weather = res["current_weather"]
    return (
        f"{city} current weather -> "
        f"temperature: {weather['temperature']}C, "
        f"windspeed: {weather['windspeed']}km/h"
    )


@tool
def recommend_clothes(temperature: float) -> str:
    """Recommend clothing based on temperature in Celsius."""
    if temperature >= 28:
        return "天气较热，建议短袖、薄裤，注意防晒和补水。"
    if temperature >= 20:
        return "天气温和，建议长袖或薄外套，穿着轻便即可。"
    if temperature >= 12:
        return "天气偏凉，建议长袖打底，外搭夹克、风衣或薄毛衣。"
    if temperature >= 5:
        return "天气较冷，建议毛衣加外套，必要时加围巾。"
    return "天气很冷，建议厚外套、保暖内层，并做好防风保暖。"


@tool
def evaluate_outing(temperature: float, windspeed: float) -> str:
    """Evaluate whether it is suitable to go outside."""
    if temperature < 0:
        return "当前较冷，不算理想出门天气，若外出请充分保暖。"
    if windspeed >= 25:
        return "风速偏大，外出体验一般，建议谨慎安排长时间户外活动。"
    if temperature <= 35:
        return "整体适合出门，体感预计较为舒适。"
    return "天气偏热，外出时请注意防晒和补水。"


TOOLS = [get_weather, recommend_clothes, evaluate_outing]
TOOLS_MAP = {tool_.name: tool_ for tool_ in TOOLS}


# ========== 2) LLM ==========
llm = ChatOpenAI(
    model="qwen3.5:cloud",
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)


# ========== 3) Shared helpers ==========
class MultiAgentState(TypedDict):
    task: str
    city: str
    weather_report: str
    temperature: float | None
    windspeed: float | None
    clothing_advice: str
    outing_advice: str
    final_answer: str
    next_agent: str
    decision_reason: str
    handoff_log: list[str]
    last_agent: str
    last_output: str
    last_tool_trace: list[str]


def _extract_temperature(text: str) -> float | None:
    match = re.search(r"temperature:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_windspeed(text: str) -> float | None:
    match = re.search(r"windspeed:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _resolve_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
    state: MultiAgentState,
    tool_trace: list[str],
) -> dict[str, Any]:
    args = dict(tool_args)
    context_text = "\n".join([state["weather_report"], state["last_output"], *tool_trace])

    if tool_name == "recommend_clothes" and "temperature" not in args:
        temperature = state["temperature"]
        if temperature is None:
            temperature = _extract_temperature(context_text)
        if temperature is not None:
            args["temperature"] = temperature

    if tool_name == "evaluate_outing":
        temperature = args.get("temperature", state["temperature"])
        windspeed = args.get("windspeed", state["windspeed"])

        if temperature is None:
            temperature = _extract_temperature(context_text)
        if windspeed is None:
            windspeed = _extract_windspeed(context_text)

        if temperature is not None:
            args["temperature"] = temperature
        if windspeed is not None:
            args["windspeed"] = windspeed

    return args


def _run_tool_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[Any],
    state: MultiAgentState,
) -> tuple[str, list[str]]:
    runnable = llm.bind_tools(tools)
    local_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    tool_trace: list[str] = []

    while True:
        response = runnable.invoke(local_messages)
        local_messages.append(response)

        if not response.tool_calls:
            final_text = response.content if isinstance(response.content, str) else str(response.content)
            return final_text, tool_trace

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = _resolve_tool_args(
                tool_name,
                tool_call["args"],
                state,
                tool_trace,
            )
            try:
                result = TOOLS_MAP[tool_name].invoke(tool_args)
            except Exception as exc:
                result = f"Tool {tool_name} failed: {exc}"

            tool_trace.append(f"[{tool_name}] args={tool_args} -> {result}")
            local_messages.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                    content=str(result),
                )
            )


def _parse_weather_fields(weather_report: str, tool_trace: list[str]) -> tuple[float | None, float | None]:
    combined = "\n".join([weather_report, *tool_trace])
    return _extract_temperature(combined), _extract_windspeed(combined)


def _base_state(task: str, city: str) -> MultiAgentState:
    return {
        "task": task,
        "city": city,
        "weather_report": "",
        "temperature": None,
        "windspeed": None,
        "clothing_advice": "",
        "outing_advice": "",
        "final_answer": "",
        "next_agent": "",
        "decision_reason": "",
        "handoff_log": [],
        "last_agent": "",
        "last_output": "",
        "last_tool_trace": [],
    }


# ========== 4) Specialist prompts ==========
WEATHER_PROMPT = """
You are the Weather Analyst agent.
Your only job is to fetch and summarize current weather facts for the user's city.
If facts are missing, call tools. Do not give clothing advice or final recommendations.
""".strip()

OUTFIT_PROMPT = """
You are the Outfit Stylist agent.
Your only job is to produce clothing advice based on the known weather facts.
Use tools if useful. Do not repeat full weather analysis and do not produce the final answer.
""".strip()

SUMMARY_PROMPT = """
You are the Trip Summary agent.
You combine the weather facts, clothing advice, and outing evaluation into one final answer.
Use tools if helpful. Keep the answer practical and concise.
""".strip()


# ========== 5) Supervisor pattern ==========
def supervisor_node(state: MultiAgentState) -> Command:
    if not state["weather_report"]:
        next_agent = "weather_agent"
        reason = "缺少天气事实，先让 Weather Analyst 查询实时天气。"
    elif not state["clothing_advice"]:
        next_agent = "outfit_agent"
        reason = "已经有天气数据，下一步交给 Outfit Stylist 生成穿衣建议。"
    elif not state["final_answer"]:
        next_agent = "summary_agent"
        reason = "素材已齐，交给 Trip Summary agent 做最终整合。"
    else:
        reason = "所有关键产物都已生成，Supervisor 结束流程。"
        return Command(
            update={
                "next_agent": "",
                "last_agent": "supervisor",
                "decision_reason": reason,
                "handoff_log": state["handoff_log"] + [f"supervisor -> {END} | {reason}"],
            },
            goto=END,
        )

    return Command(
        update={
            "next_agent": next_agent,
            "decision_reason": reason,
            "last_agent": "supervisor",
            "handoff_log": state["handoff_log"] + [f"supervisor -> {next_agent} | {reason}"],
        },
        goto=next_agent,
    )


def weather_agent_node(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=WEATHER_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"城市：{state['city']}\n"
            "请只产出天气事实，不要扩展到穿衣建议。"
        ),
        tools=[get_weather],
        state=state,
    )
    temperature, windspeed = _parse_weather_fields(content, tool_trace)

    return Command(
        update={
            "weather_report": content,
            "temperature": temperature,
            "windspeed": windspeed,
            "last_agent": "weather_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
        },
        goto="supervisor",
    )


def outfit_agent_node(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=OUTFIT_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"天气事实：{state['weather_report']}\n"
            f"temperature={state['temperature']}, windspeed={state['windspeed']}\n"
            "请只给穿衣建议。"
        ),
        tools=[recommend_clothes],
        state=state,
    )

    return Command(
        update={
            "clothing_advice": content,
            "last_agent": "outfit_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
        },
        goto="supervisor",
    )


def summary_agent_node(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=SUMMARY_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"城市：{state['city']}\n"
            f"天气事实：{state['weather_report']}\n"
            f"穿衣建议：{state['clothing_advice']}\n"
            f"temperature={state['temperature']}, windspeed={state['windspeed']}\n"
            "请给出完整最终答复，并明确今天是否适合出门。"
        ),
        tools=[evaluate_outing],
        state=state,
    )

    outing_advice = ""
    for item in tool_trace:
        if item.startswith("[evaluate_outing]"):
            outing_advice = item
            break

    return Command(
        update={
            "outing_advice": outing_advice,
            "final_answer": content,
            "last_agent": "summary_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
        },
        goto="supervisor",
    )


supervisor_workflow = StateGraph(MultiAgentState)
supervisor_workflow.add_node("supervisor", supervisor_node)
supervisor_workflow.add_node("weather_agent", weather_agent_node)
supervisor_workflow.add_node("outfit_agent", outfit_agent_node)
supervisor_workflow.add_node("summary_agent", summary_agent_node)
supervisor_workflow.add_edge(START, "supervisor")
supervisor_app = supervisor_workflow.compile()


# ========== 6) Swarm pattern ==========
def swarm_weather_agent(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=WEATHER_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"城市：{state['city']}\n"
            "你完成天气分析后，请把任务交给 Outfit Stylist。"
        ),
        tools=[get_weather],
        state=state,
    )
    temperature, windspeed = _parse_weather_fields(content, tool_trace)
    reason = "Weather Analyst 已完成天气事实收集，handoff 给 Outfit Stylist。"

    return Command(
        update={
            "weather_report": content,
            "temperature": temperature,
            "windspeed": windspeed,
            "last_agent": "swarm_weather_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
            "decision_reason": reason,
            "handoff_log": state["handoff_log"] + [f"swarm_weather_agent -> swarm_outfit_agent | {reason}"],
        },
        goto="swarm_outfit_agent",
    )


def swarm_outfit_agent(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=OUTFIT_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"天气事实：{state['weather_report']}\n"
            f"temperature={state['temperature']}, windspeed={state['windspeed']}\n"
            "你完成穿衣建议后，请把任务交给 Trip Summary agent。"
        ),
        tools=[recommend_clothes],
        state=state,
    )
    reason = "Outfit Stylist 已完成穿衣建议，handoff 给 Trip Summary agent。"

    return Command(
        update={
            "clothing_advice": content,
            "last_agent": "swarm_outfit_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
            "decision_reason": reason,
            "handoff_log": state["handoff_log"] + [f"swarm_outfit_agent -> swarm_summary_agent | {reason}"],
        },
        goto="swarm_summary_agent",
    )


def swarm_summary_agent(state: MultiAgentState) -> Command:
    content, tool_trace = _run_tool_agent(
        system_prompt=SUMMARY_PROMPT,
        user_prompt=(
            f"用户任务：{state['task']}\n"
            f"城市：{state['city']}\n"
            f"天气事实：{state['weather_report']}\n"
            f"穿衣建议：{state['clothing_advice']}\n"
            f"temperature={state['temperature']}, windspeed={state['windspeed']}\n"
            "请生成最终结论并结束。"
        ),
        tools=[evaluate_outing],
        state=state,
    )
    reason = "Trip Summary agent 已整合完成最终答复，Swarm 流程结束。"

    return Command(
        update={
            "final_answer": content,
            "last_agent": "swarm_summary_agent",
            "last_output": content,
            "last_tool_trace": tool_trace,
            "decision_reason": reason,
            "handoff_log": state["handoff_log"] + [f"swarm_summary_agent -> {END} | {reason}"],
        },
        goto=END,
    )


swarm_workflow = StateGraph(MultiAgentState)
swarm_workflow.add_node("swarm_weather_agent", swarm_weather_agent)
swarm_workflow.add_node("swarm_outfit_agent", swarm_outfit_agent)
swarm_workflow.add_node("swarm_summary_agent", swarm_summary_agent)
swarm_workflow.add_edge(START, "swarm_weather_agent")
swarm_app = swarm_workflow.compile()


# ========== 7) Demo runners ==========
def run_supervisor_demo() -> None:
    print("\n===== Supervisor Demo =====\n")
    inputs = _base_state(
        task="请帮我查郑州天气，给出穿衣建议，并判断今天适不适合出门。",
        city="郑州",
    )

    for step in supervisor_app.stream(inputs, stream_mode="updates"):
        for node, state in step.items():
            print(f"---------------- {node.upper()} ----------------")

            if node == "supervisor":
                print("[Supervisor]")
                print(f"Decision: {state['decision_reason']}")
                if state.get("next_agent"):
                    print(f"Route To: {state['next_agent']}")

            elif node == "weather_agent":
                print("[Weather Analyst]")
                print(state["weather_report"])
                if state["last_tool_trace"]:
                    print("Tool Trace:")
                    for item in state["last_tool_trace"]:
                        print(f"  {item}")

            elif node == "outfit_agent":
                print("[Outfit Stylist]")
                print(state["clothing_advice"])
                if state["last_tool_trace"]:
                    print("Tool Trace:")
                    for item in state["last_tool_trace"]:
                        print(f"  {item}")

            elif node == "summary_agent":
                print("[Trip Summary Agent]")
                print(state["final_answer"])
                if state["last_tool_trace"]:
                    print("Tool Trace:")
                    for item in state["last_tool_trace"]:
                        print(f"  {item}")

            print()


def run_swarm_demo() -> None:
    print("\n===== Swarm Demo =====\n")
    inputs = _base_state(
        task="请帮我查郑州天气，给出穿衣建议，并判断今天适不适合出门。",
        city="郑州",
    )

    for step in swarm_app.stream(inputs, stream_mode="updates"):
        for node, state in step.items():
            print(f"---------------- {node.upper()} ----------------")

            if node == "swarm_weather_agent":
                print("[Swarm Weather Agent]")
                print(state["weather_report"])
                print(f"Handoff: {state['decision_reason']}")

            elif node == "swarm_outfit_agent":
                print("[Swarm Outfit Agent]")
                print(state["clothing_advice"])
                print(f"Handoff: {state['decision_reason']}")

            elif node == "swarm_summary_agent":
                print("[Swarm Summary Agent]")
                print(state["final_answer"])
                print(f"Handoff: {state['decision_reason']}")

            if state["last_tool_trace"]:
                print("Tool Trace:")
                for item in state["last_tool_trace"]:
                    print(f"  {item}")

            print()


if __name__ == "__main__":
    print("Running LangGraph multi-agent demos...")
    run_supervisor_demo()
    run_swarm_demo()
