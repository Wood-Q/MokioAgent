from typing import Annotated

import requests
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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
        f"{city}天气状态 -> 温度: {weather['temperature']}°C, "
        f"风速: {weather['windspeed']}km/h"
    )


# ========== 2️⃣ 初始化 LLM ==========
llm = ChatOpenAI(
    model="qwen3.5:cloud",
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

tools = [get_weather]
generator_llm = llm.bind_tools(tools)
critic_llm = llm


# ========== 3️⃣ 定义 Reflection State ==========
class ReflectionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    critic_feedback: str
    approved: bool
    reflection_round: int


MAX_REFLECTION_ROUNDS = 2

GENERATOR_SYSTEM_PROMPT = """
你是 ReAct 工作流中的 Generator（生成者）。
你的目标是先像普通 ReAct 一样思考：缺信息就调用工具，信息足够就给出答案。

要求：
1. 如果关键信息不足，请优先调用工具。
2. 如果已经具备足够信息，就直接给出清晰、自然的最终回答。
3. 如果你收到了 Critic 的反馈，请逐条修正，不要重复原答案中的问题。
4. 回答尽量基于工具结果，不要凭空编造。
""".strip()

CRITIC_SYSTEM_PROMPT = """
你是 Reflection 工作流中的 Critic（批评家 / 测试者）。
你不负责调用工具，你只负责检查 Generator 最新答案是否存在问题。

请重点检查：
1. 是否真正回答了用户问题。
2. 是否遗漏了工具结果里的关键信息。
3. 是否有主观臆测、幻觉或跳步推理。
4. 是否给出了可执行、可信的建议。

输出格式必须严格遵守：
DECISION: APPROVE 或 REVISE
FEEDBACK: 具体反馈

如果是第一次评审（round=0），即便答案基本正确，也尽量提出至少一个可改进点，
这样更容易演示 Reflection 的“批评 -> 修正”循环。
如果答案已经没有明显事实错误、遗漏或幻觉，只剩轻微措辞优化，请直接给出 APPROVE。
""".strip()


def _latest_ai_draft(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def generator_node(state: ReflectionState) -> ReflectionState:
    prompt_messages: list[BaseMessage] = [SystemMessage(content=GENERATOR_SYSTEM_PROMPT)]
    prompt_messages.extend(state["messages"])

    if state["critic_feedback"]:
        prompt_messages.append(
            HumanMessage(
                content=(
                    "下面是 Critic 对你上一版答案的反馈，请你据此修正并重写最终答案：\n"
                    f"{state['critic_feedback']}"
                )
            )
        )

    response = generator_llm.invoke(prompt_messages)
    return {
        "messages": [response],
        "critic_feedback": "",
    }


def critic_node(state: ReflectionState) -> ReflectionState:
    latest_draft = _latest_ai_draft(state["messages"])

    review_request = HumanMessage(
        content=(
            f"用户原始问题与上下文如下：\n{state['messages']}\n\n"
            f"当前评审轮次 round={state['reflection_round']}\n"
            f"请重点评审 Generator 最新答案：\n{latest_draft}"
        )
    )
    review = critic_llm.invoke(
        [SystemMessage(content=CRITIC_SYSTEM_PROMPT), review_request]
    )

    review_text = review.content if isinstance(review.content, str) else str(review.content)
    is_approved = "DECISION: APPROVE" in review_text.upper()

    return {
        "critic_feedback": review_text,
        "approved": is_approved,
        "reflection_round": state["reflection_round"] + 1,
    }


def route_after_generator(state: ReflectionState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "critic"


def route_after_critic(state: ReflectionState) -> str:
    if state["approved"] or state["reflection_round"] >= MAX_REFLECTION_ROUNDS:
        return END
    return "generator"


# ========== 4️⃣ 构建 Generator -> Tools -> Critic 的图 ==========
tool_node = ToolNode(tools)

workflow = StateGraph(ReflectionState)
workflow.add_node("generator", generator_node)
workflow.add_node("tools", tool_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "generator")
workflow.add_conditional_edges(
    "generator",
    route_after_generator,
    {
        "tools": "tools",
        "critic": "critic",
    },
)
workflow.add_edge("tools", "generator")
workflow.add_conditional_edges("critic", route_after_critic)

app = workflow.compile()


# ========== 5️⃣ 运行并展示 Reflection 循环 ==========
inputs: ReflectionState = {
    "messages": [
        ("user", "请帮我查一下郑州的天气怎么样？并且根据温度建议我穿什么。")
    ],
    "critic_feedback": "",
    "approved": False,
    "reflection_round": 0,
}

print("\n🚀 开始启动 ReAct + Reflection 循环...\n")

latest_answer = ""
final_approved = False

for step in app.stream(inputs, stream_mode="updates"):
    for node, state in step.items():
        print(f"================== 🟡 当前节点: {node.upper()} ==================")

        if node == "generator":
            msg = state["messages"][-1]
            print("[Generator (生成者)]")
            if msg.tool_calls:
                print("🧠 Generator 判定：信息不足，先调用工具补齐事实。")
                for tool_call in msg.tool_calls:
                    print(
                        f"👉 准备使用工具: [{tool_call['name']}], 参数: {tool_call['args']}"
                    )
            else:
                latest_answer = msg.content
                print("🧠 Generator 先给出一版草稿答案：")
                print(msg.content)

        elif node == "tools":
            msg = state["messages"][-1]
            print("[Tools (行动环节)]")
            print("🛠️ 工具执行完毕，返回结果：")
            print(f"   => {msg.content}")
            print("🔄 工具结果已回流给 Generator，准备进入下一轮推理。")

        elif node == "critic":
            final_approved = state["approved"]
            print("[Critic (批评家 / 测试者)]")
            print(state["critic_feedback"])
            if state["approved"]:
                print("✅ Critic 认为答案已经可靠，可以结束流程。")
            elif state["reflection_round"] >= MAX_REFLECTION_ROUNDS:
                print("⚠️ 已达到最大反思轮数，流程结束。")
            else:
                print("🔄 Critic 指出了问题，Generator 将继续修正答案。")

        print("\n")

print("📌 当前流程收敛出的最终答案：")
print(latest_answer)
print(f"\n📌 Critic 最终结论: {'APPROVED' if final_approved else 'NOT FULLY APPROVED'}")
print("✅ Reflection 演示结束！")
