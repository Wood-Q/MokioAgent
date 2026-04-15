from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI


DEFAULT_PROMPT = """你是一个工具调用决策器。\n\
你的任务：根据用户输入，决定是否调用工具。\n\
你必须只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。\n\
输出格式（仅二选一）：\n\
1) 需要工具时：\n\
{\n\
  \"need_tool\": true,\n\
  \"tool\": \"工具名\",\n\
  \"arguments\": { ... }\n\
}\n\
2) 不需要工具时：\n\
{\n\
  \"need_tool\": false,\n\
  \"response\": \"直接回复用户的话\"\n\
}\n\
严格要求：\n\
- 不要返回任何非 JSON 字符。\n\
- 路径参数必须是字符串。\n\
- 只允许使用我提供的工具。\n\
可用工具：\n\
{tools_json}\n\
"""


@dataclass(frozen=True)
class ToolDecision:
    need_tool: bool
    raw: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    response: str | None = None


def _load_prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "toolcall.txt"
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_PROMPT


def _extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError(f"Model output is not valid JSON: {raw_text}")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model output is not valid JSON: {raw_text}") from exc


def decide_tool_call(
    user_input: str, tools: list[dict[str, Any]], model: str = "gpt-4o-mini"
) -> ToolDecision:
    prompt_template = _load_prompt_template()
    prompt = prompt_template.format(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2)
    )

    llm = ChatOpenAI(model=model, temperature=0)
    message = llm.invoke(
        [
            ("system", prompt),
            ("user", user_input),
        ]
    )

    if isinstance(message.content, str):
        raw = message.content
    elif isinstance(message.content, list):
        raw = "\n".join(str(item) for item in message.content)
    else:
        raw = str(message.content)
    payload = _extract_json(raw)

    need_tool = bool(payload.get("need_tool"))
    if need_tool:
        tool_name = payload.get("tool")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Invalid tool name from model output: {raw}")
        if not isinstance(arguments, dict):
            raise ValueError(f"Invalid arguments from model output: {raw}")
        return ToolDecision(
            need_tool=True, raw=raw, tool=tool_name, arguments=arguments
        )

    response = payload.get("response")
    if not isinstance(response, str):
        response = "我不需要调用工具来完成这个请求。"
    return ToolDecision(need_tool=False, raw=raw, response=response)
