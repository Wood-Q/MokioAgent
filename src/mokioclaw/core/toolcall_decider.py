from __future__ import annotations

from typing import Any

from mokioclaw.core.types import ToolDecision
from mokioclaw.prompts.toolcall_prompt import build_toolcall_system_prompt
from mokioclaw.providers.ollama_provider import invoke_chat
from mokioclaw.utils.json_utils import extract_json


def decide_tool_call(
    user_input: str, tools: list[dict[str, Any]], model: str = "gpt-4o-mini"
) -> ToolDecision:
    system_prompt = build_toolcall_system_prompt(tools)
    raw = invoke_chat(
        messages=[
            ("system", system_prompt),
            ("user", user_input),
        ],
        model=model,
    )

    payload = extract_json(raw)
    need_tool = bool(payload.get("need_tool"))

    if need_tool:
        tool_name = payload.get("tool")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Invalid tool name from model output: {raw}")
        if not isinstance(arguments, dict):
            raise ValueError(f"Invalid arguments from model output: {raw}")
        return ToolDecision(
            need_tool=True,
            raw=raw,
            tool=tool_name,
            arguments=arguments,
        )

    response = payload.get("response")
    if not isinstance(response, str):
        response = "我不需要调用工具来完成这个请求。"
    return ToolDecision(need_tool=False, raw=raw, response=response)
