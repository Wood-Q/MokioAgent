from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    model = os.getenv("MODEL", "qwen3.6-flash")

    client = OpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "北京天气怎么样？"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的天气。",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市英文名，例如 Beijing、Shanghai。",
                            }
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        temperature=0,
    )

    print("=== API 原始返回值 ===")
    print(response.model_dump_json(indent=2))

    print("\n=== 单独取出模型生成的 tool call ===")
    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    for tool_call in tool_calls:
        print(f"id        = {tool_call.id}")
        print(f"type      = {tool_call.type}")
        print(f"name      = {tool_call.function.name}")
        print(f"arguments = {tool_call.function.arguments}")

        print("\n=== arguments 解析成 Python dict 后 ===")
        print(
            json.dumps(
                json.loads(tool_call.function.arguments),
                ensure_ascii=False,
                indent=2,
            )
        )

    if not tool_calls:
        print("没有拿到 tool_calls。请确认当前模型支持 Chat Completions tool calling。")


if __name__ == "__main__":
    main()
