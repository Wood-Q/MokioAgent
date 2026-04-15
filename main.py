from __future__ import annotations

import argparse
import sys

from src.llm.llm import decide_tool_call
from src.tools.registry import execute_tool_call, tools_for_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claw",
        description="Single-step tool caller demo for Agent fundamentals.",
    )
    parser.add_argument("message", help="Natural language request")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    decision = decide_tool_call(args.message, tools_for_prompt(), model=args.model)

    print("=== Model ToolCall (raw) ===")
    print(decision.raw)

    if not decision.need_tool:
        print("\n=== Assistant Response ===")
        print(decision.response)
        return 0

    print("\n=== Execute Tool ===")
    print(f"tool={decision.tool}")
    print(f"arguments={decision.arguments}")

    try:
        result = execute_tool_call(decision.tool or "", decision.arguments or {})
    except Exception as exc:
        print("\n=== Tool Error ===")
        print(str(exc))
        return 1

    print("\n=== Tool Result ===")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
