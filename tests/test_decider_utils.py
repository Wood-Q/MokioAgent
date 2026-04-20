from __future__ import annotations

from mokioclaw.prompts.toolcall_prompt import build_toolcall_system_prompt
from mokioclaw.utils.json_utils import extract_json


def test_extract_json_parses_wrapped_json():
    raw = 'prefix {"need_tool": false, "response": "ok"} suffix'
    payload = extract_json(raw)
    assert payload["need_tool"] is False
    assert payload["response"] == "ok"


def test_build_toolcall_system_prompt_includes_tools_json():
    prompt = build_toolcall_system_prompt(
        [
            {
                "name": "move_file",
                "description": "Move a file.",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert "你是一个工具调用决策器。" in prompt
    assert '"name": "move_file"' in prompt
