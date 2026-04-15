from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.tools.file_tools import move_file


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]


MOVE_FILE_TOOL = ToolDef(
    name="move_file",
    description="Move a file from src path to dst path.",
    parameters={
        "type": "object",
        "properties": {
            "src": {"type": "string", "description": "Source file path."},
            "dst": {"type": "string", "description": "Destination file path."},
        },
        "required": ["src", "dst"],
        "additionalProperties": False,
    },
    handler=move_file,
)

TOOLS: dict[str, ToolDef] = {
    MOVE_FILE_TOOL.name: MOVE_FILE_TOOL,
}


def tools_for_prompt() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        for tool in TOOLS.values()
    ]


def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool = TOOLS[tool_name]
    return tool.handler(**arguments)
