from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import StructuredTool

from mokioclaw.tools.file_tools import file_tree, move_file


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

FILE_TREE_TOOL = ToolDef(
    name="file_tree",
    description="Return a plain-text file tree for a file or directory path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root file or directory path to inspect.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth to expand. Defaults to 3.",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files and directories.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    handler=file_tree,
)

TOOLS: dict[str, ToolDef] = {
    MOVE_FILE_TOOL.name: MOVE_FILE_TOOL,
    FILE_TREE_TOOL.name: FILE_TREE_TOOL,
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


def tools_for_agent() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=tool.handler,
            name=tool.name,
            description=tool.description,
        )
        for tool in TOOLS.values()
    ]
