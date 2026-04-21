from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from mokioclaw.tools.file_tools import file_tree, move_file
from mokioclaw.tools.workspace_tools import bash, file_edit, file_write


def _structured_tool(func, *, name: str, description: str) -> StructuredTool:
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
    )


TOOLS: tuple[BaseTool, ...] = (
    _structured_tool(
        move_file,
        name="move_file",
        description="Move a file from src path to dst path.",
    ),
    _structured_tool(
        file_tree,
        name="file_tree",
        description="Return a plain-text file tree for a file or directory path.",
    ),
    file_edit,
    file_write,
    bash,
)


PROMPT_TOOLS: tuple[dict[str, object], ...] = (
    {
        "name": "move_file",
        "description": "Move a file from src path to dst path.",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Source file path."},
                "dst": {"type": "string", "description": "Destination file path."},
            },
            "required": ["src", "dst"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_tree",
        "description": "Return a plain-text file tree for a file or directory path.",
        "parameters": {
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
    },
    {
        "name": "file_edit",
        "description": (
            "Safely edit an existing text file after it has been read in the "
            "current run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Existing file to modify."},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace in the current file content.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text to write back.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Whether to replace all matches instead of one.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
    {
        "name": "file_write",
        "description": (
            "Create a new text file or fully overwrite an existing text file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {
                    "type": "string",
                    "description": "Complete file content to write.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Set true to replace an existing file in full.",
                    "default": False,
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bash",
        "description": (
            "Execute a workspace-scoped search/read/list shell command in the "
            "foreground. Allowed entry commands: find, grep, rg, ag, ack, locate, "
            "cat, head, tail, less, more, ls, tree, du."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Single shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Workspace-relative working directory.",
                    "default": ".",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum execution time before aborting.",
                    "default": 20,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
)


def tools_for_prompt() -> list[dict[str, object]]:
    return list(PROMPT_TOOLS)


def tools_for_agent() -> list[BaseTool]:
    return list(TOOLS)
