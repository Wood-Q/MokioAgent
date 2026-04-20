from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any] | None = None
    result: str | None = None


@dataclass(frozen=True)
class LoopOutcome:
    need_tool: bool
    raw: str
    response: str | None = None
    tool_calls: list[ToolExecution] | None = None
    memory: list[str] | None = None
    tool_error: str | None = None
