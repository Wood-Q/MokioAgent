from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDecision:
    need_tool: bool
    raw: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    response: str | None = None


@dataclass(frozen=True)
class LoopOutcome:
    need_tool: bool
    raw: str
    response: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    tool_result: str | None = None
    tool_error: str | None = None

