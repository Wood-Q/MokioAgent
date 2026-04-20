from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    """Shared runtime context for future multi-step loop evolution."""

    user_input: str
    model: str = "gpt-4o-mini"
    metadata: dict[str, Any] = field(default_factory=dict)

