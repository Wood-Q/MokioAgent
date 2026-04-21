from __future__ import annotations

from typing import Annotated, NotRequired, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class FileSnapshot(TypedDict):
    sha256: str
    mtime_ns: int
    size: int
    source: str


def merge_text_lists(
    left: list[str] | None,
    right: list[str] | None,
) -> list[str]:
    return [*(left or []), *(right or [])]


def merge_file_snapshots(
    left: dict[str, FileSnapshot] | None,
    right: dict[str, FileSnapshot] | None,
) -> dict[str, FileSnapshot]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class MokioclawState(TypedDict, total=False):
    """State used during a single Plan & Execute agent run."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    user_input: NotRequired[str]
    short_term_memory: NotRequired[list[str]]
    file_snapshots: NotRequired[
        Annotated[dict[str, FileSnapshot], merge_file_snapshots]
    ]
    plan: NotRequired[list[str]]
    completed_steps: NotRequired[list[str]]
    current_step_index: NotRequired[int]
    final_response: NotRequired[str]
    turn_events: NotRequired[Annotated[list[str], merge_text_lists]]
