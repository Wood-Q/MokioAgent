from __future__ import annotations

from typing import Annotated, NotRequired, Required, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class MokioclawState(TypedDict, total=False):
    """Short-term state used during a single ReAct-style agent run."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    user_input: NotRequired[str]
    short_term_memory: NotRequired[list[str]]
