from __future__ import annotations

from langchain.agents import AgentState


class MokioclawState(AgentState, total=False):
    """Short-term state used during a single ReAct-style agent run."""

    user_input: str
    short_term_memory: list[str]

