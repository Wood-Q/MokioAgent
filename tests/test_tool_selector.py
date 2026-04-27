from __future__ import annotations

from langchain_core.messages import HumanMessage

from mokioclaw.core.state import MokioclawState
from mokioclaw.tools.selector import (
    select_agent_tools_for_executor,
    select_prompt_tools_for_executor,
    select_prompt_tools_for_planner,
)


def _tool_names(tools: list[dict[str, object]]) -> list[str]:
    return [str(tool["name"]) for tool in tools]


def _state(user_input: str) -> MokioclawState:
    return MokioclawState(
        messages=[HumanMessage(content=user_input)],
        user_input=user_input,
    )


def test_planner_exposes_only_matching_file_tools():
    tools = select_prompt_tools_for_planner("把 demo/a.txt 移动到 archive/a.txt")

    assert _tool_names(tools) == ["move_file"]


def test_planner_does_not_expose_tools_for_plain_chat():
    assert select_prompt_tools_for_planner("你好，介绍一下你自己") == []


def test_executor_always_gets_session_tools():
    tools = select_prompt_tools_for_executor(
        _state("帮我整理 demo 目录"),
        "总结当前步骤",
    )

    assert _tool_names(tools) == ["todo_write", "notepad_write"]


def test_executor_matches_current_step_before_user_input():
    tools = select_prompt_tools_for_executor(
        _state("帮我整理 demo 目录"),
        "读取 demo 和 archive 文件内容并记录发现",
    )

    assert _tool_names(tools) == ["todo_write", "notepad_write", "bash"]


def test_executor_falls_back_to_user_input_when_step_has_no_match():
    tools = select_prompt_tools_for_executor(
        _state("把 demo/a.txt 移动到 archive/a.txt"),
        "执行当前动作",
    )

    assert _tool_names(tools) == ["todo_write", "notepad_write", "move_file"]


def test_executor_agent_tools_match_prompt_tools():
    state = _state("生成 demo/summary.md 总结报告")
    prompt_tool_names = _tool_names(
        select_prompt_tools_for_executor(state, "创建 summary.md 总结文件")
    )
    agent_tools = select_agent_tools_for_executor(
        state,
        "创建 summary.md 总结文件",
    )
    agent_tool_names = [tool.name for tool in agent_tools]

    assert agent_tool_names == prompt_tool_names
    assert agent_tool_names == ["todo_write", "notepad_write", "file_write"]
