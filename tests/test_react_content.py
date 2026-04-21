from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mokioclaw.core.memory import (
    build_initial_state,
    build_short_term_memory,
    collect_tool_executions,
    extract_final_response,
    render_message_trace,
)
from mokioclaw.prompts.react_prompt import (
    build_planner_system_prompt,
    build_react_system_prompt,
)


def test_build_executor_system_prompt_includes_tools_json():
    prompt = build_react_system_prompt(
        [
            {
                "name": "move_file",
                "description": "Move a file.",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert "Executor" in prompt
    assert "Plan & Execute" in prompt
    assert '"name": "move_file"' in prompt


def test_build_planner_system_prompt_includes_json_contract():
    prompt = build_planner_system_prompt(
        [
            {
                "name": "move_file",
                "description": "Move a file.",
                "parameters": {"type": "object"},
            }
        ]
    )
    assert "Planner" in prompt
    assert '"steps"' in prompt
    assert '"final_response"' in prompt
    assert '"name": "move_file"' in prompt


def test_build_initial_state_seeds_memory():
    state = build_initial_state("帮我移动文件")

    assert state["user_input"] == "帮我移动文件"
    assert state["short_term_memory"] == ["User request: 帮我移动文件"]
    assert len(state["messages"]) == 1


def test_memory_helpers_extract_trace_and_tool_steps():
    messages = [
        HumanMessage(content="Use the move_file tool"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "move_file",
                    "args": {"src": "demo/a.txt", "dst": "archive/a.txt"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Moved file from '/tmp/a.txt' to '/tmp/archive/a.txt'.",
            name="move_file",
            tool_call_id="call_1",
        ),
        AIMessage(content="文件已经移动完成。"),
    ]

    tool_executions = collect_tool_executions(messages)
    memory = build_short_term_memory("Use the move_file tool", messages)
    trace = render_message_trace(messages)
    response = extract_final_response(messages)

    assert len(tool_executions) == 1
    assert tool_executions[0].name == "move_file"
    assert tool_executions[0].arguments == {"src": "demo/a.txt", "dst": "archive/a.txt"}
    assert tool_executions[0].result is not None
    assert response == "文件已经移动完成。"
    assert any("Tool used: move_file" in item for item in memory)
    assert "Tool Call: move_file" in trace
    assert "Tool Result [move_file]" in trace
