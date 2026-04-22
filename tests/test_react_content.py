from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mokioclaw.core.memory import (
    build_initial_state,
    build_short_term_memory,
    coerce_todo_snapshots,
    collect_tool_executions,
    extract_final_response,
    render_message_trace,
    render_notepad,
    render_todo_panel,
)
from mokioclaw.prompts.react_prompt import (
    build_compact_system_prompt,
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
    assert "Todo" in prompt
    assert "NotePad" in prompt
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
    assert '"needs_clarification"' in prompt
    assert '"clarification_question"' in prompt
    assert '"missing_information"' in prompt
    assert '"name": "move_file"' in prompt


def test_build_compact_system_prompt_includes_focus_and_state():
    prompt = build_compact_system_prompt(
        plan=["读取目录", "整理文件"],
        completed_steps=["读取目录"],
        todos=[{"content": "整理文件", "status": "in_progress"}],
        notepad=["archive 已确认包含面试资料"],
        verification_nudge="Consider adding a verification step.",
        focus="重点保留测试结果和代码修改",
    )

    assert "Compactor" in prompt
    assert "重点保留测试结果和代码修改" in prompt
    assert "读取目录" in prompt
    assert "整理文件" in prompt
    assert "archive 已确认包含面试资料" in prompt


def test_build_initial_state_seeds_memory():
    state = build_initial_state("帮我移动文件")

    assert state["user_input"] == "帮我移动文件"
    assert state["short_term_memory"] == ["User request: 帮我移动文件"]
    assert state["todos"] == []
    assert state["todo_snapshot"] == []
    assert state["notepad"] == []
    assert state["clarification_attempts"] == 0
    assert state["last_clarification_signature"] == ""
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


def test_todo_and_notepad_render_helpers():
    todos = coerce_todo_snapshots(
        [
            {"content": "读取目录", "status": "completed"},
            {"content": "整理文件", "status": "in_progress"},
        ]
    )
    notes = ["记录第一条发现", "记录第二条发现"]

    assert "[x] 读取目录" in render_todo_panel(todos)
    assert "[-] 整理文件" in render_todo_panel(todos)
    assert "- 记录第一条发现" in render_notepad(notes)
