from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.errors import GraphRecursionError

import mokioclaw.core.loop as loop
import mokioclaw.tools.workspace_tools as workspace_tools

ModelResponse = AIMessage | str | Callable[[list[object], int], AIMessage]


def _planner_payload(steps: list[str], final_response: str = "") -> str:
    return json.dumps(
        {
            "steps": steps,
            "final_response": final_response,
        },
        ensure_ascii=False,
    )


def _tool_call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


class StageModel:
    def __init__(self, *, stage: str, responses: list[ModelResponse]):
        self.stage = stage
        self.responses = responses
        self.calls = 0
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> StageModel:
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1

        assert isinstance(messages[0], SystemMessage)
        prompt = str(messages[0].content)
        if self.stage == "planner":
            assert "Planner" in prompt
        elif self.stage == "executor":
            assert "Executor" in prompt
            assert "Todo" in prompt
            assert "NotePad" in prompt
            assert self.bound_tools
        elif self.stage == "finalizer":
            assert "Finalizer" in prompt
        elif self.stage == "compactor":
            assert "Compactor" in prompt
        elif self.stage == "casual":
            assert "普通聊天" in prompt
        else:
            raise AssertionError(f"Unexpected stage: {self.stage}")

        if self.calls > len(self.responses):
            raise AssertionError(f"{self.stage} invoked more times than expected")

        response = self.responses[self.calls - 1]
        if isinstance(response, AIMessage):
            return response
        if isinstance(response, str):
            return AIMessage(content=response)
        return response(messages, self.calls)


def _build_chat_model_factory(*models: StageModel):
    remaining = list(models)

    def _build_chat_model(model: str) -> StageModel:
        assert model == "demo-model"
        if not remaining:
            raise AssertionError("build_chat_model called more times than expected")
        return remaining.pop(0)

    return _build_chat_model


def test_run_single_step_uses_plan_execute_todo_and_tool_loop(tmp_path, monkeypatch):
    source = tmp_path / "demo" / "a.txt"
    target = tmp_path / "archive" / "a.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("hello", encoding="utf-8")

    planner = StageModel(
        stage="planner",
        responses=[_planner_payload(["Move the file from source to target."])],
    )
    executor = StageModel(
        stage="executor",
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "todo_write",
                        {
                            "todos": [
                                {
                                    "content": "Move the file from source to target.",
                                    "status": "in_progress",
                                }
                            ]
                        },
                        "call_1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "move_file",
                        {"src": str(source), "dst": str(target)},
                        "call_2",
                    )
                ],
            ),
            AIMessage(content="步骤完成：文件已移动。"),
        ],
    )
    finalizer = StageModel(
        stage="finalizer",
        responses=[AIMessage(content="文件已经移动完成。")],
    )

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    outcome = loop.run_single_step(
        f"把 {source} 移动到 {target}",
        model="demo-model",
    )

    assert outcome.need_tool is True
    assert outcome.response == "文件已经移动完成。"
    assert outcome.tool_calls is not None
    assert any(tool_call.name == "todo_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "move_file" for tool_call in outcome.tool_calls)
    assert "Planner: generated execution plan" in outcome.raw
    assert "Completed Step 1/1" in outcome.raw
    assert "Todo Panel:" in outcome.raw
    assert target.exists()
    assert not source.exists()
    assert outcome.todos is not None
    assert outcome.todos[0].status == "completed"


def test_session_preserves_multi_turn_history(monkeypatch):
    def _session_reply(messages: list[object], _: int) -> AIMessage:
        human_message_count = sum(
            1 for message in messages if getattr(message, "type", None) == "human"
        )
        return AIMessage(content=f"目前会话里有 {human_message_count} 条用户消息。")

    planner = StageModel(
        stage="planner",
        responses=[_session_reply, _session_reply],
    )
    executor = StageModel(stage="executor", responses=[])
    finalizer = StageModel(stage="finalizer", responses=[])

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    session = loop.MokioclawSession(model="demo-model")
    first = session.run_turn("请简短回答当前会话里有几条用户消息。")
    second = session.run_turn("请再回答一次。")

    assert first.response == "目前会话里有 1 条用户消息。"
    assert second.response == "目前会话里有 2 条用户消息。"
    assert "Planner: returned a direct response without execution." in first.raw
    assert session.state is not None
    assert len(session.state["messages"]) == 4


def test_casual_chat_turn_skips_plan_execute_graph(monkeypatch):
    class CasualChatModel:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert isinstance(messages[0], SystemMessage)
            assert "普通聊天" in str(messages[0].content)
            return AIMessage(content="你好，我在。")

    class FakeGraph:
        def invoke(self, input_state, config=None):
            raise AssertionError("graph should not be used for casual chat")

    monkeypatch.setattr(loop, "build_plan_execute_graph", lambda model: FakeGraph())
    monkeypatch.setattr(loop, "build_chat_model", lambda model: CasualChatModel())

    session = loop.MokioclawSession(model="demo-model")
    outcome = session.run_turn("你好")

    assert outcome.need_tool is False
    assert outcome.response == "你好，我在。"
    assert "normal conversation" in outcome.raw
    assert session.state is not None
    assert len(session.state["messages"]) == 2


def test_manual_compact_rewrites_session_context(monkeypatch):
    planner = StageModel(
        stage="planner",
        responses=[
            json.dumps(
                {
                    "steps": [],
                    "final_response": "我已经记住这段上下文了。",
                },
                ensure_ascii=False,
            )
        ],
    )
    executor = StageModel(stage="executor", responses=[])
    finalizer = StageModel(stage="finalizer", responses=[])
    compactor = StageModel(
        stage="compactor",
        responses=[
            AIMessage(
                content=(
                    "## Active Objective\n- 继续后续会话\n\n"
                    "## Confirmed Decisions\n- 保留关键事实\n\n"
                    "## Files And Changes\n- 暂无\n\n"
                    "## Current Working State\n- 已完成一次回答\n\n"
                    "## Open Questions Or Risks\n- 无"
                )
            )
        ],
    )

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer, compactor),
    )

    session = loop.MokioclawSession(
        model="demo-model",
        context_char_limit=160,
        compact_tail_messages=1,
    )
    session.run_turn("请记住：" + ("很长的上下文。" * 20))
    outcome = session.compact_session("保留关键事实")

    assert outcome.response is not None
    assert "已完成手动上下文压缩" in outcome.response
    assert session.state is not None
    assert session.state["compaction_count"] == 1
    assert "保留关键事实" == session.state["last_compaction_focus"]
    assert session.state["compaction_summary"].startswith("## Active Objective")
    assert isinstance(session.state["messages"][0], SystemMessage)
    assert str(session.state["messages"][0].content).startswith(
        loop.COMPACTION_SYSTEM_PREFIX
    )


def test_auto_compact_runs_before_turn_when_context_exceeds_limit(monkeypatch):
    planner = StageModel(
        stage="planner",
        responses=[
            json.dumps(
                {
                    "steps": [],
                    "final_response": "第一轮完成。",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "steps": [],
                    "final_response": "第二轮继续。",
                },
                ensure_ascii=False,
            ),
        ],
    )
    executor = StageModel(stage="executor", responses=[])
    finalizer = StageModel(stage="finalizer", responses=[])
    compactor = StageModel(
        stage="compactor",
        responses=[
            AIMessage(
                content=(
                    "## Active Objective\n- 延续上一轮对话\n\n"
                    "## Confirmed Decisions\n- 保留最近结论\n\n"
                    "## Files And Changes\n- 暂无\n\n"
                    "## Current Working State\n- 等待下一轮输入\n\n"
                    "## Open Questions Or Risks\n- 无"
                )
            )
        ],
    )

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer, compactor),
    )

    session = loop.MokioclawSession(
        model="demo-model",
        context_char_limit=150,
        compact_tail_messages=1,
    )
    session.run_turn("请先记住：" + ("上下文很长。" * 20))
    outcome = session.run_turn("继续")

    assert outcome.response == "第二轮继续。"
    assert "Compaction: automatic context compaction completed." in outcome.raw
    assert session.state is not None
    assert session.state["compaction_count"] == 1


def test_run_single_step_can_organize_workspace_with_todos_and_notepad(
    tmp_path, monkeypatch
):
    demo_dir = tmp_path / "demo"
    archive_dir = tmp_path / "archive"
    demo_dir.mkdir()
    archive_dir.mkdir()
    (demo_dir / "a.txt").write_text("AI IDE 规则和扩展性建议\n", encoding="utf-8")
    (demo_dir / "b.md").write_text("每周训练计划\n", encoding="utf-8")
    (demo_dir / "c.md").write_text("成长心态与刻意练习\n", encoding="utf-8")
    (archive_dir / "b.txt").write_text("LLM 面试题整理\n", encoding="utf-8")

    renamed_demo_ai = demo_dir / "ai_ide_rules.txt"
    renamed_demo_plan = demo_dir / "weekly_fitness_plan.md"
    renamed_demo_growth = demo_dir / "growth_notes.md"
    renamed_archive_llm = archive_dir / "llm_interview_topics.txt"

    planner = StageModel(
        stage="planner",
        responses=[
            _planner_payload(
                [
                    "检查 archive 和 demo 里的文件内容并按主题分类。",
                    "按分类结果重命名文件。",
                    "为每个目录写入整理总结。",
                    "复查总结文件并补充最终整理结论。",
                ]
            )
        ],
    )
    executor = StageModel(
        stage="executor",
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "todo_write",
                        {
                            "todos": [
                                {
                                    "content": (
                                        "检查 archive 和 demo "
                                        "里的文件内容并按主题分类。"
                                    ),
                                    "status": "in_progress",
                                },
                                {
                                    "content": "按分类结果重命名文件。",
                                    "status": "pending",
                                },
                                {
                                    "content": "为每个目录写入整理总结。",
                                    "status": "pending",
                                },
                                {
                                    "content": "复查总结文件并补充最终整理结论。",
                                    "status": "pending",
                                },
                            ]
                        },
                        "call_1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "bash",
                        {"command": "find demo archive -maxdepth 1 -type f"},
                        "call_2",
                    ),
                    _tool_call("bash", {"command": "cat demo/a.txt"}, "call_3"),
                    _tool_call("bash", {"command": "cat demo/b.md"}, "call_4"),
                    _tool_call("bash", {"command": "cat demo/c.md"}, "call_5"),
                    _tool_call("bash", {"command": "cat archive/b.txt"}, "call_6"),
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "notepad_write",
                        {
                            "note": (
                                "demo 包含 AI IDE、健身计划、成长笔记；archive 包含 "
                                "LLM 面试题。"
                            )
                        },
                        "call_7",
                    )
                ],
            ),
            AIMessage(content="步骤1完成：已读取并归类文件内容。"),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "move_file",
                        {
                            "src": str(demo_dir / "a.txt"),
                            "dst": str(renamed_demo_ai),
                        },
                        "call_8",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(demo_dir / "b.md"),
                            "dst": str(renamed_demo_plan),
                        },
                        "call_9",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(demo_dir / "c.md"),
                            "dst": str(renamed_demo_growth),
                        },
                        "call_10",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(archive_dir / "b.txt"),
                            "dst": str(renamed_archive_llm),
                        },
                        "call_11",
                    ),
                ],
            ),
            AIMessage(content="步骤2完成：已按主题完成重命名。"),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "file_write",
                        {
                            "path": "demo/summary.md",
                            "content": (
                                "# demo 内容概览\n\n"
                                "- ai_ide_rules.txt: AI IDE 使用规范与扩展性建议。\n"
                                "- weekly_fitness_plan.md: 一周力量训练安排。\n"
                                "- growth_notes.md: 成长心态与刻意练习笔记。\n\n"
                                "占位：待补充整理结论\n"
                            ),
                        },
                        "call_12",
                    ),
                    _tool_call(
                        "file_write",
                        {
                            "path": "archive/summary.md",
                            "content": (
                                "# archive 内容概览\n\n"
                                "- llm_interview_topics.txt: 大模型基础、微调、"
                                "RAG、推理部署面试题整理。\n"
                            ),
                            "overwrite": False,
                        },
                        "call_13",
                    ),
                ],
            ),
            AIMessage(content="步骤3完成：已写入目录总结。"),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("bash", {"command": "cat demo/summary.md"}, "call_14")
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "file_edit",
                        {
                            "path": "demo/summary.md",
                            "old_string": "占位：待补充整理结论",
                            "new_string": (
                                "整理结论：demo 文件已按主题重命名，并整理为 "
                                "AI 工具经验、训练计划和成长笔记三类内容。"
                            ),
                        },
                        "call_15",
                    )
                ],
            ),
            AIMessage(content="步骤4完成：已补充最终整理结论。"),
        ],
    )
    finalizer = StageModel(
        stage="finalizer",
        responses=[
            AIMessage(
                content=(
                    "整理完成：archive 和 demo 中的文件已经按内容主题分类、重命名，"
                    "并分别生成了 summary.md。"
                )
            )
        ],
    )

    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    outcome = loop.run_single_step(
        "归类 archive 和 demo 的不同类型文件，并整理归纳内容后重命名。",
        model="demo-model",
    )

    assert outcome.response is not None
    assert "整理完成" in outcome.response
    assert renamed_demo_ai.exists()
    assert renamed_demo_plan.exists()
    assert renamed_demo_growth.exists()
    assert renamed_archive_llm.exists()
    assert "# demo 内容概览" in (demo_dir / "summary.md").read_text(encoding="utf-8")
    assert "整理结论：" in (demo_dir / "summary.md").read_text(encoding="utf-8")
    assert "archive 内容概览" in (archive_dir / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "Planner: generated execution plan" in outcome.raw
    assert "Completed Step 4/4" in outcome.raw
    assert outcome.tool_calls is not None
    assert any(tool_call.name == "todo_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "notepad_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "bash" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_edit" for tool_call in outcome.tool_calls)
    assert outcome.todos is not None
    assert all(todo.status == "completed" for todo in outcome.todos)
    assert outcome.notepad is not None
    assert "LLM 面试题" in outcome.notepad[0]
    assert outcome.verification_nudge is None


def test_complex_plan_without_verification_emits_nudge(monkeypatch):
    planner = StageModel(
        stage="planner",
        responses=[
            _planner_payload(
                [
                    "读取输入目录。",
                    "重命名相关文件。",
                    "生成整理总结。",
                ]
            )
        ],
    )
    executor = StageModel(
        stage="executor",
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "todo_write",
                        {
                            "todos": [
                                {"content": "读取输入目录。", "status": "in_progress"},
                                {"content": "重命名相关文件。", "status": "pending"},
                                {"content": "生成整理总结。", "status": "pending"},
                            ]
                        },
                        "call_1",
                    )
                ],
            ),
            AIMessage(content="步骤1完成"),
            AIMessage(content="步骤2完成"),
            AIMessage(content="步骤3完成"),
        ],
    )
    finalizer = StageModel(
        stage="finalizer",
        responses=[AIMessage(content="任务完成。")],
    )

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    outcome = loop.run_single_step("整理目录", model="demo-model")

    assert outcome.verification_nudge is not None
    assert "verification step" in outcome.verification_nudge


def test_planner_clarification_is_specific_and_actionable(monkeypatch):
    planner = StageModel(
        stage="planner",
        responses=[
            json.dumps(
                {
                    "steps": [],
                    "final_response": "",
                    "needs_clarification": True,
                    "clarification_question": "你希望我按主题分类，还是只做重命名？",
                    "missing_information": ["缺少整理方式"],
                    "suggested_user_replies": ["按主题分类", "只做重命名"],
                    "assumption_if_user_unsure": "按主题分类继续",
                },
                ensure_ascii=False,
            )
        ],
    )
    executor = StageModel(stage="executor", responses=[])
    finalizer = StageModel(stage="finalizer", responses=[])

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    outcome = loop.run_single_step("帮我整理 demo 和 archive", model="demo-model")

    assert outcome.response is not None
    assert "我还缺少以下信息才能继续" in outcome.response
    assert "缺少整理方式" in outcome.response
    assert "按主题分类" in outcome.response
    assert "默认假设继续" in outcome.response


def test_repeated_clarification_triggers_loop_guard_message(monkeypatch):
    clarification_payload = json.dumps(
        {
            "steps": [],
            "final_response": "",
            "needs_clarification": True,
            "clarification_question": "你希望我按主题分类，还是只做重命名？",
            "missing_information": ["缺少整理方式"],
            "suggested_user_replies": ["按主题分类", "只做重命名"],
            "assumption_if_user_unsure": "按主题分类继续",
        },
        ensure_ascii=False,
    )
    planner = StageModel(
        stage="planner",
        responses=[clarification_payload, clarification_payload],
    )
    executor = StageModel(stage="executor", responses=[])
    finalizer = StageModel(stage="finalizer", responses=[])

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        _build_chat_model_factory(planner, executor, finalizer),
    )

    session = loop.MokioclawSession(model="demo-model")
    first = session.run_turn("帮我整理 demo 和 archive")
    second = session.run_turn("你先看看")

    assert first.response is not None
    assert second.response is not None
    assert "为避免重复卡住" in second.response
    assert "Loop Guard: repeated clarification detected" in second.raw


def test_graph_recursion_guard_returns_user_friendly_response(monkeypatch):
    class FakeGraph:
        def invoke(self, input_state, config=None):
            raise GraphRecursionError("too many steps")

    monkeypatch.setattr(loop, "build_plan_execute_graph", lambda model: FakeGraph())

    outcome = loop.run_single_step("一直循环的任务", model="demo-model")

    assert outcome.response is not None
    assert "我停止了本轮执行" in outcome.response
    assert "Loop Guard" in outcome.raw
