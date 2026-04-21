from __future__ import annotations

import json
from collections.abc import Callable

from langchain_core.messages import AIMessage, SystemMessage

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
            assert self.bound_tools
        elif self.stage == "finalizer":
            assert "Finalizer" in prompt
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


def test_run_single_step_uses_plan_execute_tool_loop(tmp_path, monkeypatch):
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
                        "move_file",
                        {"src": str(source), "dst": str(target)},
                        "call_1",
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
    assert outcome.tool_calls[0].name == "move_file"
    assert outcome.tool_calls[0].arguments == {"src": str(source), "dst": str(target)}
    assert outcome.tool_calls[0].result is not None
    assert "Planner: generated execution plan" in outcome.raw
    assert "Completed Step 1/1" in outcome.raw
    assert "Tool Result [move_file]" in outcome.raw
    assert target.exists()
    assert not source.exists()


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
    first = session.run_turn("你好")
    second = session.run_turn("继续")

    assert first.response == "目前会话里有 1 条用户消息。"
    assert second.response == "目前会话里有 2 条用户消息。"
    assert "Planner: returned a direct response without execution." in first.raw
    assert session.state is not None
    assert len(session.state["messages"]) == 4


def test_run_single_step_can_organize_workspace_with_new_tools(tmp_path, monkeypatch):
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
                        "bash",
                        {"command": "find demo archive -maxdepth 1 -type f"},
                        "call_1",
                    ),
                    _tool_call("bash", {"command": "cat demo/a.txt"}, "call_2"),
                    _tool_call("bash", {"command": "cat demo/b.md"}, "call_3"),
                    _tool_call("bash", {"command": "cat demo/c.md"}, "call_4"),
                    _tool_call("bash", {"command": "cat archive/b.txt"}, "call_5"),
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
                        "call_6",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(demo_dir / "b.md"),
                            "dst": str(renamed_demo_plan),
                        },
                        "call_7",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(demo_dir / "c.md"),
                            "dst": str(renamed_demo_growth),
                        },
                        "call_8",
                    ),
                    _tool_call(
                        "move_file",
                        {
                            "src": str(archive_dir / "b.txt"),
                            "dst": str(renamed_archive_llm),
                        },
                        "call_9",
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
                        "call_10",
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
                        "call_11",
                    ),
                ],
            ),
            AIMessage(content="步骤3完成：已写入目录总结。"),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("bash", {"command": "cat demo/summary.md"}, "call_12")
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
                        "call_13",
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
    assert any(tool_call.name == "bash" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_edit" for tool_call in outcome.tool_calls)
