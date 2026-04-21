from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

import mokioclaw.core.loop as loop
import mokioclaw.tools.workspace_tools as workspace_tools


class FakeChatModel:
    def __init__(self, *, src: str, dst: str):
        self.src = src
        self.dst = dst
        self.calls = 0
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> FakeChatModel:
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1

        assert isinstance(messages[0], SystemMessage)
        assert "ReAct" in str(messages[0].content)
        assert self.bound_tools

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "move_file",
                        "args": {"src": self.src, "dst": self.dst},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(content="文件已经移动完成。")


class SessionAwareFakeChatModel:
    def __init__(self):
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> SessionAwareFakeChatModel:
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        human_message_count = sum(
            1
            for message in messages
            if getattr(message, "type", None) == "human"
        )
        return AIMessage(content=f"目前会话里有 {human_message_count} 条用户消息。")


class OrganizeWorkspaceFakeChatModel:
    def __init__(self, *, workspace_root):
        self.workspace_root = workspace_root
        self.calls = 0
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> OrganizeWorkspaceFakeChatModel:
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls += 1

        assert isinstance(messages[0], SystemMessage)
        assert "file_edit" in str(messages[0].content)
        assert "bash" in str(messages[0].content)
        assert self.bound_tools

        demo_ai = self.workspace_root / "demo" / "ai_ide_rules.txt"
        demo_plan = self.workspace_root / "demo" / "weekly_fitness_plan.md"
        demo_growth = self.workspace_root / "demo" / "growth_notes.md"
        archive_llm = self.workspace_root / "archive" / "llm_interview_topics.txt"

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "find demo archive -maxdepth 1 -type f"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "bash",
                        "args": {"command": "cat demo/a.txt"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                    {
                        "name": "bash",
                        "args": {"command": "cat demo/b.md"},
                        "id": "call_3",
                        "type": "tool_call",
                    },
                    {
                        "name": "bash",
                        "args": {"command": "cat demo/c.md"},
                        "id": "call_4",
                        "type": "tool_call",
                    },
                    {
                        "name": "bash",
                        "args": {"command": "cat archive/b.txt"},
                        "id": "call_5",
                        "type": "tool_call",
                    },
                ],
            )

        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "move_file",
                        "args": {
                            "src": str(self.workspace_root / "demo" / "a.txt"),
                            "dst": str(demo_ai),
                        },
                        "id": "call_6",
                        "type": "tool_call",
                    },
                    {
                        "name": "move_file",
                        "args": {
                            "src": str(self.workspace_root / "demo" / "b.md"),
                            "dst": str(demo_plan),
                        },
                        "id": "call_7",
                        "type": "tool_call",
                    },
                    {
                        "name": "move_file",
                        "args": {
                            "src": str(self.workspace_root / "demo" / "c.md"),
                            "dst": str(demo_growth),
                        },
                        "id": "call_8",
                        "type": "tool_call",
                    },
                    {
                        "name": "move_file",
                        "args": {
                            "src": str(self.workspace_root / "archive" / "b.txt"),
                            "dst": str(archive_llm),
                        },
                        "id": "call_9",
                        "type": "tool_call",
                    },
                ],
            )

        if self.calls == 3:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_write",
                        "args": {
                            "path": "demo/summary.md",
                            "content": (
                                "# demo 内容概览\n\n"
                                "- ai_ide_rules.txt: AI IDE 使用规范与扩展性建议。\n"
                                "- weekly_fitness_plan.md: 一周力量训练安排。\n"
                                "- growth_notes.md: 成长心态与刻意练习笔记。\n\n"
                                "占位：待补充整理结论\n"
                            ),
                        },
                        "id": "call_10",
                        "type": "tool_call",
                    },
                    {
                        "name": "file_write",
                        "args": {
                            "path": "archive/summary.md",
                            "content": (
                                "# archive 内容概览\n\n"
                                "- llm_interview_topics.txt: 大模型基础、微调、"
                                "RAG、推理部署面试题整理。\n"
                            ),
                        },
                        "id": "call_11",
                        "type": "tool_call",
                    },
                ],
            )

        if self.calls == 4:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "cat demo/summary.md"},
                        "id": "call_12",
                        "type": "tool_call",
                    }
                ],
            )

        if self.calls == 5:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "file_edit",
                        "args": {
                            "path": "demo/summary.md",
                            "old_string": "占位：待补充整理结论",
                            "new_string": (
                                "整理结论：demo 文件已按主题重命名，并整理为 "
                                "AI 工具经验、"
                                "训练计划和成长笔记三类内容。"
                            ),
                        },
                        "id": "call_13",
                        "type": "tool_call",
                    },
                ],
            )

        return AIMessage(
            content=(
                "整理完成：archive 和 demo 中的文件已经按内容主题分类、重命名，"
                "并分别生成了 summary.md。"
            )
        )


def test_run_single_step_uses_langgraph_tool_loop(tmp_path, monkeypatch):
    source = tmp_path / "demo" / "a.txt"
    target = tmp_path / "archive" / "a.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(
        loop,
        "build_chat_model",
        lambda model: FakeChatModel(src=str(source), dst=str(target)),
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
    assert "Tool Result [move_file]" in outcome.raw
    assert target.exists()
    assert not source.exists()


def test_session_preserves_multi_turn_history(monkeypatch):
    monkeypatch.setattr(
        loop,
        "build_chat_model",
        lambda model: SessionAwareFakeChatModel(),
    )

    session = loop.MokioclawSession(model="demo-model")
    first = session.run_turn("你好")
    second = session.run_turn("继续")

    assert first.response == "目前会话里有 1 条用户消息。"
    assert second.response == "目前会话里有 2 条用户消息。"
    assert session.state is not None
    assert len(session.state["messages"]) == 4


def test_run_single_step_can_organize_workspace_with_new_tools(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    (tmp_path / "archive").mkdir()
    (tmp_path / "demo" / "a.txt").write_text(
        "AI IDE 规则和扩展性建议\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "b.md").write_text(
        "每周训练计划\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "c.md").write_text(
        "成长心态与刻意练习\n",
        encoding="utf-8",
    )
    (tmp_path / "archive" / "b.txt").write_text(
        "LLM 面试题整理\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(workspace_tools, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(
        loop,
        "build_chat_model",
        lambda model: OrganizeWorkspaceFakeChatModel(workspace_root=tmp_path),
    )

    outcome = loop.run_single_step(
        "归类 archive 和 demo 的不同类型文件，并整理归纳内容后重命名。",
        model="demo-model",
    )

    assert outcome.response is not None
    assert "整理完成" in outcome.response
    assert (tmp_path / "demo" / "ai_ide_rules.txt").exists()
    assert (tmp_path / "demo" / "weekly_fitness_plan.md").exists()
    assert (tmp_path / "demo" / "growth_notes.md").exists()
    assert (tmp_path / "archive" / "llm_interview_topics.txt").exists()
    assert "# demo 内容概览" in (tmp_path / "demo" / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "整理结论：" in (tmp_path / "demo" / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "archive 内容概览" in (tmp_path / "archive" / "summary.md").read_text(
        encoding="utf-8"
    )
    assert outcome.tool_calls is not None
    assert any(tool_call.name == "bash" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_write" for tool_call in outcome.tool_calls)
    assert any(tool_call.name == "file_edit" for tool_call in outcome.tool_calls)
