from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

import mokioclaw.core.loop as loop


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
