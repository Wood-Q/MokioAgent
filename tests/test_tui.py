from __future__ import annotations

import asyncio

from textual.containers import Container
from textual.widgets import Static

from mokioclaw.core.types import LoopOutcome, TodoSnapshot
from mokioclaw.tui.app import ChatComposer, MokioclawTextualApp


class FakeSession:
    def __init__(self) -> None:
        self.turns: list[str] = []
        self.reset_calls = 0
        self.compact_calls: list[str | None] = []

    def run_turn(self, user_input: str) -> LoopOutcome:
        self.turns.append(user_input)
        return LoopOutcome(
            need_tool=True,
            raw="Planner: generated execution plan",
            response="整理完成。",
            todos=[
                TodoSnapshot(content="读取目录", status="completed"),
                TodoSnapshot(content="整理文件", status="in_progress"),
            ],
            notepad=[
                "## Findings\n\n- archive has interview notes\n- demo has mixed content"
            ],
            verification_nudge="Consider adding a verification step.",
        )

    def compact_session(self, focus: str | None = None) -> LoopOutcome:
        self.compact_calls.append(focus)
        return LoopOutcome(
            need_tool=False,
            raw="Compaction: manual context compaction completed.",
            response="已完成手动上下文压缩。",
            todos=[
                TodoSnapshot(content="整理文件", status="in_progress"),
            ],
            notepad=["## Compacted\n\n- session summary refreshed"],
        )

    def resolve_pending_approval(self, approved: bool) -> LoopOutcome:
        return LoopOutcome(
            need_tool=False,
            raw="Approval resolved.",
            response="approval resolved.",
        )

    def reset(self) -> None:
        self.reset_calls += 1


def test_textual_app_renders_shell():
    async def _run() -> None:
        app = MokioclawTextualApp(model="demo-model", session=FakeSession())
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#chat-input", ChatComposer)
            assert app.query_one("#composer-prefix", Static)
            assert app.query_one("#conversation-zone")
            assert app.query_one("#welcome-card")
            assert app.query_one("#logo-mark", Static)
            assert app.query_one("#approval-panel")
            assert app.query_one("#verification-panel")
            assert len(app.query(".todo-card")) == 0
            assert len(app.query(".note-card")) == 0

    asyncio.run(_run())


def test_textual_app_submits_input_and_streams_assistant_response():
    async def _run() -> None:
        session = FakeSession()
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)
            input_widget.load_text("organize files")
            await pilot.press("enter")
            await pilot.pause(0.4)

            assert session.turns == ["organize files"]
            assert len(app.query(".chat-card.user")) == 1
            assert len(app.query(".chat-card.thinking")) == 0
            assert len(app.query(".chat-card.assistant")) == 1
            assert all(
                "Selecting tools" not in card.content
                and "Checking approvals" not in card.content
                for card in app.query(".chat-card")
            )
            assistant_card = app.query(".chat-card.assistant").last()
            assert "整理完成" in assistant_card.content
            assert len(app.query(".todo-card")) == 0
            assert len(app.query(".note-card")) == 0
            verification_text = app.query_one("#verification-text", Static)
            assert "verification step" in str(verification_text.render())

    asyncio.run(_run())


class ApprovalSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[bool] = []

    def run_turn(self, user_input: str) -> LoopOutcome:
        self.turns.append(user_input)
        return LoopOutcome(
            need_tool=True,
            raw="Approval: waiting",
            response="Human approval required before executing this action.",
            pending_approval={
                "id": "call_1",
                "message": "Human approval required before executing this action.",
                "tool_calls": [],
            },
        )

    def resolve_pending_approval(self, approved: bool) -> LoopOutcome:
        self.decisions.append(approved)
        return LoopOutcome(
            need_tool=False,
            raw="Approval resolved.",
            response="approval resolved.",
            pending_approval=None,
        )



def test_textual_app_renders_and_resolves_approval_panel():
    async def _run() -> None:
        session = ApprovalSession()
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)
            input_widget.load_text("move file")
            await pilot.press("enter")
            await pilot.pause(0.4)

            approval_panel = app.query_one("#approval-panel")
            approval_text = app.query_one("#approval-text", Static)
            assert not approval_panel.has_class("hidden")
            assert "Human approval required" in str(approval_text.render())

            input_widget.load_text("/approve")
            await pilot.press("enter")
            await pilot.pause(0.4)

            assert session.decisions == [True]
            assert approval_panel.has_class("hidden")

    asyncio.run(_run())


def test_textual_todo_and_notepad_commands_render_snapshots_in_chat():
    async def _run() -> None:
        session = FakeSession()
        session.state = {
            "todos": [],
            "todo_snapshot": [
                {"content": "读取目录", "status": "completed"},
                {"content": "整理文件", "status": "in_progress"},
            ],
            "notepad": ["## Findings\n\n- archive has interview notes"],
        }
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)

            input_widget.load_text("/todo")
            await pilot.press("enter")
            await pilot.pause(0.2)
            todo_card = app.query(".chat-card.system").last()
            assert "## Todo" in todo_card.content
            assert "读取目录" in todo_card.content

            input_widget.load_text("/notepad")
            await pilot.press("enter")
            await pilot.pause(0.2)
            notepad_card = app.query(".chat-card.system").last()
            assert "## NotePad" in notepad_card.content
            assert "archive has interview notes" in notepad_card.content

    asyncio.run(_run())


def test_textual_help_command_renders_grouped_command_card():
    async def _run() -> None:
        app = MokioclawTextualApp(model="demo-model", session=FakeSession())
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)
            input_widget.load_text("/help")
            await pilot.press("enter")
            await pilot.pause(0.2)

            help_card = app.query(".chat-card.system").last()
            assert "## Commands" in help_card.content
            assert "**Session**" in help_card.content
            assert "**State**" in help_card.content
            assert "**Approval**" in help_card.content

    asyncio.run(_run())



def test_textual_busy_state_marks_composer_shell():
    async def _run() -> None:
        app = MokioclawTextualApp(model="demo-model", session=FakeSession())
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            composer_shell = app.query_one("#composer-shell", Container)

            app._set_busy(True)
            assert composer_shell.has_class("busy")

            app._set_busy(False)
            assert not composer_shell.has_class("busy")

    asyncio.run(_run())


def test_textual_app_clear_command_resets_session():
    async def _run() -> None:
        session = FakeSession()
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)
            input_widget.load_text("/clear")
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert session.reset_calls == 1
            assert len(app.query(".todo-card")) == 0
            assert len(app.query(".note-card")) == 0
            assert app.query_one("#welcome-card")
            assert app.query_one("#command-cheatsheet")
            assert app.query_one("#logo-mark", Static)

    asyncio.run(_run())


def test_textual_composer_grows_with_multiple_lines():
    async def _run() -> None:
        app = MokioclawTextualApp(model="demo-model", session=FakeSession())
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            composer = app.query_one("#chat-input", ChatComposer)
            composer.load_text("line 1\nline 2\nline 3")
            await pilot.pause()

            assert composer.wrapped_document.height >= 3
            assert composer.outer_size.height >= 3

    asyncio.run(_run())


def test_textual_compact_command_uses_session_compaction():
    async def _run() -> None:
        session = FakeSession()
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            composer = app.query_one("#chat-input", ChatComposer)
            composer.load_text("/compact 保留文件修改")
            await pilot.press("enter")
            await pilot.pause(0.4)

            assert session.compact_calls == ["保留文件修改"]
            assert len(app.query(".chat-card.assistant")) == 1

    asyncio.run(_run())
