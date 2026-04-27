from __future__ import annotations

import asyncio

from textual.containers import Container
from textual.widgets import Static

from mokioclaw.core.types import LoopOutcome, TodoSnapshot
from mokioclaw.tui.app import ChatComposer, MokioclawTextualApp, TodoCard


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
            assert app.query_one("#todo-panel")
            assert app.query_one("#notepad-panel")

    asyncio.run(_run())


def test_textual_app_submits_input_and_updates_side_panels():
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
            assert len(app.query(".chat-card.assistant")) == 1
            assert len(app.query(".todo-card")) == 2
            assert len(app.query(".note-card")) == 1
            verification_text = app.query_one("#verification-text", Static)
            assert "verification step" in str(verification_text.render())

    asyncio.run(_run())


class CompletedTodoSession(FakeSession):
    def run_turn(self, user_input: str) -> LoopOutcome:
        self.turns.append(user_input)
        return LoopOutcome(
            need_tool=True,
            raw="Todo Panel cleared after all items completed.",
            response="任务完成。",
            todos=None,
        )


def test_textual_app_clears_todo_panel_when_outcome_has_no_active_todos():
    async def _run() -> None:
        session = CompletedTodoSession()
        app = MokioclawTextualApp(model="demo-model", session=session)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            input_widget = app.query_one("#chat-input", ChatComposer)
            input_widget.load_text("finish task")
            await pilot.press("enter")
            await pilot.pause(0.4)

            assert len(app.query(".todo-card")) == 0
            empty_message = app.query_one("#todo-scroll .empty-panel-message", Static)
            assert "No active checklist yet." in str(empty_message.render())

    asyncio.run(_run())


def test_textual_todo_render_updates_existing_cards_for_status_only_changes():
    async def _run() -> None:
        app = MokioclawTextualApp(model="demo-model", session=FakeSession())
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await app._render_todos(
                [
                    TodoSnapshot(content="读取目录", status="in_progress"),
                    TodoSnapshot(content="整理文件", status="pending"),
                ]
            )
            await pilot.pause()
            original_cards = list(app.query(TodoCard))

            await app._render_todos(
                [
                    TodoSnapshot(content="读取目录", status="completed"),
                    TodoSnapshot(content="整理文件", status="in_progress"),
                ]
            )
            await pilot.pause()
            updated_cards = list(app.query(TodoCard))

            assert updated_cards == original_cards
            assert updated_cards[0].has_class("completed")
            assert updated_cards[1].has_class("in_progress")
            assert "[x] Step 1" in str(
                updated_cards[0].query_one(".todo-kicker", Static).render()
            )

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
