from __future__ import annotations

import asyncio

from textual.widgets import Static

from mokioclaw.core.types import LoopOutcome, TodoSnapshot
from mokioclaw.tui.app import ChatComposer, MokioclawTextualApp


class FakeSession:
    def __init__(self) -> None:
        self.turns: list[str] = []
        self.reset_calls = 0

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
