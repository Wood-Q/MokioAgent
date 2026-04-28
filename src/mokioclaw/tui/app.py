from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from rich_pixels import HalfcellRenderer, Pixels
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Key, Resize
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Header, Markdown, Static, TextArea

from mokioclaw.core.loop import MokioclawSession
from mokioclaw.core.memory import (
    coerce_todo_snapshots,
    render_notepad,
    render_todo_panel,
)
from mokioclaw.core.state import PendingApprovalState
from mokioclaw.core.types import LoopOutcome

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMANDS = {"/help", "help"}
CLEAR_COMMANDS = {"/clear"}
COMPACT_COMMAND = "/compact"
APPROVE_COMMANDS = {"/approve", "approve"}
DENY_COMMANDS = {"/deny", "deny"}
TODO_COMMANDS = {"/todo", "todo"}
NOTEPAD_COMMANDS = {"/notepad", "notepad"}
LOGO_IMAGE_PATH = Path(__file__).resolve().parents[3] / "logo with no words.png"
TYPEWRITER_CHUNK_SIZE = 12
TYPEWRITER_DELAY_SECONDS = 0.01
THINKING_TICK_SECONDS = 0.35
THINKING_STEPS = (
    "Thinking",
    "Reading the conversation",
    "Waiting for response",
)


@dataclass(frozen=True)
class SessionViewState:
    response: str
    verification_nudge: str | None
    pending_approval: PendingApprovalState | None


class SessionLike(Protocol):
    def run_turn(self, user_input: str) -> LoopOutcome: ...

    def compact_session(self, focus: str | None = None) -> LoopOutcome: ...

    def resolve_pending_approval(self, approved: bool) -> LoopOutcome: ...

    def reset(self) -> None: ...


class WelcomePanel(Widget):
    def compose(self) -> ComposeResult:
        with Horizontal(id="welcome-card"):
            with Vertical(id="welcome-left"):
                yield Static(_logo_mark(), id="logo-mark")
                yield Static("Welcome back!", id="welcome-title")
                yield Static("MokioAgent · Textual Session", id="welcome-meta")
            with Vertical(id="welcome-right"):
                yield Static(
                    "Tips for getting started",
                    classes="welcome-section-title",
                )
                yield Static(
                    "Run /help to list slash commands",
                    classes="welcome-line",
                )
                yield Static(
                    "Use /todo or /notepad to inspect saved state",
                    classes="welcome-line",
                )
                yield Static("Commands", classes="welcome-section-title")
                yield Markdown(
                    _command_help_markdown(compact=True),
                    id="command-cheatsheet",
                )
                yield Static("Recent activity", classes="welcome-section-title")
                yield Static("No recent activity", classes="welcome-line muted")


class ChatCard(Widget):
    def __init__(
        self,
        *,
        role: Literal["user", "assistant", "system", "thinking", "error"],
        content: str,
    ) -> None:
        super().__init__(classes=f"chat-card {role}")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield Static(_role_label(self.role), classes="chat-kicker")
        if self.role == "user":
            yield Static(self.content, classes="chat-plain")
        else:
            yield Markdown(self.content, classes="chat-markdown")

    def update_content(self, content: str) -> None:
        self.content = content
        if self.role == "user":
            self.query_one(".chat-plain", Static).update(content)
        else:
            self.query_one(".chat-markdown", Markdown).update(content)


class AgentTurnReady(Message):
    def __init__(self, outcome: LoopOutcome) -> None:
        self.outcome = outcome
        super().__init__()


class AgentTurnFailed(Message):
    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        super().__init__()


class ComposerSubmitted(Message):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__()


class ChatComposer(TextArea):
    MIN_LINES = 1
    MAX_LINES = 5

    def on_mount(self) -> None:
        self.sync_height()

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.action_submit()
            return
        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.action_insert_newline()

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(ComposerSubmitted(text))

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def sync_height(self) -> None:
        wrapped_height = max(1, getattr(self.wrapped_document, "height", 1))
        self.styles.height = min(self.MAX_LINES, max(self.MIN_LINES, wrapped_height))


class MokioclawTextualApp(App[None]):
    CSS_PATH = Path(__file__).with_name("mokioclaw.tcss")
    BINDINGS = [
        Binding("ctrl+j", "focus_input", "Focus Input"),
        Binding("ctrl+l", "clear_session", "Clear Session"),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        *,
        model: str,
        initial_message: str | None = None,
        session: SessionLike | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.initial_message = initial_message
        self.session = session or MokioclawSession(model=model)
        self._pending_card: ChatCard | None = None
        self._thinking_task: asyncio.Task[None] | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="shell"):
            with Vertical(id="conversation-zone"):
                yield Static("Conversation", id="conversation-title")
                with VerticalScroll(id="chat-scroll"):
                    yield WelcomePanel()
                with Container(id="composer-shell"):
                    yield Static(_composer_hint(), id="composer-help")
                    with Horizontal(id="composer-row"):
                        yield Static("›", id="composer-prefix")
                        yield ChatComposer(
                            soft_wrap=True,
                            show_line_numbers=False,
                            highlight_cursor_line=False,
                            compact=True,
                            placeholder="先聊聊，或直接给我一个任务...",
                            id="chat-input",
                        )
            with Vertical(id="sidebar-zone"):
                yield Static("Runtime Panels", id="sidebar-title")
                with Vertical(
                    id="approval-panel",
                    classes="side-panel approval-panel hidden",
                ):
                    yield Static("Human Approval", classes="panel-title")
                    yield Static("", id="approval-text")
                with Vertical(
                    id="verification-panel",
                    classes="side-panel verify-panel hidden",
                ):
                    yield Static("Verifier", classes="panel-title")
                    yield Static("", id="verification-text")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Mokioclaw"
        self.sub_title = f"Textual Session · {self.model}"
        self._sync_composer_height()
        self.query_one("#chat-input", ChatComposer).focus()
        if self.initial_message:
            self.set_timer(0, self._dispatch_initial_message)

    def action_focus_input(self) -> None:
        self.query_one("#chat-input", ChatComposer).focus()

    async def action_clear_session(self) -> None:
        self.session.reset()
        await self._reset_panels()
        await self._replace_chat_with_welcome()
        self.sub_title = f"Textual Session · {self.model}"
        self.notify("Session cleared.", timeout=2)

    async def on_composer_submitted(self, message: ComposerSubmitted) -> None:
        composer = self.query_one("#chat-input", ChatComposer)
        user_input = message.value.strip()
        composer.clear()
        self._sync_composer_height()
        if not user_input or self._busy:
            return
        if user_input in EXIT_COMMANDS:
            self.exit()
            return
        if user_input in HELP_COMMANDS:
            await self._mount_chat_card(
                ChatCard(
                    role="system",
                    content=_command_help_markdown(compact=False),
                )
            )
            return
        if user_input in CLEAR_COMMANDS:
            await self.action_clear_session()
            return
        if user_input in APPROVE_COMMANDS:
            await self._begin_approval_resolution(user_input, approved=True)
            return
        if user_input in DENY_COMMANDS:
            await self._begin_approval_resolution(user_input, approved=False)
            return
        if user_input in TODO_COMMANDS:
            await self._show_todo_snapshot()
            return
        if user_input in NOTEPAD_COMMANDS:
            await self._show_notepad_snapshot()
            return
        compact_focus = _parse_compact_command(user_input)
        if compact_focus is not None:
            await self._begin_compact(user_input, compact_focus or None)
            return
        await self._begin_turn(user_input)

    @on(TextArea.Changed, "#chat-input")
    def _on_composer_changed(self, event: TextArea.Changed) -> None:
        self._sync_composer_height()

    def on_resize(self, event: Resize) -> None:
        self.call_after_refresh(self._sync_composer_height)

    async def on_agent_turn_ready(self, message: AgentTurnReady) -> None:
        await self._finish_pending_card()
        assistant_card = ChatCard(role="assistant", content="")
        await self._mount_chat_card(assistant_card)
        await self._stream_card_content(
            assistant_card,
            message.outcome.response or "(no response)",
        )
        await self._apply_outcome(
            SessionViewState(
                response=message.outcome.response or "(no response)",
                verification_nudge=message.outcome.verification_nudge,
                pending_approval=message.outcome.pending_approval,
            )
        )
        self._set_busy(False)

    async def on_agent_turn_failed(self, message: AgentTurnFailed) -> None:
        await self._finish_pending_card()
        await self._mount_chat_card(
            ChatCard(
                role="error",
                content=(
                    "## Runtime Error\n\n"
                    f"```\n{message.error_text}\n```"
                ),
            )
        )
        self._set_busy(False)

    @work(thread=True, group="agent-turn", exclusive=True, exit_on_error=False)
    def run_agent_turn(self, user_input: str) -> None:
        try:
            outcome = self.session.run_turn(user_input)
        except Exception as exc:  # pragma: no cover - covered through message branch
            self.call_from_thread(self.post_message, AgentTurnFailed(str(exc)))
            return
        self.call_from_thread(self.post_message, AgentTurnReady(outcome))

    @work(thread=True, group="agent-turn", exclusive=True, exit_on_error=False)
    def run_approval_resolution(self, approved: bool) -> None:
        try:
            outcome = self.session.resolve_pending_approval(approved)
        except Exception as exc:  # pragma: no cover - covered through message branch
            self.call_from_thread(self.post_message, AgentTurnFailed(str(exc)))
            return
        self.call_from_thread(self.post_message, AgentTurnReady(outcome))

    @work(thread=True, group="agent-turn", exclusive=True, exit_on_error=False)
    def run_compact_turn(self, focus: str | None) -> None:
        try:
            outcome = self.session.compact_session(focus)
        except Exception as exc:  # pragma: no cover - covered through message branch
            self.call_from_thread(self.post_message, AgentTurnFailed(str(exc)))
            return
        self.call_from_thread(self.post_message, AgentTurnReady(outcome))

    def _dispatch_initial_message(self) -> None:
        assert self.initial_message is not None
        self.run_worker(self._begin_turn(self.initial_message))

    async def _show_todo_snapshot(self) -> None:
        state = getattr(self.session, "state", None)
        todos = []
        if state:
            todos = coerce_todo_snapshots(
                state.get("todos") or state.get("todo_snapshot")
            )
        content = (
            "## Todo\n\n" + render_todo_panel(todos)
            if todos
            else "## Todo\n\nNo todo items yet."
        )
        await self._mount_chat_card(ChatCard(role="system", content=content))

    async def _show_notepad_snapshot(self) -> None:
        state = getattr(self.session, "state", None)
        notes = list(state.get("notepad", [])) if state else []
        content = (
            "## NotePad\n\n" + render_notepad(notes)
            if notes
            else "## NotePad\n\nNo saved notes yet."
        )
        await self._mount_chat_card(ChatCard(role="system", content=content))

    async def _begin_turn(self, user_input: str) -> None:
        self._set_busy(True)
        await self._mount_chat_card(ChatCard(role="user", content=user_input))
        pending_card = ChatCard(
            role="thinking",
            content="### Thinking\n\nWaiting for response...",
        )
        self._pending_card = pending_card
        await self._mount_chat_card(pending_card)
        self._start_thinking_stream(pending_card)
        self.run_agent_turn(user_input)

    async def _begin_approval_resolution(
        self,
        command_text: str,
        *,
        approved: bool,
    ) -> None:
        self._set_busy(True)
        await self._mount_chat_card(ChatCard(role="user", content=command_text))
        pending_card = ChatCard(
            role="thinking",
            content=(
                "### Resolving approval\n\n"
                "Applying the human approval decision to the pending tool call."
            ),
        )
        self._pending_card = pending_card
        await self._mount_chat_card(pending_card)
        self.run_approval_resolution(approved)

    async def _begin_compact(
        self,
        command_text: str,
        focus: str | None,
    ) -> None:
        self._set_busy(True)
        await self._mount_chat_card(ChatCard(role="user", content=command_text))
        pending_card = ChatCard(
            role="thinking",
            content=(
                "### Compacting context\n\n"
                "Compressing the current session context so later turns can "
                "continue with a smaller prompt window."
            ),
        )
        self._pending_card = pending_card
        await self._mount_chat_card(pending_card)
        self.run_compact_turn(focus)

    async def _apply_outcome(self, state: SessionViewState) -> None:
        await self._render_pending_approval(state.pending_approval)
        verification_panel = self.query_one("#verification-panel", Vertical)
        verification_text = self.query_one("#verification-text", Static)
        if state.verification_nudge:
            verification_panel.remove_class("hidden")
            verification_text.update(state.verification_nudge)
        else:
            verification_panel.add_class("hidden")
            verification_text.update("")

    async def _replace_chat_with_welcome(self) -> None:
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        await chat_scroll.remove_children()
        await chat_scroll.mount(WelcomePanel())
        chat_scroll.scroll_end(animate=False)

    async def _mount_chat_card(self, card: ChatCard) -> None:
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        await chat_scroll.mount(card)
        chat_scroll.scroll_end(animate=False)

    async def _stream_card_content(self, card: ChatCard, content: str) -> None:
        if not content:
            card.update_content("")
            return
        for end in range(TYPEWRITER_CHUNK_SIZE, len(content), TYPEWRITER_CHUNK_SIZE):
            card.update_content(content[:end])
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
            await asyncio.sleep(TYPEWRITER_DELAY_SECONDS)
        card.update_content(content)
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)

    def _start_thinking_stream(self, card: ChatCard) -> None:
        self._thinking_task = asyncio.create_task(self._run_thinking_stream(card))

    async def _run_thinking_stream(self, card: ChatCard) -> None:
        ticks = 0
        while True:
            step = THINKING_STEPS[ticks % len(THINKING_STEPS)]
            dots = "." * ((ticks % 3) + 1)
            card.update_content(f"### Thinking\n\n{step}{dots}")
            self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)
            ticks += 1
            await asyncio.sleep(THINKING_TICK_SECONDS)

    async def _finish_pending_card(self) -> None:
        if self._thinking_task is not None:
            self._thinking_task.cancel()
            try:
                await self._thinking_task
            except asyncio.CancelledError:
                pass
            self._thinking_task = None
        if self._pending_card is not None:
            await self._pending_card.remove()
            self._pending_card = None

    async def _render_pending_approval(
        self,
        pending_approval: PendingApprovalState | None,
    ) -> None:
        approval_panel = self.query_one("#approval-panel", Vertical)
        approval_text = self.query_one("#approval-text", Static)
        if pending_approval:
            approval_panel.remove_class("hidden")
            approval_text.update(str(pending_approval.get("message", "")))
        else:
            approval_panel.add_class("hidden")
            approval_text.update("")

    async def _reset_panels(self) -> None:
        await self._render_pending_approval(None)
        verification_panel = self.query_one("#verification-panel", Vertical)
        verification_text = self.query_one("#verification-text", Static)
        verification_panel.add_class("hidden")
        verification_text.update("")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        input_widget = self.query_one("#chat-input", ChatComposer)
        help_widget = self.query_one("#composer-help", Static)
        composer_shell = self.query_one("#composer-shell", Container)
        composer_shell.set_class(busy, "busy")
        input_widget.disabled = busy
        if busy:
            help_widget.update("正在处理这一轮消息...")
            self.sub_title = f"Textual Session · {self.model} · Running"
        else:
            help_widget.update(_composer_hint())
            self.sub_title = f"Textual Session · {self.model}"
            input_widget.focus()

    def _sync_composer_height(self) -> None:
        composer = self.query_one("#chat-input", ChatComposer)
        composer.sync_height()


def run_textual_chat(*, message: str | None, model: str) -> int:
    app = MokioclawTextualApp(model=model, initial_message=message)
    app.run()
    return 0



def _logo_mark() -> Pixels:
    return Pixels.from_image_path(
        LOGO_IMAGE_PATH,
        resize=(44, 18),
        renderer=HalfcellRenderer(default_color="#12100f"),
    )


def _command_help_markdown(*, compact: bool) -> str:
    if compact:
        return (
            "`/help` commands · `/todo` checklist · `/notepad` notes · "
            "`/compact` context · `/clear` reset"
        )
    return (
        "## Commands\n\n"
        "**Session**\n"
        "- `/clear` reset the conversation and runtime panels\n"
        "- `/compact [focus]` compact the session context\n"
        "- `/exit` quit the app\n\n"
        "**State**\n"
        "- `/todo` show the latest todo snapshot in chat\n"
        "- `/notepad` show saved notepad entries in chat\n\n"
        "**Approval**\n"
        "- `/approve` approve pending tool calls\n"
        "- `/deny` deny pending tool calls\n\n"
        "**Help**\n"
        "- `/help` show this command card\n\n"
        "Thinking and assistant responses stream inline. Todo and NotePad "
        "are shown on demand instead of as persistent side panels."
    )


def _composer_hint() -> str:
    return "/help for commands · Enter to send · Shift+Enter newline · Ctrl+Q quit"


def _role_label(role: str) -> str:
    if role == "user":
        return "You"
    if role == "assistant":
        return "Mokioclaw"
    if role == "thinking":
        return "Running"
    if role == "error":
        return "Error"
    return "System"


def _parse_compact_command(user_input: str) -> str | None:
    if user_input == COMPACT_COMMAND:
        return ""
    if not user_input.startswith(f"{COMPACT_COMMAND} "):
        return None
    focus = user_input[len(COMPACT_COMMAND) :].strip()
    return focus or ""
