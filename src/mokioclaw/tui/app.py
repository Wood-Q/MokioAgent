from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Key, Resize
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Header, Markdown, Static, TextArea

from mokioclaw.core.loop import MokioclawSession
from mokioclaw.core.types import LoopOutcome, TodoSnapshot

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}
HELP_COMMANDS = {"/help", "help"}
CLEAR_COMMANDS = {"/clear"}
COMPACT_COMMAND = "/compact"


@dataclass(frozen=True)
class SessionViewState:
    response: str
    todos: list[TodoSnapshot]
    notepad: list[str]
    verification_nudge: str | None


class SessionLike(Protocol):
    def run_turn(self, user_input: str) -> LoopOutcome: ...

    def compact_session(self, focus: str | None = None) -> LoopOutcome: ...

    def reset(self) -> None: ...


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


class TodoCard(Widget):
    def __init__(self, *, index: int, todo: TodoSnapshot) -> None:
        super().__init__(classes=f"todo-card {todo.status}")
        self.index = index
        self.todo = todo

    def compose(self) -> ComposeResult:
        yield Static(
            f"{_todo_icon(self.todo.status)} Step {self.index}",
            classes="todo-kicker",
        )
        yield Static(self.todo.content, classes="todo-content")


class NoteCard(Widget):
    def __init__(self, *, index: int, note: str) -> None:
        super().__init__(classes="note-card")
        self.index = index
        self.note = note

    def compose(self) -> ComposeResult:
        yield Static(f"Note {self.index}", classes="note-kicker")
        yield Markdown(self.note, classes="note-markdown")


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
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="shell"):
            with Vertical(id="conversation-zone"):
                yield Static("Conversation", id="conversation-title")
                with VerticalScroll(id="chat-scroll"):
                    yield ChatCard(
                        role="system",
                        content=(
                            "## Welcome\n\n"
                            "- 普通聊天会直接回复，不会自动当成任务执行。\n"
                            "- 需要整理文件、修改内容或查看目录时，我会进入工作流。\n"
                            "- `/compact` 可主动压缩上下文，"
                            "`/clear` 重置当前会话，`/exit` 结束界面。"
                        ),
                    )
                with Container(id="composer-shell"):
                    yield Static(
                        "Enter 发送 · Shift+Enter 换行 · "
                        "/compact 压缩 · /clear 重置 · /exit 退出",
                        id="composer-help",
                    )
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
                yield Static("Workspace Panels", id="sidebar-title")
                with Vertical(id="todo-panel", classes="side-panel todo-panel"):
                    yield Static("Todo Board", classes="panel-title")
                    with VerticalScroll(id="todo-scroll", classes="panel-scroll"):
                        yield Static(
                            "No active checklist yet.",
                            classes="empty-panel-message",
                        )
                with Vertical(id="notepad-panel", classes="side-panel note-panel"):
                    yield Static("NotePad", classes="panel-title")
                    with VerticalScroll(id="notepad-scroll", classes="panel-scroll"):
                        yield Static(
                            "No saved notes yet.",
                            classes="empty-panel-message",
                        )
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
                    content=(
                        "## Commands\n\n"
                        "- `/help`: show this help card\n"
                        "- `/compact [focus]`: compact the session context\n"
                        "- `/clear`: reset the conversation and side panels\n"
                        "- `/exit`: quit the app"
                    ),
                )
            )
            return
        if user_input in CLEAR_COMMANDS:
            await self.action_clear_session()
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
        await self._mount_chat_card(
            ChatCard(
                role="assistant",
                content=message.outcome.response or "(no response)",
            )
        )
        await self._apply_outcome(
            SessionViewState(
                response=message.outcome.response or "(no response)",
                todos=message.outcome.todos or [],
                notepad=message.outcome.notepad or [],
                verification_nudge=message.outcome.verification_nudge,
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

    async def _begin_turn(self, user_input: str) -> None:
        self._set_busy(True)
        await self._mount_chat_card(ChatCard(role="user", content=user_input))
        pending_card = ChatCard(
            role="thinking",
            content=(
                "### Thinking\n\n"
                "Reviewing the latest message and preparing a reply. "
                "If this turn needs tools or a checklist, they will appear "
                "in the side panels."
            ),
        )
        self._pending_card = pending_card
        await self._mount_chat_card(pending_card)
        self.run_agent_turn(user_input)

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
                "### Compacting\n\n"
                "Compressing the current session context so later turns can "
                "continue with a smaller prompt window."
            ),
        )
        self._pending_card = pending_card
        await self._mount_chat_card(pending_card)
        self.run_compact_turn(focus)

    async def _apply_outcome(self, state: SessionViewState) -> None:
        await self._render_todos(state.todos)
        await self._render_notepad(state.notepad)
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
        await chat_scroll.mount(
            ChatCard(
                role="system",
                content=(
                    "## Welcome\n\n"
                    "- 普通聊天会直接回复，不会自动当成任务执行。\n"
                    "- 需要整理文件、修改内容或查看目录时，我会进入工作流。\n"
                    "- `/compact` 可主动压缩上下文，"
                    "`/clear` 重置当前会话，`/exit` 结束界面。"
                ),
            )
        )
        chat_scroll.scroll_end(animate=False)

    async def _mount_chat_card(self, card: ChatCard) -> None:
        chat_scroll = self.query_one("#chat-scroll", VerticalScroll)
        await chat_scroll.mount(card)
        chat_scroll.scroll_end(animate=False)

    async def _finish_pending_card(self) -> None:
        if self._pending_card is not None:
            await self._pending_card.remove()
            self._pending_card = None

    async def _render_todos(self, todos: list[TodoSnapshot]) -> None:
        todo_scroll = self.query_one("#todo-scroll", VerticalScroll)
        await todo_scroll.remove_children()
        if not todos:
            await todo_scroll.mount(
                Static("No active checklist yet.", classes="empty-panel-message")
            )
        else:
            for index, todo in enumerate(todos, start=1):
                await todo_scroll.mount(TodoCard(index=index, todo=todo))

    async def _render_notepad(self, notes: list[str]) -> None:
        note_scroll = self.query_one("#notepad-scroll", VerticalScroll)
        await note_scroll.remove_children()
        if not notes:
            await note_scroll.mount(
                Static("No saved notes yet.", classes="empty-panel-message")
            )
        else:
            for index, note in enumerate(notes, start=1):
                await note_scroll.mount(NoteCard(index=index, note=note))

    async def _reset_panels(self) -> None:
        await self._render_todos([])
        await self._render_notepad([])
        verification_panel = self.query_one("#verification-panel", Vertical)
        verification_text = self.query_one("#verification-text", Static)
        verification_panel.add_class("hidden")
        verification_text.update("")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        input_widget = self.query_one("#chat-input", ChatComposer)
        help_widget = self.query_one("#composer-help", Static)
        input_widget.disabled = busy
        if busy:
            help_widget.update("正在处理这一轮消息...")
            self.sub_title = f"Textual Session · {self.model} · Running"
        else:
            help_widget.update(
                "Enter 发送 · Shift+Enter 换行 · "
                "/compact 压缩 · /clear 重置 · /exit 退出"
            )
            self.sub_title = f"Textual Session · {self.model}"
            input_widget.focus()

    def _sync_composer_height(self) -> None:
        composer = self.query_one("#chat-input", ChatComposer)
        composer.sync_height()


def run_textual_chat(*, message: str | None, model: str) -> int:
    app = MokioclawTextualApp(model=model, initial_message=message)
    app.run()
    return 0


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


def _todo_icon(status: str) -> str:
    if status == "completed":
        return "[x]"
    if status == "in_progress":
        return "[-]"
    return "[ ]"


def _parse_compact_command(user_input: str) -> str | None:
    if user_input == COMPACT_COMMAND:
        return ""
    if not user_input.startswith(f"{COMPACT_COMMAND} "):
        return None
    focus = user_input[len(COMPACT_COMMAND) :].strip()
    return focus or ""
