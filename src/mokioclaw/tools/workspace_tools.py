from __future__ import annotations

import difflib
import hashlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from mokioclaw.core.state import FileSnapshot, MokioclawState

BASH_SEARCH_COMMANDS = {"find", "grep", "rg", "ag", "ack", "locate"}
BASH_READ_COMMANDS = {"cat", "head", "tail", "less", "more"}
BASH_LIST_COMMANDS = {"ls", "tree", "du"}
ALLOWED_BASH_COMMANDS = (
    BASH_SEARCH_COMMANDS | BASH_READ_COMMANDS | BASH_LIST_COMMANDS
)
UNSAFE_SHELL_SEQUENCES = ("&&", "||", ";", "|", ">", "<", "$(", "`", "\n", "\r")
INTERACTIVE_BASH_COMMANDS = {"less", "more"}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path) -> str:
    root = _workspace_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve_workspace_path(path: str, *, cwd: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (cwd or _workspace_root()) / candidate
    resolved = candidate.resolve()
    _ensure_within_workspace(resolved)
    return resolved


def _ensure_within_workspace(path: Path) -> None:
    root = _workspace_root().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path is outside the workspace root: {_display_path(path)}"
        ) from exc


def _ensure_existing_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {_display_path(path)}")
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected a file, got a directory: {_display_path(path)}"
        )


def _ensure_writable(path: Path) -> None:
    target = path if path.exists() else path.parent
    if not os.access(target, os.W_OK):
        raise PermissionError(f"Write access denied: {_display_path(target)}")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "File is not valid UTF-8 text and cannot be edited safely: "
            f"{_display_path(path)}"
        ) from exc


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _snapshot_file(path: Path, *, source: str) -> FileSnapshot:
    data = _read_bytes(path)
    stat = path.stat()
    return FileSnapshot(
        sha256=hashlib.sha256(data).hexdigest(),
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
        source=source,
    )


def _snapshot_matches(path: Path, snapshot: FileSnapshot) -> bool:
    if not path.exists():
        return False
    current = _snapshot_file(path, source=snapshot["source"])
    return (
        current["sha256"] == snapshot["sha256"]
        and current["mtime_ns"] == snapshot["mtime_ns"]
        and current["size"] == snapshot["size"]
    )


def _truncate_text(text: str, *, max_chars: int = 4000, max_lines: int = 120) -> str:
    lines = text.splitlines()
    truncated_lines = lines[:max_lines]
    clipped = "\n".join(truncated_lines)
    if len(lines) > max_lines:
        clipped += f"\n... ({len(lines) - max_lines} more lines omitted)"
    if len(clipped) > max_chars:
        clipped = clipped[: max_chars - 40] + "\n... (output truncated)"
    return clipped


def _build_diff(
    old_content: str,
    new_content: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    diff_lines = list(
        difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
    if not diff_lines:
        return "No textual changes."
    return _truncate_text("\n".join(diff_lines), max_chars=5000, max_lines=160)


def _build_tool_command(
    *,
    tool_call_id: str,
    content: str,
    state_update: dict[str, object] | None = None,
) -> Command:
    update = dict(state_update or {})
    update["messages"] = [ToolMessage(content=content, tool_call_id=tool_call_id)]
    return Command(update=update)


def _require_tool_call_id(runtime: ToolRuntime) -> str:
    if runtime.tool_call_id is None:
        raise ValueError("Tool runtime is missing a tool_call_id.")
    return runtime.tool_call_id


def _looks_like_path(token: str, *, cwd: Path) -> bool:
    if not token or token == "-" or token.startswith("-"):
        return False
    if token.startswith(("~", "/", "./", "../")):
        return True
    if "/" in token:
        return True
    return (cwd / token).exists()


def _collect_workspace_paths(tokens: list[str], *, cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for token in tokens:
        if not _looks_like_path(token, cwd=cwd):
            continue
        paths.append(_resolve_workspace_path(token, cwd=cwd))
    return paths


def _bash_category(program: str) -> str:
    if program in BASH_SEARCH_COMMANDS:
        return "search"
    if program in BASH_READ_COMMANDS:
        return "read"
    return "list"


def _describe_existing_status(path: Path) -> str:
    if not path.exists():
        return "new file"
    stat = path.stat()
    return f"existing file ({stat.st_size} bytes)"


@tool("file_write")
def file_write(
    path: str,
    content: str,
    runtime: ToolRuntime,
    overwrite: bool = False,
) -> Command:
    """Create a new text file or fully overwrite an existing text file."""

    target = _resolve_workspace_path(path)
    existed = target.exists()

    if existed and target.is_dir():
        raise IsADirectoryError(f"Expected a file path, got a directory: {path}")
    if existed and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file without overwrite=True: {path}"
        )

    status_before = _describe_existing_status(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    _ensure_writable(target)

    old_content = _read_text(target) if existed else ""
    target.write_text(content, encoding="utf-8")
    snapshot = _snapshot_file(target, source="write")
    diff = _build_diff(
        old_content,
        content,
        fromfile="/dev/null" if not existed else _display_path(target),
        tofile=_display_path(target),
    )

    summary = "\n".join(
        [
            f"file_write completed for {_display_path(target)}",
            f"mode: {'overwrite' if existed else 'create'}",
            f"status_before: {status_before}",
            f"bytes_written: {len(content.encode('utf-8'))}",
            "diff:",
            diff,
        ]
    )
    return _build_tool_command(
        tool_call_id=_require_tool_call_id(runtime),
        content=summary,
        state_update={"file_snapshots": {str(target): snapshot}},
    )


@tool("file_edit")
def file_edit(
    path: str,
    old_string: str,
    new_string: str,
    runtime: ToolRuntime,
    replace_all: bool = False,
) -> Command:
    """Safely edit an existing text file after it has been read in the current run."""

    target = _resolve_workspace_path(path)
    _ensure_existing_file(target)
    _ensure_writable(target)

    if not old_string:
        raise ValueError("old_string must not be empty.")
    if old_string == new_string:
        raise ValueError("old_string and new_string are identical; no edit is needed.")

    state = runtime.state or MokioclawState(messages=[])
    snapshots = state.get("file_snapshots", {})
    snapshot = snapshots.get(str(target))
    if snapshot is None or snapshot["source"] not in {"read", "edit"}:
        raise ValueError(
            "FileEditTool requires a fresh read snapshot. Read the file first "
            "with bash using cat/head/tail, then retry the edit."
        )
    if not _snapshot_matches(target, snapshot):
        raise ValueError(
            "The file changed after it was read. Read it again before applying an edit."
        )

    current_content = _read_text(target)
    occurrence_count = current_content.count(old_string)
    if occurrence_count == 0:
        raise ValueError(
            "old_string was not found in the current file content. Re-read the file "
            "and make the edit target more specific."
        )
    if occurrence_count > 1 and not replace_all:
        raise ValueError(
            "old_string matched multiple locations. Provide a more specific string or "
            "set replace_all=True."
        )

    updated_content = (
        current_content.replace(old_string, new_string)
        if replace_all
        else current_content.replace(old_string, new_string, 1)
    )
    target.write_text(updated_content, encoding="utf-8")
    new_snapshot = _snapshot_file(target, source="edit")
    diff = _build_diff(
        current_content,
        updated_content,
        fromfile=_display_path(target),
        tofile=_display_path(target),
    )

    summary = "\n".join(
        [
            f"file_edit completed for {_display_path(target)}",
            "checks:",
            "- path and workspace permission: ok",
            f"- prior read snapshot: ok ({snapshot['source']})",
            "- file unchanged since last read: ok",
            f"- replacements_applied: {'all' if replace_all else 1}",
            "diff:",
            diff,
        ]
    )
    return _build_tool_command(
        tool_call_id=_require_tool_call_id(runtime),
        content=summary,
        state_update={"file_snapshots": {str(target): new_snapshot}},
    )


@tool("bash")
def bash(
    command: str,
    runtime: ToolRuntime,
    cwd: str = ".",
    timeout_seconds: int = 20,
) -> Command:
    """Execute a workspace-scoped search/read/list shell command in the foreground."""

    stripped_command = command.strip()
    if not stripped_command:
        raise ValueError("Command must not be empty.")
    if any(item in stripped_command for item in UNSAFE_SHELL_SEQUENCES):
        raise ValueError(
            "Only a single direct command is allowed. Shell control operators "
            "are not supported."
        )

    working_dir = _resolve_workspace_path(cwd, cwd=_workspace_root())
    if not working_dir.exists() or not working_dir.is_dir():
        raise NotADirectoryError(f"Working directory not found: {cwd}")

    parts = shlex.split(stripped_command)
    if not parts:
        raise ValueError("Command must not be empty.")

    program = parts[0]
    if program not in ALLOWED_BASH_COMMANDS:
        raise ValueError(
            f"Unsupported bash command '{program}'. Allowed commands: "
            f"{', '.join(sorted(ALLOWED_BASH_COMMANDS))}."
        )
    if program in INTERACTIVE_BASH_COMMANDS:
        raise ValueError(
            f"Interactive command '{program}' is not supported. Use "
            "cat/head/tail instead."
        )

    workspace_paths = _collect_workspace_paths(parts[1:], cwd=working_dir)
    if _bash_category(program) == "read" and not workspace_paths:
        raise ValueError(
            "Read commands must include an explicit file path inside the workspace."
        )

    try:
        completed = subprocess.run(
            parts,
            cwd=working_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Command is not available in the environment: {program}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"Command timed out after {timeout_seconds} seconds: {stripped_command}"
        ) from exc

    snapshot_update: dict[str, FileSnapshot] = {}
    if completed.returncode == 0 and _bash_category(program) == "read":
        for path_obj in workspace_paths:
            if path_obj.exists() and path_obj.is_file():
                snapshot_update[str(path_obj)] = _snapshot_file(
                    path_obj,
                    source="read",
                )

    lines = [
        f"bash completed: {stripped_command}",
        f"category: {_bash_category(program)}",
        f"cwd: {_display_path(working_dir)}",
        "execution_mode: foreground",
        f"exit_code: {completed.returncode}",
    ]
    if snapshot_update:
        tracked = ", ".join(
            sorted(_display_path(Path(path)) for path in snapshot_update)
        )
        lines.append(f"tracked_reads: {tracked}")
    if completed.stdout:
        lines.extend(["stdout:", _truncate_text(completed.stdout)])
    if completed.stderr:
        lines.extend(["stderr:", _truncate_text(completed.stderr)])
    if not completed.stdout and not completed.stderr:
        lines.append("(no output)")

    state_update: dict[str, object] = {}
    if snapshot_update:
        state_update["file_snapshots"] = snapshot_update
    return _build_tool_command(
        tool_call_id=_require_tool_call_id(runtime),
        content="\n".join(lines),
        state_update=state_update,
    )


file_write_impl = cast(Any, file_write).func
file_edit_impl = cast(Any, file_edit).func
bash_impl = cast(Any, bash).func
