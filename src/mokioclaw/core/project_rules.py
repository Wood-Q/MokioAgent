from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage

MOKIOCLAW_RULE_MESSAGE_PREFIX = "[Mokioclaw Project Instructions]"
MAX_RULE_IMPORT_DEPTH = 5
RULE_CANDIDATES = (
    "mokioclaw.md",
    "MOKIOCLAW.md",
    ".mokioclaw/mokioclaw.md",
    ".mokioclaw/MOKIOCLAW.md",
)


@dataclass(frozen=True)
class LoadedProjectRule:
    path: Path
    content: str


def load_project_rule_messages(cwd: Path | None = None) -> list[HumanMessage]:
    messages: list[HumanMessage] = []
    for rule in load_project_rules(cwd=cwd):
        messages.append(
            HumanMessage(
                content=(
                    f"{MOKIOCLAW_RULE_MESSAGE_PREFIX}\n"
                    f"Source: {rule.path}\n\n"
                    f"{rule.content}"
                )
            )
        )
    return messages


def load_project_rules(cwd: Path | None = None) -> list[LoadedProjectRule]:
    rules: list[LoadedProjectRule] = []
    for path in discover_project_rule_files(cwd=cwd):
        content = _render_rule_file(path=path, depth=0, stack=())
        if not content.strip():
            continue
        rules.append(LoadedProjectRule(path=path, content=content.strip()))
    return rules


def discover_project_rule_files(cwd: Path | None = None) -> list[Path]:
    current = (cwd or Path.cwd()).resolve()
    directories: list[Path] = []

    while True:
        directories.append(current)
        if current.parent == current:
            break
        current = current.parent

    if directories and directories[-1].parent == directories[-1]:
        directories.pop()
    directories.reverse()

    discovered: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        for relative in RULE_CANDIDATES:
            candidate = _resolve_case_sensitive_child(directory, relative)
            if candidate is None or candidate in seen:
                continue
            discovered.append(candidate)
            seen.add(candidate)
    return discovered


def _render_rule_file(
    *,
    path: Path,
    depth: int,
    stack: tuple[Path, ...],
) -> str:
    resolved = path.expanduser().resolve()
    if resolved in stack or depth > MAX_RULE_IMPORT_DEPTH:
        return ""

    text = resolved.read_text(encoding="utf-8")
    lines = text.splitlines()
    rendered: list[str] = []
    in_code_block = False
    in_html_comment = False

    for line in lines:
        stripped = line.lstrip()
        fence = stripped.startswith("```") or stripped.startswith("~~~")
        if fence and not in_html_comment:
            in_code_block = not in_code_block
            rendered.append(line)
            continue

        if in_code_block:
            rendered.append(line)
            continue

        if in_html_comment:
            if "-->" in line:
                in_html_comment = False
            continue

        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_html_comment = True
            continue

        rendered.append(line)
        for token in _find_import_tokens(line):
            imported = _load_import_from_token(
                token=token,
                base_dir=resolved.parent,
                depth=depth + 1,
                stack=(*stack, resolved),
            )
            if not imported:
                continue
            rendered.extend(["", imported, ""])

    return "\n".join(rendered).strip()


def _find_import_tokens(line: str) -> list[str]:
    tokens: list[str] = []
    in_inline_code = False
    index = 0

    while index < len(line):
        char = line[index]
        if char == "`":
            in_inline_code = not in_inline_code
            index += 1
            continue
        if (
            not in_inline_code
            and char == "@"
            and (index == 0 or line[index - 1].isspace() or line[index - 1] in "([{-:")
        ):
            end = index + 1
            while (
                end < len(line)
                and not line[end].isspace()
                and line[end] not in ",;)]}>"
            ):
                end += 1
            token = line[index + 1 : end].strip()
            if token:
                tokens.append(token)
            index = end
            continue
        index += 1

    return tokens


def _load_import_from_token(
    *,
    token: str,
    base_dir: Path,
    depth: int,
    stack: tuple[Path, ...],
) -> str:
    path = _resolve_import_path(token=token, base_dir=base_dir)
    if path is None or not path.is_file():
        return ""

    rendered = _render_rule_file(path=path, depth=depth, stack=stack)
    if not rendered:
        return ""
    return f"Imported from {path}:\n{rendered}"


def _resolve_import_path(token: str, base_dir: Path) -> Path | None:
    raw_path = Path(token).expanduser()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.append((base_dir / raw_path).resolve())

    if raw_path.suffix == "":
        expanded: list[Path] = []
        for candidate in candidates:
            expanded.append(candidate.with_suffix(".md"))
            expanded.append(candidate.with_suffix(".txt"))
        candidates.extend(expanded)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _resolve_case_sensitive_child(directory: Path, relative: str) -> Path | None:
    current = directory
    for part in Path(relative).parts:
        try:
            children = {child.name: child for child in current.iterdir()}
        except FileNotFoundError:
            return None
        next_path = children.get(part)
        if next_path is None:
            return None
        current = next_path

    if current.is_file():
        return current.resolve()
    return None
