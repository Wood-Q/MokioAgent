from __future__ import annotations

import shutil
from pathlib import Path


def move_file(src: str, dst: str) -> str:
    """Move a file from a source path to a destination path."""

    source = Path(src).expanduser().resolve()
    target = Path(dst).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.is_dir():
        raise IsADirectoryError(f"Source must be a file, not a directory: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return f"Moved file from '{source}' to '{target}'."


def file_tree(path: str = ".", max_depth: int = 3, show_hidden: bool = False) -> str:
    """Return a plain-text directory tree for a file or directory path."""

    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    if max_depth < 0:
        raise ValueError("max_depth must be greater than or equal to 0")

    lines = [_display_name(root)]
    if root.is_file() or max_depth == 0:
        return "\n".join(lines)

    _append_tree_lines(
        root=root,
        lines=lines,
        prefix="",
        current_depth=0,
        max_depth=max_depth,
        show_hidden=show_hidden,
    )
    return "\n".join(lines)


def _append_tree_lines(
    *,
    root: Path,
    lines: list[str],
    prefix: str,
    current_depth: int,
    max_depth: int,
    show_hidden: bool,
) -> None:
    if current_depth >= max_depth:
        return

    children = _iter_children(root, show_hidden=show_hidden)
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{_display_name(child)}")

        if child.is_dir():
            child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            _append_tree_lines(
                root=child,
                lines=lines,
                prefix=child_prefix,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                show_hidden=show_hidden,
            )


def _iter_children(root: Path, *, show_hidden: bool) -> list[Path]:
    children = [
        child
        for child in root.iterdir()
        if show_hidden or not child.name.startswith(".")
    ]
    return sorted(children, key=lambda child: (not child.is_dir(), child.name.lower()))


def _display_name(path: Path) -> str:
    return f"{path.name}/" if path.is_dir() else path.name
