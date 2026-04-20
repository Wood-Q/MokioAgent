from __future__ import annotations

import shutil
from pathlib import Path


def move_file(src: str, dst: str) -> str:
    source = Path(src).expanduser().resolve()
    target = Path(dst).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.is_dir():
        raise IsADirectoryError(f"Source must be a file, not a directory: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return f"Moved file from '{source}' to '{target}'."

