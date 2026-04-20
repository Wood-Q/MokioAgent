from __future__ import annotations

import pytest

from mokioclaw.tools.file_tools import move_file


def test_move_file_moves_file(tmp_path):
    source = tmp_path / "demo" / "a.txt"
    target = tmp_path / "archive" / "a.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("hello", encoding="utf-8")

    result = move_file(str(source), str(target))

    assert not source.exists()
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello"
    assert "Moved file from" in result


def test_move_file_raises_when_source_missing(tmp_path):
    target = tmp_path / "archive" / "a.txt"

    with pytest.raises(FileNotFoundError):
        move_file(str(tmp_path / "missing.txt"), str(target))
