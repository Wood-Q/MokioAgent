from __future__ import annotations

import pytest

from mokioclaw.tools.file_tools import file_tree, move_file


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


def test_file_tree_returns_directory_structure(tmp_path):
    root = tmp_path / "demo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a.txt").write_text("a", encoding="utf-8")
    (nested / "b.txt").write_text("b", encoding="utf-8")

    result = file_tree(str(root), max_depth=3)

    assert "demo/" in result
    assert "nested/" in result
    assert "a.txt" in result
    assert "b.txt" in result


def test_file_tree_hides_hidden_entries_by_default(tmp_path):
    root = tmp_path / "demo"
    root.mkdir()
    (root / ".secret").write_text("x", encoding="utf-8")
    (root / "visible.txt").write_text("y", encoding="utf-8")

    hidden_result = file_tree(str(root), show_hidden=False)
    shown_result = file_tree(str(root), show_hidden=True)

    assert ".secret" not in hidden_result
    assert ".secret" in shown_result
