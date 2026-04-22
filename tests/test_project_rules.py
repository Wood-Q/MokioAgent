from __future__ import annotations

from mokioclaw.core.project_rules import (
    MOKIOCLAW_RULE_MESSAGE_PREFIX,
    discover_project_rule_files,
    load_project_rule_messages,
    load_project_rules,
)


def test_discover_project_rule_files_loads_ancestors_in_order(tmp_path, monkeypatch):
    parent_rules = tmp_path / "mokioclaw.md"
    child_dir = tmp_path / "workspace" / "feature"
    child_dir.mkdir(parents=True)
    child_rules = child_dir / ".mokioclaw" / "mokioclaw.md"
    child_rules.parent.mkdir()

    parent_rules.write_text("Parent rules", encoding="utf-8")
    child_rules.write_text("Child rules", encoding="utf-8")
    monkeypatch.chdir(child_dir)

    discovered = discover_project_rule_files()

    assert discovered == [parent_rules.resolve(), child_rules.resolve()]


def test_load_project_rules_supports_imports_and_strips_comments(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    imported = docs_dir / "style.md"
    imported.write_text("Use snake_case names.", encoding="utf-8")

    rule_file = tmp_path / "mokioclaw.md"
    rule_file.write_text(
        "\n".join(
            [
                "# Rules",
                "<!-- maintainer note should not be loaded -->",
                "See @docs/style.md for naming guidance.",
                "Inline code should not import `@docs/style.md`.",
                "```md",
                "@docs/style.md",
                "```",
            ]
        ),
        encoding="utf-8",
    )

    rules = load_project_rules(cwd=tmp_path)

    assert len(rules) == 1
    content = rules[0].content
    assert "maintainer note" not in content
    assert "Imported from" in content
    assert "Use snake_case names." in content
    assert "`@docs/style.md`" in content
    assert "```md" in content


def test_load_project_rule_messages_wraps_content(tmp_path):
    rule_file = tmp_path / "mokioclaw.md"
    rule_file.write_text("Always run tests before finishing.", encoding="utf-8")

    messages = load_project_rule_messages(cwd=tmp_path)

    assert len(messages) == 1
    content = str(messages[0].content)
    assert content.startswith(MOKIOCLAW_RULE_MESSAGE_PREFIX)
    assert "Always run tests before finishing." in content
