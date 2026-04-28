from __future__ import annotations

from mokioclaw.harness.approvals import collect_pending_approvals


def test_collect_pending_approvals_blocks_write_tools():
    approval = collect_pending_approvals(
        [
            {
                "id": "call_1",
                "name": "file_write",
                "args": {"path": "demo/report.md", "content": "hello"},
            }
        ],
        approved_tool_call_ids=set(),
    )

    assert approval is not None
    assert approval.tool_calls[0].name == "file_write"
    assert approval.tool_calls[0].risk_level == "high"
    assert "/approve" in approval.message
    assert "/deny" in approval.message


def test_collect_pending_approvals_allows_low_risk_tools():
    approval = collect_pending_approvals(
        [
            {"id": "call_1", "name": "todo_write", "args": {}},
            {"id": "call_2", "name": "file_tree", "args": {"path": "."}},
            {"id": "call_3", "name": "bash", "args": {"command": "ls"}},
        ],
        approved_tool_call_ids=set(),
    )

    assert approval is None


def test_collect_pending_approvals_skips_already_approved_calls():
    approval = collect_pending_approvals(
        [
            {
                "id": "call_1",
                "name": "move_file",
                "args": {"src": "a.txt", "dst": "b.txt"},
            }
        ],
        approved_tool_call_ids={"call_1"},
    )

    assert approval is None
