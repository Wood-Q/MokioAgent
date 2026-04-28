from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

RiskLevel = Literal["medium", "high"]

APPROVAL_REQUIRED_TOOLS: dict[str, tuple[RiskLevel, str]] = {
    "move_file": ("medium", "Moving files changes workspace structure."),
    "file_edit": ("high", "Editing files changes existing workspace content."),
    "file_write": ("high", "Writing files can create or overwrite workspace content."),
}


@dataclass(frozen=True)
class ApprovalToolCall:
    id: str
    name: str
    args: dict[str, Any]
    risk_level: RiskLevel
    reason: str


@dataclass(frozen=True)
class PendingApproval:
    id: str
    tool_calls: list[ApprovalToolCall]
    message: str


def collect_pending_approvals(
    tool_calls: list[dict[str, Any]],
    approved_tool_call_ids: set[str],
) -> PendingApproval | None:
    approval_calls: list[ApprovalToolCall] = []
    for index, tool_call in enumerate(tool_calls, start=1):
        call_id = str(tool_call.get("id") or f"call_{index}")
        if call_id in approved_tool_call_ids:
            continue

        name = str(tool_call.get("name", ""))
        risk = APPROVAL_REQUIRED_TOOLS.get(name)
        if risk is None:
            continue

        args = tool_call.get("args")
        approval_calls.append(
            ApprovalToolCall(
                id=call_id,
                name=name,
                args=args if isinstance(args, dict) else {},
                risk_level=risk[0],
                reason=risk[1],
            )
        )

    if not approval_calls:
        return None

    approval_id = "+".join(call.id for call in approval_calls)
    return PendingApproval(
        id=approval_id,
        tool_calls=approval_calls,
        message=format_pending_approval_message(approval_calls),
    )


def format_pending_approval_message(tool_calls: list[ApprovalToolCall]) -> str:
    lines = [
        "Human approval required before executing this action.",
        "",
        "Pending tool calls:",
    ]
    for index, tool_call in enumerate(tool_calls, start=1):
        lines.extend(
            [
                f"{index}. {tool_call.name} ({tool_call.risk_level} risk)",
                f"   reason: {tool_call.reason}",
                f"   args: {json.dumps(tool_call.args, ensure_ascii=False)}",
            ]
        )
    lines.extend(
        [
            "",
            "Reply with /approve to continue, or /deny to cancel this action.",
        ]
    )
    return "\n".join(lines)


def approval_to_state(approval: PendingApproval) -> dict[str, object]:
    return {
        "id": approval.id,
        "message": approval.message,
        "tool_calls": [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "args": tool_call.args,
                "risk_level": tool_call.risk_level,
                "reason": tool_call.reason,
            }
            for tool_call in approval.tool_calls
        ],
    }


def approval_from_state(state: dict[str, Any]) -> PendingApproval:
    tool_calls = [
        ApprovalToolCall(
            id=str(tool_call.get("id", "")),
            name=str(tool_call.get("name", "")),
            args=(
                tool_call.get("args")
                if isinstance(tool_call.get("args"), dict)
                else {}
            ),
            risk_level=_coerce_risk_level(tool_call.get("risk_level")),
            reason=str(tool_call.get("reason", "")),
        )
        for tool_call in state.get("tool_calls", [])
        if isinstance(tool_call, dict)
    ]
    return PendingApproval(
        id=str(state.get("id", "")),
        tool_calls=tool_calls,
        message=str(state.get("message", format_pending_approval_message(tool_calls))),
    )


def _coerce_risk_level(value: object) -> RiskLevel:
    return "medium" if value == "medium" else "high"
