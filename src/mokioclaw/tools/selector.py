from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool

from mokioclaw.core.state import MokioclawState
from mokioclaw.tools.registry import tools_for_agent, tools_for_prompt

ToolPhase = Literal["planner", "executor"]


@dataclass(frozen=True)
class ToolSelectionRule:
    phases: frozenset[ToolPhase]
    keywords: frozenset[str]
    path_sensitive: bool = False


EXECUTOR_BASE_TOOL_NAMES = ("todo_write", "notepad_write")

PATH_MARKERS = frozenset(
    {
        "./",
        "../",
        ".txt",
        ".md",
        ".py",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        "readme",
        "src",
        "tests",
        "demo",
        "archive",
        "workspace",
        "repo",
        "path",
        "directory",
        "folder",
        "file",
        "目录",
        "文件夹",
        "文件",
        "路径",
    }
)

TOOL_SELECTION_RULES: dict[str, ToolSelectionRule] = {
    "file_tree": ToolSelectionRule(
        phases=frozenset({"planner", "executor"}),
        keywords=frozenset(
            {
                "tree",
                "list",
                "ls",
                "inspect",
                "scan",
                "查看",
                "检查",
                "扫描",
                "列出",
                "结构",
                "目录树",
            }
        ),
    ),
    "move_file": ToolSelectionRule(
        phases=frozenset({"planner", "executor"}),
        keywords=frozenset(
            {
                "move",
                "rename",
                "put",
                "place",
                "organize",
                "classify",
                "移动",
                "搬移",
                "迁移",
                "放进",
                "放到",
                "放入",
                "重命名",
                "改名",
                "归档",
                "整理",
                "归类",
                "分类",
            }
        ),
    ),
    "file_edit": ToolSelectionRule(
        phases=frozenset({"planner", "executor"}),
        keywords=frozenset(
            {
                "edit",
                "replace",
                "update",
                "fix",
                "patch",
                "modify",
                "修改",
                "编辑",
                "替换",
                "更新",
                "修复",
                "补充",
                "调整",
                "改成",
            }
        ),
    ),
    "file_write": ToolSelectionRule(
        phases=frozenset({"planner", "executor"}),
        keywords=frozenset(
            {
                "write",
                "create",
                "generate",
                "report",
                "summary",
                "new file",
                "写入",
                "创建",
                "新建",
                "生成",
                "报告",
                "输出",
            }
        ),
    ),
    "bash": ToolSelectionRule(
        phases=frozenset({"planner", "executor"}),
        keywords=frozenset(
            {
                "find",
                "grep",
                "rg",
                "cat",
                "head",
                "tail",
                "search",
                "read",
                "list",
                "搜索",
                "查找",
                "读取",
                "查看",
                "检查",
                "复查",
                "核对",
                "验证",
                "列出",
                "内容",
                "分析",
            }
        ),
    ),
}


def select_prompt_tools_for_planner(user_input: str) -> list[dict[str, object]]:
    return _prompt_tools_by_name(_matching_tool_names("planner", user_input))


def select_prompt_tools_for_executor(
    state: MokioclawState,
    current_step: str,
) -> list[dict[str, object]]:
    return _prompt_tools_by_name(select_tool_names_for_executor(state, current_step))


def select_agent_tools_for_executor(
    state: MokioclawState,
    current_step: str,
) -> list[BaseTool]:
    return _agent_tools_by_name(select_tool_names_for_executor(state, current_step))


def select_tool_names_for_executor(
    state: MokioclawState,
    current_step: str,
) -> list[str]:
    selected = set(EXECUTOR_BASE_TOOL_NAMES)
    step_matches = set(_matching_tool_names("executor", current_step))
    selected.update(step_matches)

    if not step_matches and _is_generic_execution_step(current_step):
        selected.update(_matching_tool_names("executor", state.get("user_input", "")))

    return _ordered_tool_names(selected)


def _is_generic_execution_step(text: str) -> bool:
    normalized = text.casefold().strip()
    generic_markers = (
        "执行当前动作",
        "执行当前步骤",
        "完成当前动作",
        "完成当前步骤",
        "do the current action",
        "run the current step",
    )
    return any(marker in normalized for marker in generic_markers)


def _matching_tool_names(phase: ToolPhase, text: str) -> list[str]:
    selected: set[str] = set()
    for name, rule in TOOL_SELECTION_RULES.items():
        if phase not in rule.phases:
            continue
        if _contains_keyword(text, rule.keywords) or (
            rule.path_sensitive and _looks_path_related(text)
        ):
            selected.add(name)
    return _ordered_tool_names(selected)


def _contains_keyword(text: str, keywords: frozenset[str]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)


def _looks_path_related(text: str) -> bool:
    normalized = text.casefold()
    if any(marker in normalized for marker in PATH_MARKERS):
        return True
    return "/" in text or "\\" in text


def _ordered_tool_names(selected: set[str]) -> list[str]:
    prompt_order = [
        str(tool["name"])
        for tool in tools_for_prompt()
        if isinstance(tool.get("name"), str)
    ]
    return [name for name in prompt_order if name in selected]


def _prompt_tools_by_name(names: list[str]) -> list[dict[str, object]]:
    by_name = {
        str(tool["name"]): tool
        for tool in tools_for_prompt()
        if isinstance(tool.get("name"), str)
    }
    return [by_name[name] for name in names if name in by_name]


def _agent_tools_by_name(names: list[str]) -> list[BaseTool]:
    by_name = {tool.name: tool for tool in tools_for_agent()}
    return [by_name[name] for name in names if name in by_name]
