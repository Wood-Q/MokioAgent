# mokio-claw

从最小主干起步的 `mokioclaw` 项目。当前阶段已经从最小 ToolCall / ReAct 基线，推进到基于 LangGraph 的单 Agent Plan & Execute 形态。

## 当前能力

- 接收用户自然语言
- 支持 CLI 持续多轮对话，直到输入 `/exit` 或 `/quit`
- 交互模式默认使用 Textual TUI，提供更接近 Claude Code 风格的浅色终端界面
- 简单问候、寒暄、感谢或“你是谁 / 你能做什么”这类输入会直接按聊天处理，不会误触发任务澄清
- 底部输入区默认保持单行高度，输入变成长消息时会按行数小幅自适应增长
- 先由 Planner 生成步骤，或在信息不足时直接提问 / 直接答复
- 澄清问题会明确指出“缺什么信息、该怎么回答”，避免泛泛地说“需要更多信息”
- 由 Executor 聚焦当前步骤，并按需多步调用工具
- 每轮任务都会先建立 Todo 面板，并在步骤推进时自动勾选
- 支持 NotePad，用于外部化中间发现和后续步骤复用的信息
- 由 Finalizer 汇总执行结果，生成对用户的最终答复
- 对复杂任务支持 verification nudge，提醒补充验证步骤
- 内置 loop guard：重复澄清会触发去重提示，graph 递归过深会自动中止本轮执行
- 保留 graph state、短期 memory、文件读取快照，便于继续向 Reflection 演化

## 项目结构

```text
mokio-claw/
├─ README.md
├─ pyproject.toml
├─ .env.example
├─ src/
│  └─ mokioclaw/
│     ├─ main.py
│     ├─ cli/
│     │  └─ app.py
│     ├─ core/
│     │  ├─ loop.py
│     │  ├─ context.py
│     │  ├─ memory.py
│     │  ├─ state.py
│     │  └─ types.py
│     ├─ prompts/
│     │  ├─ react_prompt.py
│     │  ├─ planner_system.jinja2
│     │  ├─ react_system.jinja2
│     │  └─ finalizer_system.jinja2
│     ├─ providers/
│     │  └─ ollama_provider.py
│     ├─ tools/
│     │  ├─ registry.py
│     │  ├─ file_tools.py
│     │  ├─ session_tools.py
│     │  └─ workspace_tools.py
│     └─ tui/
│        ├─ __init__.py
│        ├─ app.py
│        └─ mokioclaw.tcss
└─ tests/
   ├─ test_cli.py
   ├─ test_loop.py
   ├─ test_react_content.py
   ├─ test_provider_env.py
   ├─ test_session_tools.py
   ├─ test_tui.py
   ├─ test_tools.py
   └─ test_workspace_tools.py
```

## 运行方式

1. 配置环境变量

```bash
cp .env.example .env
# 例如本地 Ollama:
# OPENAI_API_KEY=ollama
# BASE_URL=http://localhost:11434
# MODEL=qwen3.5:cloud
```

程序会自动读取项目根目录的 `.env`（使用 `python-dotenv`）。

2. 安装依赖

```bash
uv sync
```

3. 查看 CLI 帮助

```bash
uv run mokioclaw --help
```

4. 运行 CLI

```bash
uv run mokioclaw "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

在终端里默认会进入持续对话模式。Agent 如果需要继续确认信息，会直接追问；你可以继续输入回复，直到输入 `/exit` 或 `/quit` 结束。

当任务进入执行阶段时，Textual 界面会额外展示：

- 对话主面板
- Todo 面板
- Markdown 渲染的 NotePad 卡片
- verification nudge

平时如果只是输入 `你好`、`谢谢`、`你是谁` 这类内容，则会直接正常聊天，不会自动进入任务执行流。

也可以不带初始消息，直接进入交互会话：

```bash
uv run mokioclaw
```

如果只想执行一轮然后退出，可以显式使用：

```bash
uv run mokioclaw --one-shot "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

如果你想回退到原来的纯文本交互模式，可以显式使用：

```bash
uv run mokioclaw --ui plain "帮我整理当前目录"
```

如果你用的是本地 Ollama，模型名要换成你本机已有的模型，例如：

```bash
uv run mokioclaw "你好" --model qwen3.5:cloud
```

也可以用兼容入口：

```bash
uv run python main.py "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

交互模式内置命令：

- `/help`：查看命令
- `/clear`：清空当前会话上下文
- `/exit` / `/quit`：结束会话
- `Enter`：发送消息
- `Shift+Enter`：在输入区换行

## 开发命令

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ty check
```

## 当前编排

当前主循环已经不是单层 ReAct，而是显式的 LangGraph Plan & Execute workflow：

```text
START
  -> planner
  -> executor
  -> tools <-> executor
  -> advance
  -> finalizer
  -> END
```

各阶段职责：

- `planner`：把用户请求转成 1 到 5 个可执行步骤；如果不需要执行或信息不足，则直接返回答复 / 澄清问题
- `planner`：澄清时会返回结构化缺失信息、具体问题、建议回复和默认假设，避免模糊追问
- `executor`：只聚焦当前步骤；如果 Todo 面板为空，会先通过 `todo_write` 建立清单，再执行当前步骤
- `tools`：通过 LangGraph `ToolNode` 执行工具调用
- `advance`：更新 `completed_steps`、`current_step_index`，并自动同步 todo 勾选状态
- `finalizer`：根据计划、已完成步骤和工具结果生成最终答复

在交互层上，当前默认使用 Textual 的 `App + CSS_PATH + Header/Footer + Input + Markdown + Workers` 组合来承载会话界面。
界面样式采用浅色中性底色，橙色主要作为边框和分隔强调；底部 composer 默认是一行，并在多行输入时小幅增高，尽量把主要空间留给对话区。

## 当前实现

- CLI 层使用 `Typer`
- 交互 UI 使用 `Textual`
- Prompt 渲染使用 `Jinja2`，并拆分为 Planner / Executor / Finalizer 三类 prompt
- Agent Loop 使用 LangGraph `StateGraph`
- Tool 执行使用 LangGraph `ToolNode`
- Graph state 中维护 `plan`、`completed_steps`、`current_step_index`、`todos`、`todo_snapshot`、`notepad`、`clarification_attempts`、`last_clarification_signature`、`final_response`、`verification_nudge`、`turn_events`
- 对编辑类工具保存文件读取快照，并在写回前检查过期状态
- 对重复澄清和 graph recursion 提供 loop guard，避免任务卡在无进展状态
- 环境变量加载使用 `python-dotenv`

## 当前内置工具

- `todo_write(todos)`：创建或整体替换当前任务的 Todo 面板
- `notepad_write(note, replace=False)`：追加或替换当前任务的 NotePad
- `move_file(src, dst)`：将文件从源路径移动到目标路径
- `file_tree(path, max_depth=3, show_hidden=False)`：获取文件或目录的树状结构
- `file_edit(path, old_string, new_string, replace_all=False)`：仅允许在“当前 run 已读过且未过期”的文本文件上安全生成 patch 并写回
- `file_write(path, content, overwrite=False)`：新建文件或整文件覆盖
- `bash(command, cwd=".", timeout_seconds=20)`：执行受限的 search / read / list 类 shell 命令，并为读取过的文件记录快照
