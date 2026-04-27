<p align="center">
  <img src="logo.png" alt="Mokioclaw Logo" width="160" />
</p>

<h1 align="center">Mokioclaw</h1>

<p align="center">
  一个从 ToolCall 起步，逐步演化到 Plan & Execute、Context Engineering 与 Mini Claw 产品壳的终端优先 Agent 教学项目。
</p>

<p align="center">
  <strong>ToolCall</strong> · <strong>LangGraph</strong> · <strong>Plan & Execute</strong> · <strong>Textual TUI</strong> · <strong>Context Engineering</strong>
</p>

---

## 项目定位

Mokioclaw 是一个面向 Agent 工程学习与演示的 Python 项目。它不是追求一次性实现“万能助手”，而是以清晰的工程阶段展示一个 CLI Agent 如何从最小 ToolCall 逐步长成更接近 Claude Code / Claw 形态的工作区助手。

当前项目重点聚焦在：

- 用自然语言驱动工作区任务；
- 通过 LangGraph 编排 Planner、Executor、ToolNode 与 Finalizer；
- 使用 Todo / NotePad 把中间状态外部化；
- 用 `/compact` 和自动压缩管理长对话上下文；
- 通过 `mokioclaw.md` 注入项目级规则；
- 使用 Textual 提供更接近 AI Coding 工具的终端交互界面。

---

## 当前能力

### Agent 编排

- 基于 LangGraph `StateGraph` 实现单 Agent Plan & Execute workflow；
- Planner 负责生成 1 到 5 个可执行步骤，或在信息不足时提出结构化澄清；
- Executor 只聚焦当前步骤，并按需调用工具；
- ToolNode 执行文件、工作区、会话类工具；
- Advance 节点推进步骤、同步 Todo 状态；
- Finalizer 汇总计划执行结果，返回最终答复。

当前主流程：

```text
START
  -> planner
  -> executor
  -> tools <-> executor
  -> advance
  -> finalizer
  -> END
```

### Context Engineering

- **Todo Board**：把计划和执行进度外部化；
- **NotePad**：保存跨步骤复用的中间发现；
- **Compaction**：支持 `/compact [focus]` 主动压缩，也支持超过上下文阈值时自动压缩；
- **Project Rules**：启动时读取 `mokioclaw.md` / `.mokioclaw/mokioclaw.md` 等规则文件；
- **Dynamic Tool Selection**：Planner / Executor 按阶段和任务关键词只暴露相关工具子集，减少 prompt 噪声。

### 终端交互

- 默认使用 Textual TUI；
- 支持普通聊天与任务执行分流；
- 支持 Todo、NotePad、Verifier 侧栏；
- Todo 面板支持 active / completed / pending 状态展示；
- 任务完成后 active Todo 会清空，完整快照仍保留给 trace 和 finalizer；
- Composer 在运行中会进入 busy 状态，避免重复提交。

### 内置命令

交互模式支持：

- `/help`：查看命令；
- `/compact [focus]`：压缩当前会话上下文；
- `/clear`：清空当前会话；
- `/exit` / `/quit`：退出交互模式。

---

## 项目结构

```text
mokio-claw/
├─ README.md
├─ logo.png
├─ pyproject.toml
├─ .env.example
├─ main.py
├─ src/
│  └─ mokioclaw/
│     ├─ main.py
│     ├─ cli/
│     │  └─ app.py
│     ├─ core/
│     │  ├─ loop.py
│     │  ├─ context.py
│     │  ├─ memory.py
│     │  ├─ project_rules.py
│     │  ├─ state.py
│     │  └─ types.py
│     ├─ prompts/
│     │  ├─ planner_system.jinja2
│     │  ├─ react_system.jinja2
│     │  ├─ finalizer_system.jinja2
│     │  ├─ compact_system.jinja2
│     │  └─ react_prompt.py
│     ├─ providers/
│     │  └─ ollama_provider.py
│     ├─ tools/
│     │  ├─ registry.py
│     │  ├─ selector.py
│     │  ├─ file_tools.py
│     │  ├─ session_tools.py
│     │  └─ workspace_tools.py
│     └─ tui/
│        ├─ app.py
│        └─ mokioclaw.tcss
└─ tests/
   ├─ test_cli.py
   ├─ test_loop.py
   ├─ test_project_rules.py
   ├─ test_react_content.py
   ├─ test_provider_env.py
   ├─ test_session_tools.py
   ├─ test_tool_selector.py
   ├─ test_tui.py
   ├─ test_tools.py
   └─ test_workspace_tools.py
```

---

## 快速开始

### 1. 安装依赖

项目使用 `uv` 管理 Python 环境与依赖：

```bash
uv sync
```

### 2. 配置模型环境

复制环境变量模板：

```bash
cp .env.example .env
```

本地 Ollama 示例：

```env
OPENAI_API_KEY=ollama
BASE_URL=http://localhost:11434
MODEL=qwen3.5:cloud
MOKIOCLAW_CONTEXT_CHAR_LIMIT=24000
MOKIOCLAW_COMPACT_TAIL_MESSAGES=4
MOKIOCLAW_COMPACT_DEFAULT_FOCUS=优先保留当前任务目标、文件改动、todo 进度和待确认问题
```

### 3. 启动交互式 TUI

```bash
uv run mokioclaw
```

指定模型：

```bash
uv run mokioclaw --model qwen3.5:cloud
```

使用纯文本交互模式：

```bash
uv run mokioclaw --ui plain
```

### 4. 执行一次性任务

```bash
uv run mokioclaw --one-shot "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

兼容入口：

```bash
uv run python main.py --one-shot "帮我整理当前目录"
```

---

## 内置工具

| 工具 | 说明 |
| --- | --- |
| `todo_write` | 创建或整体替换当前任务 Todo 面板 |
| `notepad_write` | 追加或替换当前任务 NotePad |
| `move_file` | 移动文件 |
| `file_tree` | 获取文件或目录树结构 |
| `file_edit` | 在已读取且未过期的文本文件上安全替换内容 |
| `file_write` | 新建文件或整文件覆盖 |
| `bash` | 执行受限的 search / read / list 类 shell 命令 |

工具不会全部无差别暴露给模型。当前通过 `tools/selector.py` 做阶段过滤与规则过滤：

- Planner 只看到和用户请求相关的工具摘要；
- Executor 默认保留 `todo_write` / `notepad_write`；
- 文件、搜索、写入、移动等工具按当前步骤关键词动态加入；
- Finalizer 不暴露工具。

---

## 项目规则：`mokioclaw.md`

Mokioclaw 会从当前工作目录向上递归查找以下规则文件：

- `mokioclaw.md`
- `MOKIOCLAW.md`
- `.mokioclaw/mokioclaw.md`
- `.mokioclaw/MOKIOCLAW.md`

规则会按“越上层越先、越具体越后”的顺序注入模型上下文，适合存放：

- 构建、测试、lint 命令；
- 项目代码风格；
- 架构边界；
- 团队工作流约束。

支持类似 Claude Code `CLAUDE.md` 的 `@path` 导入能力：

```md
# Project Rules

- Always run `uv run --group dev pytest` after non-trivial Python changes.
- Prefer small, focused changes.

See @README.md for project overview.
```

---

## 开发命令

运行测试：

```bash
uv run --group dev pytest
```

运行 Ruff：

```bash
uv run --group dev ruff check .
```

运行类型检查：

```bash
uv run --group dev ty check
```

---

## 教学路线

Mokioclaw 对应一条从基础到工程化的 Agent 学习路线：

1. 最小 ToolCall：自然语言到一次工具调用；
2. Agent Loop / ReAct：从一次调用扩展到循环执行；
3. LangGraph 编排：用节点和边表达执行流；
4. Reflection / Plan & Execute：引入审查、规划与执行分离；
5. Context Engineering：Todo、NotePad、压缩、规则注入；
6. Harness Engineering：权限、审批、恢复、追踪、沙箱；
7. Skills / MultiAgent：能力包与多代理协作；
8. Mini Claw：最终形成可扩展的终端产品壳。

当前代码已经推进到 **LangGraph Plan & Execute + Context Engineering 基础能力 + Textual TUI** 阶段。

---

## 设计原则

- **先静态注册，再动态暴露**：工具全集由代码声明，模型只在每轮看到相关子集；
- **状态外部化**：把计划、Todo、NotePad、压缩摘要从消息历史中拆出来；
- **小步演进**：每一阶段只引入一个关键抽象；
- **教学优先**：实现保持可读、可讲解，避免过早引入复杂框架或隐藏魔法；
- **安全边界前置**：文件修改、shell 执行、项目规则都保留明确边界。

---

## License

This project is currently intended for personal learning, experimentation, and course demonstration.
