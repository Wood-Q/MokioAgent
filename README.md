# mokio-claw

从最小主干起步的 `mokioclaw` 项目。当前阶段已经从单步 ToolCall 升级为基于 LangGraph 的最简 ReAct Agent。

## 当前能力

- 接收用户自然语言
- 通过 ReAct 方式自行决定是否调用工具
- 允许模型按需多步调用工具
- 保留短期 state 和 memory 结构，便于后续扩展
- 返回执行结果

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
│     │  └─ react_system.jinja2
│     ├─ providers/
│     │  └─ ollama_provider.py
│     ├─ tools/
│     │  ├─ registry.py
│     │  └─ file_tools.py
│     │  └─ workspace_tools.py
└─ tests/
   ├─ test_cli.py
   ├─ test_loop.py
   ├─ test_react_content.py
   ├─ test_provider_env.py
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

也可以不带初始消息，直接进入交互会话：

```bash
uv run mokioclaw
```

如果只想执行一轮然后退出，可以显式使用：

```bash
uv run mokioclaw --one-shot "把 ./demo/a.txt 移动到 ./archive/a.txt"
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

## 开发命令

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ty check
```

## 当前实现

- CLI 层使用 `Typer`
- Prompt 渲染使用 `Jinja2`
- Agent Loop 使用 LangGraph `StateGraph`
- Tool 执行使用 LangGraph `ToolNode`
- 对编辑类工具保存文件读取快照，并在写回前检查过期状态
- 环境变量加载使用 `python-dotenv`

## 当前内置工具

- `move_file(src, dst)`：将文件从源路径移动到目标路径
- `file_tree(path, max_depth=3, show_hidden=False)`：获取文件或目录的树状结构
- `file_edit(path, old_string, new_string, replace_all=False)`：在已读取且未过期的文本文件上安全生成 patch 并写回
- `file_write(path, content, overwrite=False)`：新建文件或整文件覆盖
- `bash(command, cwd=".", timeout_seconds=20)`：执行受限的 search / read / list 类 shell 命令
