# mokio-claw

从最小主干起步的 `mokioclaw` 项目。当前阶段已经从单步 ToolCall 升级为基于 LangChain `create_agent` 的最简 ReAct Agent。

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
└─ tests/
   ├─ test_cli.py
   ├─ test_react_content.py
   ├─ test_provider_env.py
   └─ test_tools.py
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

如果你用的是本地 Ollama，模型名要换成你本机已有的模型，例如：

```bash
uv run mokioclaw "你好" --model qwen3.5:cloud
```

也可以用兼容入口：

```bash
uv run python main.py "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

程序会打印：

- Agent 的 ReAct 执行轨迹
- 实际工具调用步骤
- Memory 快照
- 最终答复

## 开发命令

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ty check
```

## 当前实现

- CLI 层使用 `Typer`
- Prompt 渲染使用 `Jinja2`
- Agent Loop 使用 LangChain `create_agent`
- 环境变量加载使用 `python-dotenv`

## 当前内置工具

- `move_file(src, dst)`：将文件从源路径移动到目标路径
- `file_tree(path, max_depth=3, show_hidden=False)`：获取文件或目录的树状结构
