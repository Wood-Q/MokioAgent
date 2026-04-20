# mokio-claw

从最小主干起步的 `mokioclaw` 项目。当前阶段实现的是单步 ToolCall，后续会按 Agent Loop、多 Agent、工程化等方向持续演进。

## 当前能力

- 接收用户自然语言
- 让模型判断是否调用工具
- 解析工具参数
- 执行单个工具
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
│     │  ├─ toolcall_decider.py
│     │  └─ types.py
│     ├─ providers/
│     │  └─ ollama_provider.py
│     ├─ tools/
│     │  ├─ registry.py
│     │  └─ file_tools.py
│     ├─ prompts/
│     │  ├─ toolcall.jinja2
│     │  └─ toolcall_prompt.py
│     └─ utils/
│        └─ json_utils.py
└─ tests/
   ├─ test_cli.py
   ├─ test_decider_utils.py
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

- 模型原始 ToolCall JSON
- 执行的工具与参数
- 工具执行结果

## 开发命令

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ty check
```

## 当前实现

- CLI 层使用 `Typer`
- Prompt 渲染使用 `Jinja2`
- 环境变量加载使用 `python-dotenv`

## 当前内置工具

- `move_file(src, dst)`：将文件从源路径移动到目标路径
