# MokioAgent

## 食用方式

此仓库分为两个分支
- main分支用于展示MokioAgent的claw项目推进
- theory分支用于展示Agent的相关理论代码示例

## 当前阶段：单步 ToolCall

这一阶段不是完整 Agent，而是一个最小可运行的“单步工具调用器”：

- 接收用户自然语言
- 判断是否需要调用工具
- 提取工具参数
- 执行一个工具
- 返回结果

核心思想：工具调用本质上是让模型输出结构化字符串（JSON），程序解析后执行。

## 运行方式

1. 配置环境变量：

- `OPENAI_API_KEY`

2. 安装依赖（任选你常用方式，例如 uv/pip）。

3. 执行 CLI：

```bash
claw "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

或：

```bash
python main.py "把 ./demo/a.txt 移动到 ./archive/a.txt"
```

程序会打印：

- 模型原始 ToolCall JSON
- 执行的工具与参数
- 工具执行结果

## 当前内置工具

- `move_file(src, dst)`：将文件从源路径移动到目标路径