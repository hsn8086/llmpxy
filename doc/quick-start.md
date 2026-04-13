# 快速开始

本文档用于最短路径启动 `llmpxy`。

## 1. 安装依赖

```bash
uv sync
```

## 2. 准备配置文件

```bash
cp config.example.toml config.toml
```

## 3. 设置环境变量

例如：

```bash
export PROVIDER_A_API_KEY="your-openai-like-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

也可以直接写入 `.env` 文件，例如：

```dotenv
PROVIDER_A_API_KEY=your-openai-like-key
ANTHROPIC_API_KEY=your-anthropic-key
```

`llmpxy` 会默认尝试读取：

- 当前工作目录下的 `.env`
- 配置文件所在目录下的 `.env`

## 4. 启动服务

```bash
uv run llmpxy serve --config ./config.toml
```

## 5. 调用 OpenAI Responses 入口

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "你好，介绍一下你自己"
  }'
```

## 6. 调用 OpenAI Chat 入口

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

## 7. 调用 Anthropic 入口

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "你好"}]}
    ],
    "max_tokens": 256
  }'
```

## 8. 健康检查

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/readyz
```
