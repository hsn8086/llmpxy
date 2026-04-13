# API 使用说明

`llmpxy` 对外暴露 3 个主要协议入口。

## 1. OpenAI Responses

路径：`POST /v1/responses`

示例：

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "介绍一下 FastAPI"
  }'
```

`previous_response_id` 仅这个入口支持。

## 2. OpenAI Chat Completions

路径：`POST /v1/chat/completions`

示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "messages": [
      {"role": "user", "content": "介绍一下 FastAPI"}
    ]
  }'
```

## 3. Anthropic Messages

路径：`POST /v1/messages`

示例：

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "介绍一下 FastAPI"}]}
    ],
    "max_tokens": 256
  }'
```

## 4. 协议间转发

客户端请求协议和上游 provider 协议可以不同。

例如：

- 客户端打 `oairesp`
- 最终路由到 `anthropic` provider

或者：

- 客户端打 `anthropic`
- 最终路由到 `oaichat` provider

## 5. 流式输出

3 个入口都支持流式，底层会根据入站协议返回对应风格的 SSE 事件。
