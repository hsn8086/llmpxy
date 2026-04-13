# llmpxy

一个多协议 LLM 代理，支持 OpenAI `Responses`、OpenAI `Chat Completions` 和 Anthropic `Messages` 之间的统一接入与转发。

## 功能

- 暴露 `POST /v1/responses`
- 暴露 `POST /v1/chat/completions`
- 暴露 `POST /v1/messages`
- 支持 `oairesp`、`oaichat`、`anthropic` 三种协议入站和出站
- 支持 provider 和 provider group
- 支持 `fallback` 和 `load_balance`
- 支持 provider 连续失败切换和指数退避
- 支持 SQLite / 文件会话存储
- `previous_response_id` 仅对 `oairesp` 生效
- 支持 HTTP 代理

## 文档

完整文档见 `doc/` 目录。

仓库协作约定见：

- `docs/git-workflow.md`
- `agent.md`

## 启动

```bash
uv sync
export PROVIDER_A_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
uv run llmpxy serve --config ./config.toml
```

也可以把这些密钥写到 `.env` 文件中，`llmpxy` 会默认读取当前工作目录或配置文件同目录下的 `.env`。
