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
- 支持入站 API key 鉴权与总额度 / provider / group 额度控制
- 支持 provider 默认定价和分模型定价
- 支持配置文件变更后自动校验并热重载
- 支持独立 admin token 的远程管理 API
- 支持 remote CLI 和实时 dashboard 流
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

# 开发时可开启自动重载
uv run llmpxy serve --config ./config.toml --reload
```

上游 provider 密钥也可以写到 `.env` 文件中，`llmpxy` 会默认读取当前工作目录或配置文件同目录下的 `.env`。

服务请求需要携带 `Authorization: Bearer <your-client-key>`，其中这个 client key 直接配置在 `config.toml` 的 `[[api_keys]]` 里，并且每个 key 都需要配置一个稳定 `uuid`；配额和计费规则见 `config.example.toml`。

远程管理使用独立 `admin.token`，例如：

```bash
uv run llmpxy remote status --base-url http://127.0.0.1:8080 --admin-token your-admin-token
uv run llmpxy remote dashboard --base-url http://127.0.0.1:8080 --admin-token your-admin-token
```
