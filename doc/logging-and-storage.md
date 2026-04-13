# 日志、代理与存储

## 1. 日志

日志仍然分为：

- `info.log`
- `debug.log`

并带有：

- `request_id`
- `provider`
- `round`
- `attempt`

## 2. HTTP 代理

支持两级配置：

1. 全局代理：`[network].proxy`
2. provider 级代理：`providers[].proxy`

优先级：

- `providers[].proxy`
- `[network].proxy`
- 无代理

## 3. 会话存储

当前支持：

- `sqlite`
- `file`

## 4. previous_response_id 的作用域

只有 `oairesp` 入口支持服务端会话续聊。

即：

- `/v1/responses`: 支持
- `/v1/chat/completions`: 不支持
- `/v1/messages`: 不支持
