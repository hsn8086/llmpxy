# 配置说明

`llmpxy` 使用 TOML 配置文件。

## 配置结构概览

```toml
[server]
[route]
[network]
[proxy]
[retry]
[logging]
[storage]

[[providers]]
[providers.models]

[[provider_groups]]
```

## 1. route

```toml
[route]
type = "group"
name = "default"
```

- `type`: `provider` 或 `group`
- `name`: 对应 provider 名字或 group 名字

## 2. network

```toml
[network]
proxy = "http://127.0.0.1:7893"
trust_env = false
connect_timeout_seconds = 30.0
read_timeout_seconds = 120.0
write_timeout_seconds = 120.0
pool_timeout_seconds = 30.0
```

- `proxy`: 全局 HTTP 代理
- `trust_env`: 是否信任系统环境变量代理
- `connect_timeout_seconds`: 连接超时
- `read_timeout_seconds`: 读取超时
- `write_timeout_seconds`: 写入超时
- `pool_timeout_seconds`: 连接池超时

## 3. providers

每个 provider 都必须声明自己的协议：

```toml
[[providers]]
name = "openai-chat-a"
protocol = "oaichat"
base_url = "https://provider-a.example.com/v1"
api_key_env = "PROVIDER_A_API_KEY"
timeout_seconds = 120.0
proxy = "http://127.0.0.1:7893"

[providers.models]
"gpt-4.1" = "provider-a-chat-model"
```

支持的 `protocol`：

- `oairesp`
- `oaichat`
- `anthropic`

Anthropic provider 还支持：

```toml
anthropic_version = "2023-06-01"
```

## 4. provider_groups

```toml
[[provider_groups]]
name = "default"
strategy = "fallback"
members = ["openai-chat-a", "anthropic-a"]
```

`members` 可以引用：

- provider 名
- provider group 名

## 5. previous_response_id

当前仅对 `oairesp` 入站接口生效。

也就是：

- `POST /v1/responses` 支持
- `POST /v1/chat/completions` 不支持
- `POST /v1/messages` 不支持

## 6. .env 默认加载

配置加载时会默认尝试读取 `.env`：

- 当前工作目录中的 `.env`
- 配置文件所在目录中的 `.env`

如果同名环境变量已经存在，则不会被 `.env` 覆盖。
