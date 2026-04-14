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
[admin]
[storage]

[[api_keys]]

[[providers]]
[providers.models]
[providers.pricing.default]
[providers.pricing.models.<model>]

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

[providers.pricing.default]
input_per_million_tokens_usd = 0.5
output_per_million_tokens_usd = 2.0

[providers.pricing.models."gpt-4.1"]
input_per_million_tokens_usd = 1.0
output_per_million_tokens_usd = 4.0
```

支持的 `protocol`：

- `oairesp`
- `oaichat`
- `anthropic`

Anthropic provider 还支持：

```toml
anthropic_version = "2023-06-01"
```

定价规则：

- `pricing.default`: provider 默认价格
- `pricing.models.<model>`: 对指定请求模型或映射后的上游模型覆盖价格

成本按 token 计算：

- `input_per_million_tokens_usd`
- `output_per_million_tokens_usd`

## 3.5 admin

远程管理接口使用独立 admin token：

```toml
[admin]
enabled = true
token = "replace-with-a-strong-admin-token"
```

- `enabled`: 是否启用 `/admin/*` 管理接口
- `token`: 管理接口 Bearer token

## 4. api_keys

入站请求必须带 `Authorization: Bearer <key>`，并在配置中声明对应 API key：

```toml
[[api_keys]]
uuid = "11111111-1111-1111-1111-111111111111"
name = "default-client"
key = "replace-with-a-strong-client-key"
limit_usd = 50.0
provider_limits_usd = { openai-chat-a = 30.0, anthropic-a = 20.0 }
group_limits_usd = { default = 40.0 }
```

- `uuid`: 这个 API key 的稳定唯一标识，用于数据库记账和统计
- `name`: 逻辑名称，用于记录统计
- `key`: 入站 Bearer token，直接写在配置文件中
- `limit_usd`: 这个 key 的总额度
- `provider_limits_usd`: 这个 key 在单个 provider 上的额度
- `group_limits_usd`: 这个 key 在单个 provider group 上的额度

额度生效规则：

- 请求前会检查当前路由下哪些 provider 还没超限
- 请求会优先走仍可用的 provider
- 总额度、provider 额度、group 额度同时存在时，实际约束等价于取最严格的那一个
- 用量数据库按 `uuid` 记录，`name` 仅用于展示

## 5. provider_groups

```toml
[[provider_groups]]
name = "default"
strategy = "fallback"
model_whitelist_only = true
models = ["gpt-4.1", "gpt-4.1-mini", "claude-sonnet"]
members = ["openai-chat-a", "anthropic-a"]
```

`members` 可以引用：

- provider 名
- provider group 名

provider group 也可以声明模型白名单：

- `model_whitelist_only = true` 时，仅允许 `models` 中列出的请求模型进入该 group
- group 白名单会在路由入口处先校验一次
- group 放行后，成员 provider 仍会继续应用自己的模型约束

## 6. previous_response_id

当前仅对 `oairesp` 入站接口生效。

也就是：

- `POST /v1/responses` 支持
- `POST /v1/chat/completions` 不支持
- `POST /v1/messages` 不支持

## 7. .env 默认加载

配置加载时会默认尝试读取 `.env`：

- 当前工作目录中的 `.env`
- 配置文件所在目录中的 `.env`

如果同名环境变量已经存在，则不会被 `.env` 覆盖。

## 8. 自动重载配置

`serve` 运行时会在请求进入时检查配置文件修改时间：

- 配置文件有变化时，会先重新加载并校验
- 只有校验成功，才切换到新配置
- 如果新配置非法，会继续保留旧配置提供服务
