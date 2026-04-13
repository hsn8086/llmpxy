# 路由与失败切换

本文档说明 `provider`、`provider group`、失败切换和退避行为。

## 1. 顶层 route

### 直接走 provider

```toml
[route]
type = "provider"
name = "provider-a"
```

行为：

- 所有请求都只打到 `provider-a`
- 不会再走其他 provider 或 group
- 如果它失败，就按自身重试阈值和总轮次处理

### 走 group

```toml
[route]
type = "group"
name = "default"
```

行为：

- 先解析 group 成员顺序
- 再按 group 策略展开成实际 provider 列表

## 2. fallback 组策略

配置：

```toml
[[provider_groups]]
name = "default"
strategy = "fallback"
members = ["provider-a", "provider-b", "provider-c"]
```

行为：

- 按顺序尝试 `provider-a -> provider-b -> provider-c`
- 某个 provider 连续失败达到阈值后，切到下一个
- 如果该轮所有 provider 都失败，进入指数退避

适合场景：

- 明确主备关系
- 希望先用主 provider，失败再切备份

## 3. load_balance 组策略

配置：

```toml
[[provider_groups]]
name = "balanced"
strategy = "load_balance"
members = ["provider-a", "provider-b"]
```

行为：

- 每次请求轮换起始成员
- 第一次可能先从 `provider-a` 开始
- 第二次会优先从 `provider-b` 开始
- 如果成员是 group，则先轮换组成员，再递归展开

适合场景：

- 多 provider 分担流量
- 避免请求总落在同一个 provider 上

## 4. 单个 provider 的失败切换规则

配置：

```toml
[retry]
provider_error_threshold = 3
```

行为：

- 一个 provider 连续失败 1 次，不立刻切走
- 连续失败 2 次，仍在当前 provider 内继续尝试
- 连续失败 3 次，切到下一个 provider

## 5. 可重试错误

当前会重试并参与 provider 切换的错误包括：

- 网络错误
- 超时
- HTTP `408`
- HTTP `429`
- HTTP `5xx`

这些错误会计入 provider 的连续失败次数。

## 6. 不可重试错误

以下错误会直接返回，不再切换 provider：

- 其他 `4xx`

原因是这类错误通常表示请求本身不兼容，不适合把所有 provider 都打一遍。

## 7. 全轮失败后的指数退避

配置：

```toml
[retry]
base_backoff_seconds = 1.0
max_backoff_seconds = 60.0
max_rounds = 5
```

行为：

- 第 1 轮全失败后，等待 `1s`
- 第 2 轮全失败后，等待 `2s`
- 第 3 轮全失败后，等待 `4s`
- 以此类推，直到 `max_backoff_seconds`
- 最多尝试 `max_rounds`

## 8. 嵌套 group

例如：

```toml
[[provider_groups]]
name = "cn"
strategy = "fallback"
members = ["provider-a", "provider-b"]

[[provider_groups]]
name = "intl"
strategy = "fallback"
members = ["provider-c", "provider-d"]

[[provider_groups]]
name = "default"
strategy = "load_balance"
members = ["cn", "intl"]
```

说明：

- `default` 先在 `cn` 和 `intl` 之间做轮转
- 每个子 group 再根据自己的策略展开

注意：

- 不允许 group 循环引用
- provider 和 group 名字不能重复
