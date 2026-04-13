# FAQ

## 1. 这个项目到底做什么？

它是一个本地兼容层。

对外：

- 接收 OpenAI `Responses API` 风格请求

对内：

- 转成 `Chat Completions API` 风格请求
- 转发到你配置的上游地址
- 再把结果包装成 `Responses API` 风格返回

适合场景：

- 你的上游只支持 `Chat Completions`
- 但你的客户端已经写成了 `Responses API`

## 2. 为什么我明明配置了多个 provider，但只有一个在工作？

先检查 `[route]`：

```toml
[route]
type = "provider"
name = "provider-a"
```

如果你这样配置，服务只会走 `provider-a`。

如果你想让多个 provider 参与路由，需要改成：

```toml
[route]
type = "group"
name = "default"
```

并配置对应的 `provider_groups`。

## 3. `fallback` 和 `load_balance` 有什么区别？

`fallback`：

- 固定顺序展开成员
- 更像主备
- 优先使用前面的 provider

`load_balance`：

- 每次请求轮转起始成员
- 更像简单轮询
- 适合多 provider 分摊流量

## 4. 为什么 provider 失败一次没有立刻切换？

因为当前默认策略不是“失败一次就切换”，而是“同一个 provider 连续失败达到阈值后再切换”。

默认配置：

```toml
[retry]
provider_error_threshold = 3
```

也就是说：

- 第 1 次失败：还留在当前 provider
- 第 2 次失败：还留在当前 provider
- 第 3 次失败：切到下一个 provider

## 5. 哪些错误会触发重试和 provider 切换？

会触发重试的错误：

- 网络异常
- 请求超时
- HTTP `408`
- HTTP `429`
- HTTP `5xx`

这些错误说明 provider 当前不可用、限流或短时异常，适合重试或切换。

## 6. 哪些错误不会触发切换？

其他 `4xx` 一般不会切换。

例如：

- `400`
- `401`
- `403`
- `404`

通常这类错误表示：

- 请求格式不对
- 密钥不对
- 模型名不对
- 上游接口不兼容

这时把所有 provider 全打一遍往往没有意义。

## 7. 为什么会看到退避等待？

当一整轮 provider 都失败后，服务不会无限快速重试，而是进入指数退避。

默认是：

- 第 1 轮失败后等 `1s`
- 第 2 轮失败后等 `2s`
- 第 3 轮失败后等 `4s`

这样能避免：

- 上游雪崩时持续打爆对方
- 本地日志爆量
- 无效重试占用太多资源

## 8. `previous_response_id` 为什么会失效？

常见原因：

1. 该 ID 从未保存成功
2. 会话已经过期，被 TTL 清理
3. 你切换了存储后端或存储文件位置
4. 你删掉了 SQLite 文件或会话目录

重点检查：

```toml
[storage]
backend = "sqlite"
sqlite_path = "./data/gpt-trans.db"
ttl_seconds = 604800
```

## 9. SQLite 和 file 后端怎么选？

推荐默认用 `sqlite`。

用 `sqlite` 的情况：

- 长期运行
- 希望更稳
- 会话量较多

用 `file` 的情况：

- 本地调试
- 想直接查看 JSON 文件
- 部署特别轻量

## 10. 为什么流式返回的事件和官方 OpenAI 还不完全一样？

因为这个项目本质是兼容层，不是 OpenAI 官方服务本体。

当前已经尽量输出接近 `Responses API` 的事件，例如：

- `response.created`
- `response.output_item.added`
- `response.output_text.delta`
- `response.function_call_arguments.delta`
- `response.output_item.done`
- `response.completed`

但不同上游对 Chat Completions 流式 chunk 的细节实现并不完全一致，所以无法保证和官方 100% 字节级一致。

## 11. 为什么模型映射好像没生效？

先确认模型映射是写在对应 provider 下面，而不是全局：

```toml
[[providers]]
name = "provider-a"
base_url = "https://provider-a.example.com/v1"
api_key_env = "PROVIDER_A_API_KEY"

[providers.models]
"gpt-4.1" = "provider-a-chat-model"
```

还要注意：

- 映射是按最终选中的 provider 生效
- 不同 provider 可以把同一个请求模型映射成不同的上游模型

## 12. 如果没配置模型映射会怎样？

当前行为是：

- 如果某个 provider 没有该模型的映射
- 就直接把请求里的模型名原样透传给上游

因此如果上游不认识这个模型名，就可能返回 `400` 或 `404`。

## 13. 日志会不会泄露密钥？

当前实现会自动对一些常见敏感字段做脱敏，包括：

- `Authorization`
- `api_key`
- `token`
- `secret`
- `password`
- `*_key`

但如果你把密钥塞进了非常规字段名里，依然有可能被记录。

建议：

- 不要把密钥放进业务 `metadata`
- 不要把敏感数据放进用户输入里

## 14. `info.log` 和 `debug.log` 的区别是什么？

`info.log`：

- 业务事件为主
- 更适合线上日常查看

`debug.log`：

- 请求摘要、响应摘要、调度细节更多
- 更适合排查问题

## 15. 上游 API Key 从哪里读取？

从 provider 自己的 `api_key_env` 对应环境变量读取。

例如：

```toml
[[providers]]
name = "provider-a"
api_key_env = "PROVIDER_A_API_KEY"
```

然后启动前设置：

```bash
export PROVIDER_A_API_KEY="your-secret"
```

## 16. 为什么启动时报配置错误？

常见原因：

1. `route.name` 指向了不存在的 provider 或 group
2. group 成员拼错名字
3. provider 和 group 名重了
4. group 形成循环引用
5. `storage.backend` 填错

可以优先检查：

- `doc/configuration.md`
- `doc/examples/`

## 17. 如何判断请求到底落到了哪个 provider？

看日志。

`info.log` 和 `debug.log` 里都会带：

- `request_id`
- `provider`
- `round`
- `attempt`

你可以按 `request_id` 追完整条调用链。

## 18. 如何做最简单的生产部署？

建议至少做到：

1. 用 `sqlite` 存储会话
2. 把 `logs/` 和 `data/` 挂到持久卷
3. 明确设置所有 provider 的环境变量
4. 为服务加进程守护或容器重启策略
5. 用 `healthz` / `readyz` 做探活

## 19. 如果我想新增 provider，应该怎么做？

步骤通常是：

1. 在 `[[providers]]` 里增加一段新配置
2. 设置新的 `api_key_env`
3. 增加该 provider 的模型映射
4. 把它加入某个 `provider_group`
5. 重启服务

## 20. 文档从哪里开始看最合适？

推荐阅读顺序：

1. `doc/quick-start.md`
2. `doc/configuration.md`
3. `doc/routing-and-failover.md`
4. `doc/api-usage.md`
5. `doc/logging-and-storage.md`
6. `doc/examples/`
