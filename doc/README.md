# 文档目录

本目录包含 `llmpxy` 的使用、配置、运行和开发文档。

`llmpxy` 是一个多协议 LLM 代理，支持：

- OpenAI Responses
- OpenAI Chat Completions
- Anthropic Messages

同时支持：

- provider
- provider group
- `fallback`
- `load_balance`
- HTTP 代理
- SQLite / 文件会话存储

## 文档列表

- `quick-start.md`: 快速开始，适合第一次启动项目
- `configuration.md`: 完整配置说明，包括协议、provider、group、路由、代理、存储、日志和重试
- `routing-and-failover.md`: provider/group 路由、`fallback`/`load_balance`、失败切换和退避机制说明
- `api-usage.md`: 多协议接口使用说明，包括 `oairesp`、`oaichat`、`anthropic`
- `logging-and-storage.md`: 日志输出、HTTP 代理和会话持久化说明
- `development.md`: 本地开发、测试、格式化和类型检查说明
- `faq.md`: 常见问题与排障说明
- `examples/README.md`: 配置样例目录说明
- `../docs/requirements.md`: 用户系统、订阅、管理后台与用户前端的需求草案

## 推荐阅读顺序

1. `quick-start.md`
2. `configuration.md`
3. `routing-and-failover.md`
4. `api-usage.md`
5. `logging-and-storage.md`
6. `examples/README.md`
7. `faq.md`
8. `../docs/requirements.md`
