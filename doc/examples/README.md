# 配置样例

本目录提供 `llmpxy` 的参考配置样例。

当前旧样例仍可作为 group/provider 结构参考，但协议字段需要迁移到新格式：

- provider 必须增加 `protocol`
- 可以增加 `proxy`
- 可以增加 `anthropic_version`

推荐直接参考项目根目录的 `config.example.toml` 作为最新版本基线。
