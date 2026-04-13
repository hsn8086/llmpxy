# 开发说明

## 1. 安装依赖

```bash
uv sync
```

## 2. 本地运行

```bash
uv run gpt-trans serve --config ./config.toml
```

## 3. 格式化

```bash
uvx ruff format .
```

## 4. 类型检查

```bash
uvx ty check --python .venv/bin/python src tests
```

## 5. 测试

```bash
uv run pytest
```

## 6. 当前测试覆盖范围

当前测试已覆盖：

- 配置解析
- provider/group 配置校验
- fallback 和 load_balance 调度
- SQLite 和 file 存储
- 普通请求
- 流式请求
- `previous_response_id`
- 日志脱敏

## 7. 核心代码位置

- `src/llmpxy/config.py`: 配置模型和校验
- `src/llmpxy/dispatcher.py`: provider/group 调度、切换和退避
- `src/llmpxy/proxy_client.py`: 上游请求、HTTP 代理和错误分类
- `src/llmpxy/app.py`: FastAPI 多协议入口
- `src/llmpxy/protocols/`: 协议适配层
- `src/llmpxy/storage_sqlite.py`: SQLite 会话存储
- `src/llmpxy/storage_file.py`: 文件会话存储
- `src/llmpxy/logging_utils.py`: 日志初始化和脱敏

## 8. 新增功能时的建议

建议每次新增功能都同步补：

- 配置文档
- 使用文档
- 单元测试
- 类型检查

如果改动了 `Responses` 或流式事件映射，优先补 `app` 和 `dispatcher` 相关测试。
