# Git Workflow

本项目从现在开始使用 git 管理，所有代码、文档和配置变更都必须通过提交记录留痕。

## 基本要求

- 每次完成一个可独立交付的变更后，必须立即提交，不允许长时间保留未提交修改。
- 开始下一项任务前，先确认当前工作区已清理完毕，避免把不同目标的改动混在同一次提交里。
- 提交前必须完成本次变更相关的测试。
- 提交前应运行必要的格式化和检查，Python 项目至少包括：
  - `uvx ruff format`
  - `uvx ty check`
  - `uv run pytest`

## 提交消息格式

统一使用 Conventional Commits：

```text
<type>(<scope>): <summary>
```

说明：

- `type`：变更类型
- `scope`：受影响模块、目录或业务域
- `summary`：一句话概括本次变更意图，使用英文，首字母小写，不加句号

允许的 `type`：

- `feat`：新增功能
- `fix`：缺陷修复
- `refactor`：重构但不改变对外行为
- `test`：测试新增或调整
- `docs`：文档更新
- `chore`：工具链、脚手架、仓库维护类修改

示例：

- `feat(auth): add jwt refresh token flow`
- `fix(provider): apply per-provider proxy settings`
- `test(storage): cover sqlite retention logic`
- `docs(requirements): refine subscription priority rules`
- `chore(repo): initialize git repository`

## 建议流程

1. 修改代码或文档。
2. 运行格式化、类型检查和测试。
3. 确认只暂存本次需要提交的内容。
4. 使用符合规范的提交消息创建 commit。
5. 提交完成后再次检查工作区，确认没有遗漏。

## 禁止事项

- 不要把 `.env`、数据库文件、日志、虚拟环境和缓存文件提交到仓库。
- 不要把多个无关主题合并到同一个提交。
- 未经明确要求，不要改写历史，不要强推，不要使用破坏性 git 命令。
