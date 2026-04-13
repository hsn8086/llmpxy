# Agent Notes

## 工作约束

- 每次完成一个可独立提交的变更后，必须创建 git commit，禁止长期堆积未提交修改。
- 开始新任务前先检查工作区状态，确认当前修改是否已经提交。
- 未经明确要求，不得改写历史，不得执行破坏性 git 操作。
- 提交前必须完成与本次变更相关的测试、格式化和必要的检查。

## 提交规范

- 提交消息使用 Conventional Commits：`<type>(<scope>): <summary>`。
- `type` 允许使用：`feat`、`fix`、`refactor`、`test`、`docs`、`chore`。
- `scope` 使用受影响模块或目录，例如 `proxy`、`config`、`docs`、`agent`。
- `summary` 使用简洁英文短句，聚焦本次变更意图。
- 一次提交只做一类清晰变更；如果变更目标不同，拆成多个提交。

示例：

- `feat(subscription): add template-based subscription model`
- `fix(dispatcher): respect provider group disable state`
- `docs(agent): document commit workflow`

完整流程与约定见 `docs/git-workflow.md`。

## 文档参考

如果需要查看 OpenAI API 参考文档的使用方法，请先查看 `docs` 目录：

- `docs/openai-api-links.md`：整理后的 OpenAI API 文档链接索引
- `docs/openai-api-docs-usage.md`：如何把文档页面地址改写为可直接 `curl` 的 `.md` 地址，例如 `https://developers.openai.com/api/docs/guides/batch.md`

处理 API 相关问题时，优先参考上述文档。
