# OpenAI API 参考使用说明

`docs/openai-api-links.md` 保存了从 `oai.txt` 和 `oai2.txt` 提取并增量合并后的 OpenAI API 文档链接。

如果需要直接拉取某篇 API 文档内容，可以把页面路径改成对应的 `.md` 地址，然后用 `curl` 获取。

## 用法示例

例如页面地址：

`https://developers.openai.com/api/docs/guides/batch`

可以直接请求：

```bash
curl https://developers.openai.com/api/docs/guides/batch.md
```

通常可按下面规则推导：

```text
https://developers.openai.com/<path>
->
https://developers.openai.com/<path>.md
```

## 说明

- 该方法适合快速获取文档正文，便于脚本处理或本地检索。
- 如果原始链接已经是完整页面路径，通常只需要在路径末尾追加 `.md`。
- 可先从 `docs/openai-api-links.md` 找到目标页面，再构造对应的 `.md` 地址。
