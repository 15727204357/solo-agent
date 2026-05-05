# Solo Agent 记忆层

这个包负责第一阶段 MVP 的 SQLite 持久化边界。它的目的不是做复杂长期记忆，而是先把每次 Agent 运行保存成可调试、可复盘的数据。

## MVP Schema

- `sessions`：用户可见的会话或项目容器，通过 `SessionType` 区分类型。
- `runs`：一次 Agent 执行，记录 provider、model、状态和错误信息。
- `messages`：按顺序保存 system、user、assistant、tool 等角色消息。
- `tool_calls`：保存工具调用审计记录，包括参数、结果、状态和错误。
- `timing_points`：保存 callback 风格的可观测时间点。
- `snapshots`：保存 checkpoint、context、summary 等 JSON 快照。
- `messages_fts`：FTS5 sidecar index，用于在当前 session 内检索历史消息。
- `MEMORY.md` / `USER.md`：Hermes 风格内置记忆文件，分别保存项目长期事实和用户偏好。

## 后续演进

- 保持 `SessionType` 为稳定字符串，后续可以兼容 SQLite v11 风格的会话分类。
- 把 `snapshots.snapshot_type = "checkpoint"` 当作 checkpoint replay 的第一条缝。
- FTS5 不直接替换 `messages`，而是作为 sidecar index：给 message content 建虚拟表和触发器。
- `search_memory` 严格限定 `session_id`，避免不同会话之间的记忆污染。
- 中文短语检索以 FTS5 为主，查不到时使用 LIKE 降级，保证召回体验。
- `prefetch_all`、`sync_all`、`queue_prefetch_all` 构成基础记忆生命周期。
- `on_pre_compress` hook 接在消息压缩或 snapshot 生成之前，用来保存可检索摘要。
- 当前摘要通过模型生成；如果摘要失败，不阻塞本轮 Agent 运行。
