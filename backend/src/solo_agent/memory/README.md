# Solo Agent 记忆层

记忆包是 Solo Agent 的持久化和召回边界。它不是一个模糊的“长期记忆桶”，而是用来保存可审计运行数据、检索 Coding 相关历史，并治理哪些内容可以成为长期记忆。

## Schema

- `sessions`：用户可见的会话或项目容器。
- `runs`：一次 Agent 执行，记录 provider、model、状态和错误。
- `messages`：按顺序保存 system、user、assistant、tool 消息。
- `tool_calls`：工具调用审计记录。
- `timing_points`：可观测时间点。
- `snapshots`：checkpoint、summary、route decision、graph snapshot、review report 和 subagent run。
- `patch_proposals`：受控补丁提案和审批状态。
- `skill_change_proposals`：Skill/Recipe 变更提案和应用结果。
- `memory_candidates`：等待治理的候选长期记忆。
- `memory_entries`：已批准、已替换或已撤销的长期记忆。
- `workflow_observations`：可复用工作流观察。
- `messages_fts`：用于消息、摘要和内置记忆搜索的 FTS5 sidecar index。
- `MEMORY.md` / `USER.md`：内置项目事实和用户偏好。

## 检索策略

Coding 任务经常需要精确找回路径、符号、报错、命令输出、配置项和历史修复方式。因此 Solo Agent 默认使用 SQLite FTS5 + BM25 排序做记忆检索。

`search_memory` 会按 session 限定范围，也可以通过 `__builtin__` session 纳入内置记忆。如果 FTS5 没有结果，会使用 LIKE fallback，保证中文短语或特殊 token 仍有召回机会。

## 记忆生命周期

- `load_builtin_memory`：确保并加载 `MEMORY.md` 和 `USER.md`。
- `prefetch_all`：运行前收集内置记忆、最新摘要、最近消息和检索结果。
- `sync_all`：运行后写入用户和助手消息。
- `queue_prefetch_all`：为下一轮预取检索结果。
- `on_pre_compress`：压缩前提取候选长期记忆。
- `compress_memory`：保存压缩摘要；摘要失败不会阻塞整轮运行。

## 记忆治理

可能进入长期记忆的内容必须经过候选治理：

- pending、approved、rejected、duplicate、blocked 状态；
- 目标类型：项目记忆、用户记忆或 Skill 记忆；
- 置信度、来源片段、冲突 ID 和安全标记；
- approve、reject、replace、supersede、revoke 流程。

这样可以避免一次性上下文、错误模型输出或 prompt injection 文本静默进入长期记忆。

## 路由与回放记忆

路由决策会保存为 snapshot。这样运行结束后仍能查看当时为什么选择某个 intent、上下文范围和工具，也能支持 route replay 和 eval。
