# 上下文管理系统

本模块把上下文预算、压缩触发、任务状态保活和目录提示注入收敛到 `solo_agent.context` 包中，供 Agent graph 在工具调用边界调用。

## 产品口径

- Web 端 Agent 产品对标 DeerFlow 2.0，线程、运行和恢复能力交给 LangGraph 托管。
- Solo Agent 不自建 recovered child session，也不在 provider overflow 后做会话迁移。
- 本模块只负责在当前线程内做预算评估、压缩和状态保活。

## 触发策略

- 常规任务：成功压缩次数小于等于 2 时，使用 80% 上下文窗口阈值，并由主模型生成摘要。
- 长任务：成功压缩次数超过 2 后，使用 50% 阈值，并优先用 Ollama `qwen3.5:4b` 作为 AuxiliaryClient 生成摘要。
- 压缩只在 plan 前、respond 前、run 结束后触发，避免截断工具调用链。

## 计数策略

- 普通文本使用 `utf8_bytes / 4` 向上取整估算。
- 代码文本优先使用 `tree-sitter` 和 `tree-sitter-language-pack` 解析叶子节点估算。
- 未知语言、依赖不可用或解析失败时自动回退普通文本规则。

## TaskList 状态

- `TaskListState` 对齐 oh-my-openagent 的结构化任务思想，任务条目包含 `id`、`subject`、`description`、`status`、`activeForm`、`blockedBy`、`blocks`、`owner` 和 `metadata`。
- planner prompt 要求输出 `<task-list-json>` 块，graph 优先解析结构化 JSON，只有缺失时才回退文本规则。
- 压缩后会注入 `<task-state>`，并保留 `Continue from`，保证长期任务恢复到正确的下一步。
- 工具层提供 `task_create`、`task_get`、`task_list`、`task_update`，为后续 Web 端任务面板和 LangGraph thread state 接入预留接口。

## 目录提示与权限

- `SubdirectoryHintTracker` 根据工具参数中的路径加载工作区内目录提示文件，不改写 system prompt。
- 压缩 agent 只调用 `ChatProvider.complete()`，不会获得工具注册表或执行权限。
