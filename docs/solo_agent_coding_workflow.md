# Solo Agent 团队 Coding 工作流

本文档说明当前团队 Coding 路径。目标不是任意 Agent swarm，而是一套小而清楚的工程团队模式：明确角色、明确隔离、明确合并边界。

## 团队链路

```text
team_plan -> parallelism_gate -> team_develop -> team_test -> team_supervisor
```

角色分工：

- `planner`：把 lead plan 或任务状态整理成任务目录。
- `parallelism_gate`：判断 developer 是否适合并行，以及应该拆成几个 developer assignment。
- `developer`：只在自己的隔离工作区内修改代码。
- `tester`：运行目标验证，并把失败转成结构化反馈。
- `supervisor`：检查最终证据，把安全的 diff 转成 verified patch proposal。

## 并行策略

`parallelism_gate` 不是旧的 fan-out 分发器。在团队模式里，它只回答一个问题：这个计划应该由一个 developer 做，还是拆成多个 developer assignment？

判断依据包括：

- 问题领域是否独立；
- 所需上下文是否独立；
- 写入路径是否不冲突；
- 验证命令是否独立；
- `max_concurrent_subagents` 配置的 developer 数量上限。

如果不能证明安全拆分，就只生成一个 developer assignment。这不是失败，而是保守策略：仍然保留 tester 和 supervisor 的团队闭环，同时避免制造合并冲突。

## developer 工作区边界

developer 子 Agent 可以真实修改代码，但不能直接改主仓库。

每个 developer assignment 都会准备一个轻量工作区：

1. 如果命令工作区是 Git 仓库，优先使用 `git worktree add --detach`。
2. 把命令工作区中的未提交本地状态增量 overlay 到 worktree。
3. 如果 worktree 不可用，降级为复制工作区。
4. developer 只拿到受限的 sandbox 工具注册表。
5. 工具调用 ledger 和 diff 证据会交给 supervisor 审查。

这不是 Docker 或虚拟机级别的沙箱，但对 V1 来说足够保证 developer 不直接写主工作区。

## developer 工具循环

如果 provider 支持 tool calling，`team_develop` 会用受限工具循环执行 developer assignment。允许的能力包括有边界的读文件、搜索、准备编辑、预览补丁、在 sandbox 中应用文本编辑、运行 pytest/ruff、查看 sandbox diff。

如果 provider 不支持 tool calling，团队路径会降级为 JSON patch request，并且只把 patch 应用到隔离工作区。后续仍然要经过 tester 和 supervisor。

## tester 反馈循环

`team_test` 会执行 team plan 中的验证命令。如果没有显式命令，会使用 impact analysis 和 test relevance 给出的建议。验证失败时，tester 会生成结构化反馈，并在迭代预算内回到 `team_develop`。

这个循环是有上限的 Reflection 步骤，不是无限重试。

## supervisor 合并边界

`team_supervisor` 不相信模型最后的自然语言总结就是可合并产物。它会从最终工作区内容和 developer diff 中重建编辑，检查 developer 之间是否修改了同一文件或同一区域，然后生成普通的 verified patch proposal。

主工作区只有在正常审批通过后才会被修改。

## 保留的证据

团队运行可以保留：

- team plan 和 assignments；
- developer 工作区路径和 backend 类型，如 `worktree_overlay` 或 `copy`；
- sandbox diff；
- 工具调用 ledger；
- pytest/ruff 输出；
- tester report；
- supervisor report；
- patch proposal 元数据。
