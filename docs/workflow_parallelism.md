# 并行门控

`parallelism_gate` 当前是团队 developer 编排门控。它的职责是判断一个计划应该由一个 developer 完成，还是拆成多个 developer assignment。

它不是产品主链路里的任意子 Agent fan-out。

## 当前产品规则

团队模式只在两个条件同时满足时启动：

- `run_mode=plan`
- `subagent_enabled=true`

当前主链路是：

```text
team_plan -> parallelism_gate -> team_develop -> team_test -> team_supervisor
```

下面这条旧链路不是产品路径：

```text
parallelism_gate -> parallel_dispatch -> wait_subagents -> supervisor_review
```

这些旧节点仍然注册在图里，用于兼容测试和内部诊断。

## 门控决策

门控读取结构化团队任务，输出 developer assignment 元数据。只有在以下维度足够独立时才拆分：

- 问题领域；
- 所需上下文；
- 写入路径；
- 验证命令。

如果拆分不安全，就输出一个 developer assignment。失败时保守退回一个 developer 是刻意设计：保留 tester / supervisor 闭环，同时避免并行冲突。

## 为什么保留这个门控

developer 并行是资源和风险决策，不只是工具执行决策。门控能保证：

- 不会多个 developer 暗中写同一个主工作区；
- 不让模型临时决定任意拓扑；
- developer 数量有显式预算；
- supervisor 能拿到清楚的 assignment 元数据；
- 无法证明独立时有确定性降级策略。

## 事件

当前团队路径会发出：

- `team_plan_started`
- `team_plan_completed`
- `parallelism_gate_started`
- `parallelism_gate_completed`
- `team_developer_started`
- `team_developer_completed`
- `team_tester_started`
- `team_tester_completed`
- `team_supervisor_started`
- `team_supervisor_completed`
- `patch_proposed`
- `patch_approval_required`
