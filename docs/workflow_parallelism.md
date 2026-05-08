# Workflow Parallelism Gate

Solo Agent 仅在所有四个独立性条件全部通过时才允许并行开发。这是由确定性 Python 代码做出的安全决策，不依赖 LLM 判断。

## Reference

本设计遵循 `obra/superpowers` 的保守原则：

- `dispatching-parallel-agents`：一个独立问题域一个子代理；不并行化相关问题或共享状态工作
- `subagent-driven-development`：每个任务独立子代理，两阶段审查
- `writing-plans`：精确文件路径、精确验证命令、无占位符

## Runtime Rule

所有条件通过：

1. Problem-domain independence — 问题域独立性
2. Context independence — 上下文独立性
3. Write-set independence — 写入集独立性
4. Verification independence — 验证独立性

则：

```text
execution_strategy = "parallel"
```

任一条件失败或证据缺失：

```text
execution_strategy = "serial"
```

## 为什么故障即关 (fail-closed)？

并行 Agent 开发在任务共享文件、隐式状态、测试目标或根因时是危险的。因此系统要求明确的结构化证据。如果规划器无法提供元数据，运行时默认串行。

## 结构化 Plan 元数据

规划器可在 plan 中包含以下 JSON 块：

```json
{
  "parallel_tasks": [
    {
      "id": "T1",
      "title": "Provider tests",
      "domain": "providers",
      "description": "Add provider config tests",
      "read_paths": ["backend/src/solo_agent/providers/"],
      "write_paths": ["backend/tests/test_provider_config.py"],
      "verify_commands": ["pytest backend/tests/test_provider_config.py -q"],
      "depends_on": [],
      "needs_global_context": false,
      "risk_flags": []
    }
  ]
}
```

## 事件流

工作流发出以下 SSE 事件：

- `parallelism_gate_started`
- `parallelism_gate_completed`

完成事件负载包含：

- `mode`: `parallel` 或 `serial`
- `allowed`: boolean
- `conditions`: 四个条件判定
- `conflicts`: 串行回退的原因
- `groups`: 并行组或串行任务顺序
- `tasks`: 标准化的任务候选

## 当前范围

本次变更仅落地确定性安全闸门。不包含完整 DAG 调度器、分支/worktree 隔离、或自动补丁集成。应在本闸门稳定测试后再构建后续能力。
