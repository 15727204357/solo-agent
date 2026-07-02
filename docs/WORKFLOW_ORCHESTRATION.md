# Solo Agent 工作流编排

本文档描述当前 V1 的真实运行时。历史迁移说明已经删除，因为现在产品只有一条主工作流：`backend/src/solo_agent/workflow/graphs.py` 中的 `build_main_workflow_graph()`。

## 运行模型

Solo Agent 使用一个 LangGraph `StateGraph` 作为主运行时。图负责记忆加载、Skill 上下文、规划、上下文预算保护、意图路由、工具选择、工具执行、受控补丁、团队子 Agent、错误恢复、回复生成、Skill evolution、记忆后处理和持久化。

设计原则是：已知的流程决策尽量显式放在 graph、router 或确定性策略中，而不是全部交给模型临场决定。

## 主链路

```text
START
  -> receive_user_turn
  -> memory prelude
  -> skill_context
  -> context_guard_before_plan
  -> load_task_state
  -> plan
  -> task_state
     -> [run_mode=plan 且 subagent_enabled=true] team_plan
          -> parallelism_gate
          -> team_develop
          -> team_test
             -> [需要修复且还有迭代预算] team_develop
             -> team_supervisor
                -> [补丁提案已生成] END，等待审批
                -> [没有补丁提案] collect_context
     -> [默认] collect_context
  -> inspect
  -> intent_route
  -> select_tools
  -> execute_tools
     -> [满足重路由条件且未超过上限] intent_route
     -> [否则] spec_compliance_review
  -> propose_verified_patch
  -> subdirectory_hint
  -> context_guard_before_respond
  -> respond
  -> skill_evolution
  -> sync_memory
  -> queue_prefetch
  -> compress_memory
  -> persist_snapshot
  -> END
```

如果关闭记忆，memory prelude 和 postlude 会跳过记忆相关节点，但整体工作流形状不变。

## 单 Agent 路径

默认路径面向一个 lead agent：

1. `collect_context` 收集有限且可解释的仓库上下文和代码智能证据。
2. `inspect` 执行安全、环境、重复调用和策略检查。
3. `intent_route` 生成路由计划，包括意图、备选意图、上下文范围、候选工具、已选工具、Skill/Recipe 候选、审批边界、验证策略、风险和决策痕迹。
4. `select_tools` 消费路由计划，不再重新做任务分类。
5. `execute_tools` 执行有边界的工具。只读工具可以批量执行，写入、验证和补丁仍然受控。
6. 如果出现工具无结果、工具失败、测试失败、补丁阻塞或上下文置信度不足，可以在次数上限内回到 `intent_route` 重路由。
7. 代码修改进入 verified editing，主工作区不会因为模型输出的一段文本就被直接改写。

## 团队路径

团队模式只在两个条件同时满足时启用：

- `run_mode=plan`
- `subagent_enabled=true`

团队路径是：

```text
team_plan -> parallelism_gate -> team_develop -> team_test -> team_supervisor
```

这里的 `parallelism_gate` 是 developer 编排决策点。它判断任务能不能拆分、拆成几个 developer，并受 `max_concurrent_subagents` 限制。它不会进入旧的 `parallel_dispatch` 链路。

developer 子 Agent 会在隔离的轻量工作区里执行。工作区准备策略是：

1. 优先使用 Git worktree + 增量 overlay。
2. 如果 worktree 不可用，降级为复制工作区。
3. developer 只能修改自己的工作区。
4. `team_supervisor` 只把不冲突的 diff 转成主工作区的 verified patch proposal。

## Reflection 层

执行循环是 ReAct 风格，但 harness 同时加入了 Reflection 反馈点：

- `team_test` 会根据验证失败生成结构化反馈。
- `team_supervisor` 会在进入补丁审批前判断团队输出是否可用。
- outcome judge 和错误分类会判断 run 是成功、需要恢复还是应该停止。
- patch gate 会阻止不安全或未经验证的修改。
- route replay 和 eval scoring 能复盘路由决策。
- memory compress 和 memory candidates 会提炼可沉淀经验。
- skill_evolution 会在证据足够时提出 Skill/Recipe 变更提案。

这不是完全自动自我修改系统。Skill 变更和主工作区修改仍然要经过提案或审批边界。

## 错误恢复

节点异常会进入：

```text
classify_error -> recovery_route
  -> [可恢复] recovery_action -> repetition_guard
  -> [策略违规] blocked_response
  -> [环境问题] environment_error_response
  -> [架构性失败] architecture_failure_response
```

重复失败或结构性不安全的恢复尝试会停止，不会无限循环。

## 持久化与回放

运行时会持久化消息、工具调用、时间点、快照、补丁提案、Skill 变更提案、路由决策、子 Agent 运行记录、审查报告和 graph 快照。route replay 可以从事件流中重建每一轮路由，便于审计和评测。
