## Why

当前 agent 的错误处理仅有一层通用的 `try/except Exception`，所有异常被统一捕获并直接终止运行。缺乏错误分类（可重试 vs 可修复 vs 致命）、缺乏对话修复反馈循环（goose fix_conversation）、缺乏重试上限控制（`max_compaction_attempts=2`）。这导致：暂时性错误（如 LLM 超时、上下文压缩失败）直接中断整个会话，无法自动恢复；相同的错误会被重复尝试而非标记为架构问题。

## What Changes

- 新增 `ErrorClassifier` 模块：按错误类型分类为 `retryable`（可重试）、`fixable`（需模型修复）、`fatal`（致命停止）、`architectural`（三次相同失败）。
- 新增 `fix_conversation` 管道：在每次 LLM 调用前清洗消息历史（合并重复、修复角色、移除孤立工具调用、确保 user/assistant 交替）。
- 新增错误恢复循环：工具调用失败后，分类 → 注入修复提示 → 重试（最多 2 次压缩，3 次相同失败触发 architectural）。
- 扩展 `AgentState`：增加错误追踪字段（`last_error`、`retry_count`、`error_classification`、`compaction_attempts`）。
- 扩展 `AgentEvent`：增强 `error` 事件（`severity`、`recoverable`、`error_code`）。
- 新增 `ErrorRecoveryPolicy`：graph 层硬执行的重试/回退策略，不依赖 prompt 文本。

## Capabilities

### New Capabilities
- `error-classification`: ErrorClassifier 分类器，将异常映射为 retryable/fixable/fatal/architectural 四类，每类对应不同的恢复策略。
- `error-recovery`: 错误恢复循环，包括 fix_conversation 管道（LLM 调用前清洗消息）、注入修复提示、工具级重试、max_compaction_attempts=2 上限。
- `error-state`: AgentState 错误追踪字段，支撑重试计数、分类持久化、前端展示。

### Modified Capabilities
<!-- 本次不改动现有 spec 的需求，所有新增均为独立模块 -->

## Impact

- `backend/src/solo_agent/agent/error_classifier.py` — 新增 ErrorClassifier 模块
- `backend/src/solo_agent/agent/fix_conversation.py` — 新增 fix_conversation 管道
- `backend/src/solo_agent/agent/state.py` — 扩展 AgentState 错误字段
- `backend/src/solo_agent/agent/events.py` — 扩展 error 事件结构
- `backend/src/solo_agent/agent/graph.py` — 集成错误恢复循环（最小侵入）
- `backend/src/solo_agent/agent/policy.py` — BehaviorPolicy 增加错误恢复策略方法
- `backend/tests/test_error_classifier.py` — 错误分类器单元测试
- `backend/tests/test_fix_conversation.py` — fix_conversation 管道测试
- `backend/tests/test_error_recovery.py` — 集成恢复循环测试
