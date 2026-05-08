## Why

当前 `solo-agent` 的单一 agent graph 缺乏子代理并发编排能力和多层级任务拆解机制，无法处理需要并行研究的复杂开发任务。参考 DeerFlow 的 `create_agent` + middleware + `task` 子代理模式，引入分层工作流引擎，在保持现有外部接口不变的前提下，提升多步骤复杂任务的处理效率和安全性。

## What Changes

- 新增 `solo_agent.workflow` 运行层，实现 DeerFlow 式工作流引擎，旧 `agent.graph` 作为兼容门面桥接到新 runtime
- 新 runtime 采用 `LeadAgentFactory` 创建 lead agent，注入模型、工具、skills、memory、sandbox、subagent 开关
- 引入 `WorkflowState` 状态管理，兼容现有 `AgentState.snapshot()`，扩展 `messages`、`thread_data`、`sandbox`、`artifacts`、`todos`、`subagent_runs`
- 实现 Middleware 链：`ThreadDataMiddleware`、`SandboxMiddleware`、`ToolErrorHandlingMiddleware`、`SkillContextMiddleware`、`MemoryContextMiddleware`、`SubagentLimitMiddleware`、`Clarification/StopMiddleware`
- 实现 `task` 工具，支持 `description`、`prompt`、`subagent_type`、`max_turns`，后台运行子代理并发、超时、取消、事件回传
- 内置三个子代理：`general-purpose`（只读研究）、`code-review`（只读审查）、`quality`（运行 pytest/ruff）
- 本地沙箱第一版：为每个 run 建立隔离目录映射，所有文件读写仍通过现有 `ToolRegistry`，禁止子代理递归调用 `task`，子代理默认不允许直接编辑
- 新增事件类型：`task_started`、`task_running`、`task_completed`、`task_failed`、`subagent_limited`
- 新增配置项：`workflow_engine`、`subagent_enabled`、`max_concurrent_subagents`、`subagent_timeout_seconds`、`sandbox_mode`、`workflow_runtime_root`
- 保留现有 `run_agent_events()`、Web API、SSE 事件、SQLite memory、patch approval、`run_mode=agent|plan` 语义不变

## Capabilities

### New Capabilities
- `workflow-runtime`: DeerFlow 式工作流引擎核心，包含 LeadAgentFactory、WorkflowState、Middleware 链、与现有外部入口的兼容桥接
- `subagent-system`: 子代理注册表、executor、`task` 工具、并发控制（最多 3 个）、超时（900s）、取消、事件回传、禁止递归 `task` 调用
- `local-sandbox`: 本地沙箱隔离，每个 run/thread 的独立 workspace/uploads/outputs 运行目录，可插拔 `SandboxProvider` 接口（Docker 沙箱预留接口但不实现）
- `workflow-configuration`: 工作流引擎相关配置项定义、验证和默认值

### Modified Capabilities
<!-- 现有 spec 的 requirement 级别无变更，仅实现层面重构 -->

## Impact

- 新增模块：`backend/solo_agent/workflow/`（runtime、middleware、state、sandbox、subagent、configuration）
- 修改模块：`backend/solo_agent/agent/graph.py`（兼容门面桥接）、`backend/solo_agent/tool/`（新增 `task` 工具）、`backend/solo_agent/events.py`（新增事件类型）、`backend/solo_agent/configuration.py`（新增配置项）
- 模型层：新增 LangChain `ChatModel` 工厂以支撑 `create_agent` 工具调用，保留现有 `ChatProvider`
- 事件层：新增 5 种事件类型，SSE 推送链路扩展
- 依赖项：新增 `langchain-core`、`langgraph`（如尚未引入）
- 测试：需新增 workflow 层单元测试、子代理集成测试、沙箱隔离测试、兼容回归测试
