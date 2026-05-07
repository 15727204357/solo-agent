## ADDED Requirements

### Requirement: AgentState 错误追踪字段
系统 SHALL 在 `AgentState` dataclass 中增加 `last_error`、`retry_count`、`error_classification`、`compaction_attempts` 字段，所有新字段有默认值以保证向后兼容。

#### Scenario: 初始状态无错误信息
- **WHEN** 创建新的 `AgentState` 实例
- **THEN** `last_error` 为空字典，`retry_count` 为 0，`error_classification` 为空字符串，`compaction_attempts` 为 0

#### Scenario: 记录最近错误
- **WHEN** 错误恢复循环捕获异常并分类
- **THEN** `last_error` 被更新为 `{"error_code": "<code>", "category": "<category>", "message": "<msg>", "stage": "<stage>"}`

#### Scenario: 重试计数递增
- **WHEN** 错误分类为 `retryable` 且系统决定重试
- **THEN** `retry_count` 增加 1

#### Scenario: 压缩尝试计数递增
- **WHEN** 上下文压缩阶段失败并尝试回退
- **THEN** `compaction_attempts` 增加 1

#### Scenario: 状态快照包含错误字段
- **WHEN** `AgentState.snapshot()` 被调用
- **THEN** 返回的字典包含 `last_error`、`retry_count`、`error_classification`、`compaction_attempts` 字段

### Requirement: AgentEvent 增强的 error 事件
系统 SHALL 在 `error` 事件的 `data` 字典中增加 `severity`（warn/error/fatal）、`recoverable`（bool）、`error_code`（str）字段，不修改 `AgentEvent` 的字段结构。

#### Scenario: 增强的 error 事件包含完整信息
- **WHEN** 系统发射 `type="error"` 事件
- **THEN** `data` 字典包含 `error_type`、`severity`、`recoverable`、`error_code` 字段

#### Scenario: 致命错误 severity 为 fatal
- **WHEN** 错误分类为 `fatal` 或 `architectural`
- **THEN** error 事件的 `data["severity"]` 为 `"fatal"`，`data["recoverable"]` 为 `false`

#### Scenario: 可重试错误 recoverable 为 true
- **WHEN** 错误分类为 `retryable`
- **THEN** error 事件的 `data["recoverable"]` 为 `true`

### Requirement: 错误恢复循环的 graph 集成
系统 SHALL 在 `_run_graph()` 中集成错误恢复循环，在工具执行和上下文压缩阶段注入 `retryable`/`fixable` 错误的重试逻辑，不改变其他阶段的执行路径。

#### Scenario: 工具调用 retryable 错误触发重试
- **WHEN** 工具调用失败且 ErrorClassifier 分类为 `retryable`
- **THEN** 系统在短暂等待后重试相同工具调用（最多与 `retry_count` 上限比较）

#### Scenario: 工具调用 fixable 错误注入修复提示
- **WHEN** 工具调用失败且 ErrorClassifier 分类为 `fixable` 且 `recovery_stage="collect_context"`
- **THEN** 系统在消息上下文中注入修复提示（要求模型先读取文件），然后允许 LLM 重新生成工具调用

#### Scenario: 工具调用 fatal 错误终止运行
- **WHEN** 工具调用失败且 ErrorClassifier 分类为 `fatal`
- **THEN** 系统发射 `type="error"` 事件，设置 `severity="fatal"`，`recoverable=false`，终止 agent 运行

#### Scenario: 压缩三次失败触发 architectural
- **WHEN** 上下文压缩阶段连续失败 3 次
- **THEN** ErrorClassifier 返回 `category="architectural"`，系统发射 `type="error"` 事件并终止运行

#### Scenario: 错误恢复后继续正常执行
- **WHEN** retryable 错误恢复成功（重试或 LLM 修复后工具调用成功）
- **THEN** 系统继续执行 agent 图的下一个阶段，`retry_count` 不重置为 0（保留用于跨阶段追踪）
