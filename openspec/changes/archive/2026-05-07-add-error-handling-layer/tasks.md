## 1. Foundation: 错误分类基础类型

- [x] 1.1 创建 `backend/src/solo_agent/agent/error_classifier.py`，定义 `ErrorClassification` frozen dataclass（`category`、`error_code`、`severity`、`recoverable`、`recovery_stage`）
- [x] 1.2 实现 `ErrorClassifier` 类骨架：`classify()` 方法签名、`register()` 方法签名、内置映射表结构（exception_type → category 映射）
- [x] 1.3 在 `backend/tests/test_error_classifier.py` 中创建测试文件骨架，编写 `ErrorClassification` 序列化测试

## 2. Worker A: ErrorClassifier 核心实现

- [x] 2.1 实现 `ErrorClassifier` 内置异常映射表：`TimeoutError` → retryable, `ConnectionError` → retryable, `PermissionError` → fatal, `ValueError` → fatal（默认），`KeyError` → fatal，通过 `__cause__` 链遍历匹配
- [x] 2.2 实现 LLM 相关 HTTP 状态码错误分类（408/429/502/503/504 → retryable，401/403 → fatal），通过错误消息模式匹配 `status_code` 或 `HTTPStatus`
- [x] 2.3 实现上下文压缩失败分类（compaction_attempts 参数判断：<2 → retryable, >=2 → architectural）
- [x] 2.4 实现 `ErrorClassifier.register(exception_type, category, error_code, severity)` 可扩展注册方法
- [x] 2.5 实现 `_detect_identical_error()` 方法：基于 `(error_code, stage)` 组合判断是否为相同错误的连续出现
- [x] 2.6 在 `backend/tests/test_error_classifier.py` 中编写单元测试：覆盖所有内置异常映射、LLM HTTP 状态码分类、压缩分类、未匹配默认 fatal、注册自定义异常、同错三次检测

## 3. Worker B: fix_conversation 管道实现

- [x] 3.1 创建 `backend/src/solo_agent/agent/fix_conversation.py`，定义 `ConversationRepair` 类（`fix_conversation()` 方法 + `disabled_fixes` 配置）
- [x] 3.2 实现步骤 1+2：`_merge_text_content_items()` 合并连续 Text content blocks；`_trim_assistant_text_whitespace()` 去除 assistant 尾部空白
- [x] 3.3 实现步骤 3+4：`_remove_empty_messages()` 移除空内容消息；`_fix_empty_tool_results()` 空工具结果添加 "(empty result)" 占位
- [x] 3.4 实现步骤 5：`_fix_tool_calling()` 移除孤立 tool_use/tool_result，修复错误 role 的 content 块
- [x] 3.5 实现步骤 6+7：`_merge_consecutive_messages()` 合并相同 role 的连续消息；`_fix_lead_trail()` 确保对话以 user 开始、以 user 结束
- [x] 3.6 实现步骤 8：`_populate_if_empty()` 空对话填充 "Hello"
- [x] 3.7 在 `backend/tests/test_fix_conversation.py` 中编写单元测试：覆盖 8 步管道顺序执行、各步骤独立测试、disabled_fixes 开关、LangChain 消息格式兼容、边界情况（空列表、单消息、全 tool 消息）

## 4. Worker C: AgentState 扩展与 AgentEvent 增强

- [x] 4.1 在 `backend/src/solo_agent/agent/state.py` 中向 `AgentState` 添加字段：`last_error: dict[str, Any] = field(default_factory=dict)`、`retry_count: int = 0`、`error_classification: str = ""`、`compaction_attempts: int = 0`
- [x] 4.2 更新 `AgentState.snapshot()` 方法，将新增的三个字段纳入序列化字典
- [x] 4.3 在 `backend/src/solo_agent/agent/graph.py` 的 `_event()` 辅助函数中增强 error 事件构建：当 type="error" 时，自动注入 `severity`、`recoverable`、`error_code` 到 data 字典
- [x] 4.4 在 `backend/tests/test_agent_graph.py` 中编写测试：验证 AgentState 新字段默认值、snapshot 包含新字段、error 事件包含增强字段

## 5. Worker D: Graph 层错误恢复循环集成

- [x] 5.1 在 `backend/src/solo_agent/agent/policy.py` 的 `BehaviorPolicy` 中新增 `classify_error(exception, stage, attempt_count) -> ErrorClassification` 方法，委托给 ErrorClassifier
- [x] 5.2 在 `BehaviorPolicy` 中新增 `should_retry(classification, retry_count) -> tuple[bool, str]` 方法，判断是否应重试并返回决策理由
- [x] 5.3 在 `BehaviorPolicy` 中新增 `build_fix_prompt(classification) -> str` 方法，根据 error_code 和 recovery_stage 生成修复提示文本
- [x] 5.4 在 `_execute_tools_node()`（`graph.py:819-1027`）中集成错误恢复：在单个工具调用失败时，调用 `classify_error()` → 根据 category 分支处理（retryable 等待后重试、fixable 注入修复提示、fatal 发射 error 事件并终止、architectural 终止）
- [x] 5.5 在 `_context_guard_stage()`（`graph.py:1371-1391`）中集成 `compaction_attempts` 追踪：在 `_fallback_main_compression` 之前递增计数，达到上限后分类为 architectural 并终止
- [x] 5.6 在 `run_agent_events()`（`graph.py:109-118`）中保留现有顶层 try/except 不变，确保未在恢复循环中捕获的异常仍被统一处理和持久化

## 6. Worker E: 集成测试与验证

- [x] 6.1 在 `backend/tests/test_error_recovery.py` 中编写集成测试：模拟 retryable 错误（超时）后成功恢复、fixable 错误注入修复提示、fatal 错误终止运行、architectural 三次同错终止
- [x] 6.2 在 `backend/tests/test_error_recovery.py` 中编写上下文压缩上限测试：compaction_attempts 递增至 3 触发 architectural
- [x] 6.3 在 `backend/tests/test_error_recovery.py` 中编写端到端测试：完整 graph 运行中有工具失败 → 分类 → 恢复 → 继续执行的流程

## 7. Verification

- [x] 7.1 运行完整测试套件：`uv run --extra dev python -m pytest backend/tests/test_error_classifier.py backend/tests/test_fix_conversation.py backend/tests/test_error_recovery.py backend/tests/test_agent_graph.py -v`
- [x] 7.2 运行 linter：`uv run --extra dev ruff check backend/src/solo_agent/agent/error_classifier.py backend/src/solo_agent/agent/fix_conversation.py backend/src/solo_agent/agent/state.py backend/src/solo_agent/agent/graph.py backend/src/solo_agent/agent/policy.py`
- [x] 7.3 确认所有现有测试通过（回归验证）：`uv run --extra dev python -m pytest backend/tests/ -q --ignore=backend/tests/test_error_classifier.py --ignore=backend/tests/test_fix_conversation.py --ignore=backend/tests/test_error_recovery.py`
- [x] 7.4 确认现有 `agent` 模式行为未被改变：工具层错误返回格式不变、plan 模式不受影响、agent 模式正常执行路径不变
