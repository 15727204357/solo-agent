## Context

Solo Agent 当前使用线性 agent graph（Milestone 1）：`plan → context → inspect → tools → respond`。错误处理仅有一层顶层 `try/except Exception`（`graph.py:109-118`），所有异常被统一捕获为通用 `"error"` 事件并终止运行。工具调用层（`registry.py`）有结构化的错误返回（`KeyError`、`PermissionError`、`ValueError`、`OSError`、`TimeoutError`），但 graph 层不做分类、不做重试、不做对话修复反馈。

项目愿景（`PROJECT_BRIEF.md:38`）明确列出了错误处理层的设计方向——Hermes 的 ErrorClassifier + goose 的 fix_conversation + `max_compaction_attempts=2`，原则：三次相同失败等于架构问题。

已有相关基础设施：
- `BehaviorPolicy.tool_protocol_violation()` 已区分 `recoverable` / `non-recoverable` 违反，并生成恢复工具调用
- `ContextManager.maybe_compress()` 已有两阶段回退（主 provider → 辅助 provider），`_fallback_main_compression()` 兜底
- `AgentState` 已有 `blocked`/`block_reason`/`summary_status` 等状态字段，但无错误追踪字段
- `AgentEvent` 是通用 frozen dataclass，`error` 事件仅存 `error_type` 在 data 中

关键约束：
- **最小侵入**：不修改 agent 模式现有执行路径的行为
- **graph 层硬执行**：错误恢复策略在 graph/policy 层强制，不依赖 prompt 文本（项目原则）
- **中文注释**，保持文件编码 UTF-8
- **子 agent 并行开发**：模块独立，可并行实现和测试
- **Python 生态**：langgraph >= 1.0, langchain >= 0.3, pydantic >= 2.0

## Goals / Non-Goals

**Goals:**
- 实现 `ErrorClassifier` 模块：将异常映射为 `retryable`、`fixable`、`fatal`、`architectural` 四类
- 实现 `fix_conversation` 管道：8 步纯函数管道，在 LLM 调用前清洗消息历史
- 实现错误恢复循环：分类 → 注入修复提示 → 工具级重试，max 2 次压缩，3 次相同失败触发 architectural
- 扩展 `AgentState`：增加 `last_error`、`retry_count`、`error_classification`、`compaction_attempts` 字段（加性，不改现有字段）
- 扩展 `AgentEvent`：增加 `severity`、`recoverable`、`error_code` 到 error 事件 data 中（不改现有结构）
- 新增 `ErrorRecoveryPolicy`：graph 层硬执行的重试/回退策略
- 单元测试覆盖三个新模块

**Non-Goals:**
- 不修改 agent 模式下现有执行路径的正常行为
- 不改变现有工具层（registry.py, readonly.py）的错误返回格式
- 不改变 `AgentEvent` 现有字段结构（仅扩展 data 字典）
- 不引入新的外部依赖（仅使用已有的 langchain/langgraph/pydantic）
- 不实现分布式错误追踪或告警
- 不实现 plan 模式下的错误恢复（plan 模式为只读生成，不需要工具级重试）

## Decisions

### D1: ErrorClassifier 作为独立模块 + 可扩展映射表

`ErrorClassifier` 作为独立模块 `agent/error_classifier.py`，核心是一个 `Exception → ErrorCategory` 的映射表（可注册扩展），而非硬编码的 if-elif 链。采用 dataclass 定义分类结果：

```python
@dataclass(frozen=True)
class ErrorClassification:
    category: Literal["retryable", "fixable", "fatal", "architectural"]
    error_code: str          # 如 "LLM_TIMEOUT", "CONTEXT_OVERFLOW"
    severity: Literal["warn", "error", "fatal"]
    recoverable: bool
    recovery_stage: str | None  # 如 "collect_context", "fix_conversation"
```

映射表基于异常类型链（`__cause__` 遍历）和错误消息模式匹配。可扩展：外部可通过 `register()` 方法添加自定义映射。

**替代方案**: 在 `BehaviorPolicy` 中扩展（类似 `tool_protocol_violation`）。否定——BehaviorPolicy 专注于工具调用协议违反，错误分类是独立的横切关注点。

### D2: fix_conversation 作为 8 步纯函数管道

参考 goose 的 `fix_conversation` 实现（`crates/goose/src/conversation/mod.rs`），适配 Python/LangChain 的消息格式：

8 步管道（按 goose 源码顺序）：
1. `merge_text_content_items` — 合并 consecutive Text content blocks
2. `trim_assistant_text_whitespace` — 去除 assistant 消息尾部空白
3. `remove_empty_messages` — 移除空内容消息
4. `fix_empty_tool_results` — 空工具结果添加 "(empty result)" 占位
5. `fix_tool_calling` — 移除孤立 tool_use/tool_result，修复错误 role 的 content
6. `merge_consecutive_messages` — 合并相同 effective role 的连续消息
7. `fix_lead_trail` — 确保对话以 user 开始、以 user 结束
8. `populate_if_empty` — 空对话添加 "Hello" 占位

消息格式使用 `langchain.schema.messages`（`HumanMessage`、`AIMessage`、`ToolMessage`），与项目中 `ChatMessage` TypedDict 兼容。

**关键设计**: `fix_conversation` 不返回问题列表（简化 goose 的设计），只返回修复后的消息列表。如果某步骤无法修复，保持原样 pass。

### D3: 恢复循环集成在 graph 层，不创建单独 Service

错误恢复逻辑集成在 `_run_graph()` 中，通过 `ErrorRecoveryPolicy` 类封装策略决策。不创建单独的 ErrorRecoveryService——保持与现有 `BehaviorPolicy` 相同的模式（graph 层调用 policy 类获取决策，policy 类是纯逻辑无状态）。

恢复循环在工具执行阶段（`_execute_tools_node`）和压缩阶段（`_context_guard_stage`）注入，不影响其他阶段。

恢复流程：
```
工具调用失败 → ErrorClassifier.classify(exc) → ErrorClassification
  ├─ retryable → 增加 retry_count → 相同工具重试（等待退避）
  ├─ fixable → 注入 fix_prompt 到上下文 → LLM 重新生成 → 工具重试
  ├─ fatal → 发射 error 事件 → 终止运行
  └─ architectural → 发射 error 事件 → 终止运行 + 标记 code_review_needed
```

**替代方案**: 独立的 `ErrorRecoveryService` 包装整个 graph。否定——增加一层抽象，与现有 graph 层 pattern 不一致。

### D4: max_compaction_attempts=2，同错检测基于 error_code + 调用位置

上下文压缩重试上限 2 次，第三次相同失败触发 `architectural` 分类。同错检测基于 `(error_code, failing_stage)` 组合而非简单的错误字符串比较。

```python
compaction_attempts += 1
if compaction_attempts > 2:
    raise ArchitecturalError(
        f"Compaction failed {compaction_attempts} times at stage={failing_stage}, "
        f"error_code={error_code}. This is likely an architectural issue."
    )
```

计入 `compaction_attempts` 的只包括上下文压缩阶段的失败，不包括工具调用失败（工具失败用 `retry_count`）。

### D5: 加性修改 AgentState，向后兼容

采用"加性"策略：在 `AgentState` 中添加新字段（有默认值），不修改任何现有字段的类型或语义。所有新字段在 `snapshot()` 中同样序列化。

新增字段：
```python
last_error: dict[str, Any] = field(default_factory=dict)
# {"error_code": "...", "category": "...", "message": "...", "stage": "..."}
retry_count: int = 0
error_classification: str = ""  # retryable | fixable | fatal | architectural
compaction_attempts: int = 0
```

### D6: 中文注释，英文标识符

注释和文档字符串使用中文（项目约定），但代码标识符（类名、函数名、变量名）使用英文（Python 惯例 + 团队可读性）。所有源文件保存为 UTF-8（已是项目默认）。

### D7: 子 agent 并行开发——模块独立，接口先行

三个核心模块（`error_classifier.py`、`fix_conversation.py`、`error_recovery.py`）设计为独立模块，接口明确，可由子 agent 并行实现和测试。接口在 `design.md` 中定义为 proto，实现由 tasks.md 分配到不同并行工作流。

## Risks / Trade-offs

- **[R1] 恢复循环导致无限循环**: 如果 ErrorClassifier 将所有错误都分类为 retryable，可能导致死循环。 → `retry_count` 硬上限（相同 stage 最多 3 次），超过触发 architectural。
- **[R2] fix_conversation 修改消息可能导致上下文丢失**: 合并消息或移除孤立工具调用可能丢失重要上下文。 → 每个 fix 函数独立，可单独开关；默认全开但在紧急情况可降级。
- **[R3] 错误分类映射表需要持续维护**: 新的异常类型出现时映射表可能不准确。 → 可扩展的 `register()` 方法；未匹配的异常默认归类为 `fatal`（安全侧）。
- **[R4] Graph 层变复杂**: 在 `_run_graph()` 注入恢复逻辑增加循环复杂度。 → 恢复逻辑封装在 `ErrorRecoveryPolicy` 中，graph 层只做简单的条件调用。
- **[R5] 与现有 try/except 的交互**: 现有的多个 try/except 块（压缩、自评审、补丁提案）可能与新的 ErrorClassifier 产生双重处理。 → 保留现有 try/except 不变，ErrorClassifier 仅在新增的恢复循环中使用，不代替现有异常处理。
