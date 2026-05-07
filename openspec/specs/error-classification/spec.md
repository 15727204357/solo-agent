## ADDED Requirements

### Requirement: 异常分类映射
系统 SHALL 提供 `ErrorClassifier` 模块，将 Python 异常对象分类为四种错误类别之一：`retryable`（可重试）、`fixable`（需模型修复）、`fatal`（致命停止）、`architectural`（架构问题，三次相同失败触发）。

#### Scenario: LLM 超时被分类为 retryable
- **WHEN** LLM 调用抛出 `TimeoutError` 或状态码为 408/429/502/503/504 的异常
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="retryable", recoverable=True)`

#### Scenario: 上下文溢出被分类为 retryable
- **WHEN** 上下文压缩失败且 `compaction_attempts < 2`
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="retryable", recovery_stage="context_guard")`

#### Scenario: 工具调用缺少上下文被分类为 fixable
- **WHEN** 工具调用违反读前编辑协议（如缺少 `read_file` 就尝试 `apply_text_edit`）
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="fixable", recovery_stage="collect_context")`

#### Scenario: 权限错误被分类为 fatal
- **WHEN** 工具调用抛出 `PermissionError` 或被安全检查阻止且无法恢复
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="fatal", recoverable=False)`

#### Scenario: 未匹配异常默认分类为 fatal
- **WHEN** 异常类型不在 `ErrorClassifier` 的已知映射表中
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="fatal", recoverable=False)`

#### Scenario: 相同错误三次触发 architectural
- **WHEN** 相同 `error_code` 在同一 `failing_stage` 下累计出现三次
- **THEN** `ErrorClassifier.classify()` 返回 `ErrorClassification(category="architectural", recoverable=False, severity="fatal")`

### Requirement: 可扩展的错误映射注册
系统 SHALL 允许通过 `ErrorClassifier.register()` 方法注册额外的异常类型映射，以支持自定义异常类别。

#### Scenario: 注册自定义异常映射
- **WHEN** 调用 `classifier.register(CustomError, category="fixable")`
- **THEN** 后续 `classifier.classify(CustomError(...))` 返回 `category="fixable"`

### Requirement: 错误分类信息丰富度
系统 SHALL 为每个错误分类提供 `error_code`（机器可读标识符）和 `severity`（warn/error/fatal）字段。

#### Scenario: 错误分类包含完整信息
- **WHEN** `ErrorClassifier.classify()` 分类任何异常
- **THEN** 返回的 `ErrorClassification` 包含非空 `error_code`、`category`、`severity` 和 `recoverable` 字段
