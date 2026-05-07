## ADDED Requirements

### Requirement: fix_conversation 管道执行
系统 SHALL 在每次 LLM 调用前执行 `fix_conversation()` 管道，按顺序应用 8 步消息清洗函数，返回修复后的消息列表。

#### Scenario: 8 步管道按顺序执行
- **WHEN** `fix_conversation(messages)` 被调用
- **THEN** 系统按以下顺序应用修复函数：merge_text_content_items → trim_assistant_text_whitespace → remove_empty_messages → fix_empty_tool_results → fix_tool_calling → merge_consecutive_messages → fix_lead_trail → populate_if_empty

#### Scenario: 合并连续文本内容块
- **WHEN** assistant 消息包含多个连续 Text content block（如 `[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]`）
- **THEN** 这些 text content block 被合并为单个 `{"type": "text", "text": "a\nb"}`

#### Scenario: 去除 assistant 消息尾部空白
- **WHEN** assistant 消息的 text content 以空白字符结尾（如 `"hello   \n"`）
- **THEN** 尾部空白被移除，消息变为 `"hello"`

#### Scenario: 移除空内容消息
- **WHEN** 消息的 content 为空字符串、空列表或仅含空白
- **THEN** 该消息从列表中移除

#### Scenario: 空工具结果添加占位
- **WHEN** ToolMessage 的 content 为空字符串或 None
- **THEN** content 被替换为 `"(empty result)"`

#### Scenario: 移除孤立工具调用
- **WHEN** AIMessage 包含 `tool_calls` 但对应 `ToolMessage` 的 `tool_call_id` 不存在于回复消息中
- **THEN** 该孤立 `tool_call` 从 AIMessage 中移除

#### Scenario: 合并连续相同角色消息
- **WHEN** 两个连续的 AIMessage（不含 tool_calls）出现
- **THEN** 它们的内容被合并为单个 AIMessage

#### Scenario: 确保对话以 user 开始
- **WHEN** 消息列表的第一条不是 user/human 消息
- **THEN** 开头的非 user 消息被移除，直到遇到 user 消息

#### Scenario: 空对话填充
- **WHEN** `fix_conversation()` 接收空消息列表
- **THEN** 返回包含单个 `HumanMessage(content="Hello")` 的列表

### Requirement: 修复函数独立可开关
系统 SHALL 允许通过配置控制每个 fix 函数的启用/禁用，支持在紧急情况下降级关闭特定修复步骤。

#### Scenario: 禁用特定修复步骤
- **WHEN** 创建 `ConversationRepair` 实例并设置 `disabled_fixes={"merge_consecutive_messages"}`
- **THEN** `fix_conversation()` 执行时跳过合并连续消息步骤，执行其余 7 步

### Requirement: LangChain 消息格式兼容
系统 SHALL 以 `langchain.schema.messages` 格式（`HumanMessage`、`AIMessage`、`ToolMessage`）作为输入输出，与项目现有 `ChatMessage` TypedDict 兼容。

#### Scenario: 接受和返回 LangChain 消息
- **WHEN** `fix_conversation()` 接收 `list[BaseMessage]`（包含 HumanMessage, AIMessage, ToolMessage）
- **THEN** 返回修复后的 `list[BaseMessage]`，保持相同消息类型
