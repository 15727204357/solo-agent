# Solo Agent 项目说明

Solo Agent 是一个面向简历展示的 Python Agent 工程项目。当前场景已经升级为“场景 B：团队工程自动化工作流”：团队可以把多个工程任务提交给同一个本地优先的 Agent Runtime，由工作流引擎负责可重复执行、可审计记录和可控并行处理。

当前目标不再是“个人开发者日常编程助手”，也不再停留在“只读 Web MVP”。Solo Agent 要落地成真实可执行的团队工程自动化 Runtime：能运行、能演示、能讲清楚架构，并把任务计划、并行门控、代码编辑、双轮审查、行为策略、记忆、MCP 工具协议、错误恢复和 checkpoint replay 都收敛到可逐步强化的运行时边界。

## 使用场景

团队工程自动化工作流要解决的是这些问题：

- 同一类工程任务需要反复执行，但每次都靠人工拼提示词，难以复用。
- 多个任务可以同时推进，但如果依赖、写入范围和验证路径判断不清，容易互相踩踏。
- AI 回复缺少项目上下文和任务边界，容易把流程决策交给模型临场发挥。
- 工具调用、审查结论和错误恢复过程不可见，难以审计和复盘。
- 代码质量、安全边界和失败处理不稳定，重复失败无法及时升级为架构问题。
- 会话和运行状态没有沉淀，下一次无法复用历史、恢复任务或解释决策来源。

Solo Agent 的团队工作流闭环是：

1. 用户打开本地 Web UI。
2. 用户提交一个任务或一批已知类型的工程任务。
3. 计划层按 superpowers `writing-plans` 风格生成无占位符、2-5 分钟粒度、内联自审的计划。
4. 工作流先判断任务是否满足 4 条独立性条件；全部满足才并行，否则串行。
5. Agent 为每个任务收集有限且可解释的项目上下文。
6. Agent 通过工具注册表调用上下文、质量检查和受控编辑工具。
7. Graph 层行为策略在执行前强制 Superpowers Iron Law、Karpathy 行为约束、read-before-edit 和 hash-anchored preview 协议。
8. 审查层执行双轮审查：第一轮规范合规，第二轮代码质量。
9. 错误处理层使用本地 Hermes 风格 ErrorClassifier 做 4 分类，按单次 run 累计相同异常历史，并用本地 goose 风格 RepetitionInspector 捕捉重复工具调用，防止死循环。
10. 前端通过 SSE 看到规划、并行/串行决策、工具调用、审查、错误处理和持久化过程。
11. SQLite 保存会话、任务、运行、消息、工具调用、时间点、快照和审计线索。

核心设计取向是：已知任务类型用代码定义，任务编排和安全边界由工作流引擎控制，LLM 主要负责上下文理解、计划生成、编辑建议和审查解释。

## 最佳实践地图

这个项目会逐步吸收你指定的最佳实践：

- 工作流引擎：Python 优先 deer-flow/LangGraph 风格；如果走 Go 生态，参考 eino 的类型安全工作流。已知任务类型用代码定义，比让 LLM 决定控制流更可靠。
- 并行前提：superpowers 的 4 条独立性条件全部满足才并行，否则串行。
- 计划层：superpowers `writing-plans`，要求无占位符、2-5 分钟粒度、内联自审。
- 审查机制：双轮审查，第一轮规范合规，第二轮代码质量。
- 错误处理：本地 Hermes 风格 ErrorClassifier 4 分类 +goose fix_conversation + max_compaction_attempts=2；单次 run 内三次相同异常失败等于架构问题。
- 行为层：graph policy engine 强制 superpowers Iron Laws + Karpathy 规则；SKILL.md 只提供 SOP 内容和触发元数据。
- 上下文层：goose 80% + tool_call_cut_off + SubdirectoryHintTracker。
- 记忆层：hermes FTS5 + prefetch_all/sync_all/queue_prefetch_all + on_pre_compress hook。
- 工具层：goose MCP + hermes SKILL.md。
- 安全层：SecurityInspector + EgressInspector +  本地 goose 风格 RepetitionInspector。
- 代码编辑：oh-my-openagent hash 锚定 + goose Tree-sitter。
- 持久化：checkpoint 记录从哪里恢复执行。
- 可观测：callback TimingPoint + 工具进度显示 + snapshot。
- Provider：fast/complete 双路径 + fallback + declarative provider。

## 当前落地范围

当前版本要做到“真实、可演示、可继续扩展”：

- Web UI 优先，不做 CLI。
- 支持真实 OpenAI、DeepSeek、Ollama 配置。
- 实现 LangGraph 1.x 风格 Workflow Loop，并统一使用 TypedDict StateGraph 主编排。
- 将工作流入口从单次个人任务升级为团队工程任务运行，预留任务类型注册、任务批次和并行调度边界。
- 已知任务类型尽量代码化定义，LLM 不直接决定是否并行、是否越权写入或是否跳过审查。
- 并行执行必须先通过独立性门控；不满足条件时默认串行。
- 双轮审查作为 graph 节点：先检查规范合规，再检查代码质量。
- 错误处理将失败归类为可恢复错误、策略违规、环境问题和重复架构失败；相同异常历史按单次 run 追踪，重复工具调用交给 RepetitionInspector。
- SQLite 保存 session、run、message、tool_call、timing_point、snapshot，并用 FTS5 做内部全文索引。
- Hermes 风格记忆周期：调用前召回，回复后同步，后台预热，压缩前保存洞察。
- 工具注册表提供上下文读取、质量检查、hash 锚定编辑预览和受控写入。
- 行为策略层在 graph 中硬执行：生产代码没有失败测试信号就阻断；编辑前必须读代码；`apply_text_edit` 必须具备 hash proof 和 preview。
- 行为违规后优先自动恢复：缺上下文则补 `read_file`/`search_text`，缺 hash/preview 则补 `prepare_edit`/`preview_patch`，不可恢复时才阻断。
- 安全检查器使用确定性规则，先不依赖模型判断。
- 使用 Server-Sent Events 展示 Agent 运行过程。

不再把“只读”作为阶段边界。写入能力必须通过行为策略和工具协议受控落地；Tree-sitter 锚定、MCP 兼容、FTS5 检索、压缩 hook、checkpoint replay 和 provider fallback 继续作为后续增强项。

## 简历叙事

可以这样描述这个项目：

“基于 Python 构建了一个本地优先的团队工程自动化 Agent Runtime，支持 FastAPI Web UI、SSE 流式可观测、OpenAI/DeepSeek/Ollama Provider 抽象、LangGraph 1.x 工作流拓扑、SQLite 会话记忆、受控工具注册表、安全检查器和 graph 层行为策略。系统按 plan/independence gate/context/inspect/tool/review/respond/persist 拆分执行阶段，将已知任务类型代码化，并把 Superpowers Iron Law、Karpathy 行为规则、并行门控、双轮审查、read-before-edit、hash preview、错误分类和自动恢复纳入运行时约束。”

## 工程原则

- 展示 Agent 的过程，而不是只展示最终答案。
- 已知任务类型优先代码化，避免把核心控制流交给模型临场决定。
- 可重复执行、可审计和可恢复优先于一次性聪明回复。
- 只有独立性条件全部满足时才并行，否则串行。
- 审查是工作流节点，不是 prompt 里的善意提醒。
- 在模型判断前，优先使用确定性安全规则。
- 工具要窄、可描述、可审计，并限制在 workspace 内。
- 持久化足够多的状态，方便调试每一次运行。
- 先做小而稳定的兼容接口，再接入重型能力。
- 行为约束必须在 graph/策略层硬执行，不能只依赖 prompt 或 skill 文本提醒。
