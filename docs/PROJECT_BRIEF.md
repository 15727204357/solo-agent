# Solo Agent 项目说明

Solo Agent 是一个面向简历展示的 Python Agent 项目。它演示的是“个人开发者日常编程助手”场景：透明规划、受控工具调用、项目上下文、记忆、安全检查、Provider 抽象和流式可观测。

第一阶段不追求做成大而全的平台，而是先完成一个 Web MVP：能运行、能演示、能讲清楚架构，并为后续代码编辑、FTS5 记忆、MCP 工具协议和 checkpoint replay 留好扩展接口。

## 使用场景

个人开发者在日常编程中经常遇到这些问题：

- 重复探索代码库，浪费时间。
- AI 回复缺少上下文，容易走偏。
- 工具调用不可见，难以调试。
- 代码质量和安全边界不稳定。
- 会话没有沉淀，下一次无法复用历史。

Solo Agent 的第一版闭环是：

1. 用户打开本地 Web UI。
2. 用户输入一个编程任务。
3. Agent 生成短计划。
4. Agent 收集有限的项目上下文。
5. Agent 通过工具注册表调用只读工具。
6. Inspector 在执行前拦截危险请求和危险工具调用。
7. 前端通过 SSE 看到完整运行过程。
8. SQLite 保存会话、消息、工具调用、时间点和快照。

## 最佳实践地图

这个项目会逐步吸收你指定的最佳实践：

- 行为层：superpowers Iron Laws + Karpathy 规则。
- 计划层：superpowers writing-plans。
- 上下文层：goose 80% + tool_call_cut_off + SubdirectoryHintTracker。
- 记忆层：hermes FTS5 + prefetch_all/sync_all/queue_prefetch_all + on_pre_compress hook。
- 工具层：goose MCP + hermes SKILL.md。
- 安全层：SecurityInspector + EgressInspector + RepetitionInspector。
- 错误处理：ErrorClassifier + goose fix_conversation。
- 代码编辑：oh-my-openagent hash 锚定 + Tree-sitter。
- 持久化：SQLite v11 SessionType + checkpoint。
- 可观测：callback TimingPoint + 工具进度显示 + snapshot。
- Provider：fast/complete 双路径 + fallback + declarative provider。

## 第一阶段范围

第一阶段要做到“真实、可演示、可继续扩展”：

- Web UI 优先，不做 CLI。
- 支持真实 OpenAI、DeepSeek、Ollama 配置。
- 实现 LangGraph 1.x 风格 Agent Loop，并预留 TypedDict StateGraph 拓扑。
- SQLite 保存 session、run、message、tool_call、timing_point、snapshot，并用 FTS5 做内部全文索引。
- Hermes 风格记忆周期：调用前召回，回复后同步，后台预热，压缩前保存洞察。
- 只允许只读工具：列文件、读文件、搜索文本。
- 安全检查器使用确定性规则，先不依赖模型判断。
- 使用 Server-Sent Events 展示 Agent 运行过程。

第一阶段刻意不做真实代码写入。代码编辑、Tree-sitter 锚定、MCP 兼容、FTS5 检索、压缩 hook、checkpoint replay 和 provider fallback 会放到后续阶段。

## 简历叙事

可以这样描述这个项目：

“基于 Python 构建了一个本地优先的个人编程 Agent Runtime，支持 FastAPI Web UI、SSE 流式可观测、OpenAI/DeepSeek/Ollama Provider 抽象、LangGraph 1.x 拓扑、SQLite 会话记忆、只读工具注册表和安全检查器。系统按 plan/context/inspect/tool/respond/persist 拆分执行阶段，为后续 Tree-sitter 代码编辑、FTS5 记忆检索和 checkpoint replay 预留扩展接口。”

## 工程原则

- 展示 Agent 的过程，而不是只展示最终答案。
- 在模型判断前，优先使用确定性安全规则。
- 工具要窄、可描述、可审计，并限制在 workspace 内。
- 持久化足够多的状态，方便调试每一次运行。
- 先做小而稳定的兼容接口，再接入重型能力。
- 第一阶段要稳定可运行，后续阶段再逐步变得惊艳。
