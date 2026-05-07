# Solo Agent

Solo Agent 是一个 Python 生态的 Web 版团队工程自动化工作流 Runtime。它不是简单的聊天壳，而是一个可以写进简历的本地 Agent 工程系统：围绕可重复执行、可审计、可并行处理多个任务，提供可观测的执行流程、受控工具调用、graph 层行为策略、安全检查、Provider 抽象和 SQLite 持久化。

当前目标是做出一个能运行、能演示、能继续扩展，并能把团队工程任务真实受控执行起来的闭环：

团队成员在 Web 页面提交一个或多个工程任务，后端启动工作流运行，Agent 依次完成任务计划、独立性判断、上下文收集、安全检查、行为策略评估、受控工具调用、双轮审查、错误分类、流式回复和审计持久化。已知任务类型优先用代码定义，比让 LLM 临场决定控制流更可靠。

## 核心能力

- `Web UI`：在浏览器里提交编程任务，并实时查看 Agent 执行进度。
- `Workflow Loop`：按 `plan -> independence_gate -> context -> inspect -> tool_call -> review -> respond -> persist` 组织任务。
- `计划层`：吸收 superpowers `writing-plans` 实践，计划无占位符、2-5 分钟粒度，并内联自审。
- `并行门控`：只有满足 superpowers 的 4 条独立性条件时才并行处理任务，否则自动回到串行执行。
- `双轮审查`：第一轮检查规范合规，第二轮检查代码质量，把审查作为工作流节点而不是事后提醒。
- `错误处理`：使用本地 Hermes 风格 ErrorClassifier 做 4 类错误归因，按单次 run 累计相同异常历史，并结合本地 goose 风格 RepetitionInspector 识别重复工具调用。
- `Provider 抽象`：支持 OpenAI-compatible API、DeepSeek 和 Ollama。
- `SQLite 记忆层`：保存 session、run、message、tool_call、timing_point 和 snapshot。
- `多轮 Session 记忆`：同一 session 的后续 run 会读取 recent messages、summary snapshot 和 FTS5 检索结果。
- `Hermes 风格记忆周期`：调用前 `prefetch_all`，回复后 `sync_all`，后台 `queue_prefetch_all`，压缩前 `on_pre_compress`。
- `记忆开关`：可按 run 控制是否保存/使用记忆，以及是否参考历史聊天记录。
- `Fence 防御`：召回记忆通过 `<memory-context>` 注入，并声明它不是新的用户输入。
- `行为策略层`：在 graph 中硬执行 Superpowers Iron Law、Karpathy 行为规则、read-before-edit、hash preview 和自动恢复。
- `受控工具系统`：支持列文件、读文件、搜索文本、质量检查、hash 锚定编辑预览和受控写入，并限制在 workspace 根目录内。
- `安全检查器`：拦截危险删除、密钥泄露、可疑出站访问和重复工具调用。
- `SSE 流式事件`：前端可以看到规划、并行/串行决策、工具调用、审查、响应生成、持久化等中间过程。
- `LangGraph 1.x 拓扑预留`：Python 侧优先使用 deer-flow/LangGraph 风格工作流；如果迁移到 Go 生态，选型可参考 eino 的类型安全工作流。

更多项目背景和简历叙事见 [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)。

## 技术栈

- 后端：FastAPI
- 前端：Jinja2 + 原生 CSS/JS + Server-Sent Events
- Agent 编排：LangGraph 1.x 风格状态流，并预留 TypedDict StateGraph 拓扑
- 工作流选型：Python 优先 deer-flow/LangGraph 风格；Go 生态可参考 eino 类型安全工作流
- 模型接入：OpenAI-compatible、DeepSeek、Ollama
- 持久化：SQLite + SQLAlchemy Async + aiosqlite
- 记忆检索：SQLite FTS5 sidecar index，中文短语带 LIKE 降级
- 配置管理：pydantic-settings
- 测试：pytest + pytest-asyncio
- 代码质量：Ruff
- 项目管理：uv

## 项目结构

当前目录按 Web 端团队工程自动化产品形态拆分，前后端边界更接近 DeerFlow 2.0 这类全栈 Agent 工作流项目：

```text
backend/
  src/solo_agent/      # Python 后端、Agent runtime、API、Provider、Memory、Tools
  tests/               # 后端测试
frontend/
  templates/           # Jinja2 页面模板
  static/              # 原生 CSS/JS 静态资源
docs/                  # 产品与工程文档
skills/                # Agent skill/SOP
data/                  # 本地运行数据
```

## 本地运行

推荐使用 `uv`：

```powershell
uv sync --extra dev
uv run solo-agent-web
```

然后打开：

```text
http://127.0.0.1:8000
```

也可以使用传统虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
solo-agent-web
```

## Provider 配置

默认使用 Ollama，适合本地演示：

```powershell
$env:SOLO_AGENT_PROVIDER="ollama"
$env:SOLO_AGENT_MODEL="llama3.1"
$env:SOLO_AGENT_BASE_URL="http://localhost:11434"
```

使用 OpenAI：

```powershell
$env:SOLO_AGENT_PROVIDER="openai"
$env:SOLO_AGENT_MODEL="gpt-4.1-mini"
$env:SOLO_AGENT_API_KEY="你的 API Key"
```

使用 DeepSeek：

```powershell
$env:SOLO_AGENT_PROVIDER="deepseek"
$env:SOLO_AGENT_MODEL="deepseek-chat"
$env:SOLO_AGENT_API_KEY="你的 API Key"
```

也可以复制 `.env.example` 为 `.env`，统一管理环境变量。

记忆开关：

```powershell
$env:SOLO_AGENT_MEMORY_ENABLED="true"
$env:SOLO_AGENT_CONVERSATION_HISTORY_ENABLED="true"
```

## HTTP API

- `GET /`：打开 Web UI。
- `GET /api/health`：健康检查。
- `POST /api/sessions`：创建会话。
- `GET /api/sessions`：获取会话列表。
- `GET /api/sessions/{session_id}`：获取单个会话和运行记录。
- `GET /api/sessions/{session_id}/messages`：获取当前 session 的多轮消息历史。
- `POST /api/sessions/{session_id}/runs`：提交一次 Agent 运行。
- `GET /api/sessions/{session_id}/runs/{run_id}/events`：通过 SSE 流式读取运行事件。

提交 run 时可以覆盖记忆开关：

```json
{
  "prompt": "继续刚才的设计",
  "memory_enabled": true,
  "conversation_history_enabled": true
}
```

## 测试

运行测试：

```powershell
uv run --extra dev python -m pytest -q
```

代码检查：

```powershell
uv run --extra dev ruff check .
```

当前测试覆盖：

- Provider 配置和工厂创建。
- SQLite session、run、message、tool_call、timing_point、snapshot 写入。
- 多轮消息历史、FTS5 session 内检索、summary snapshot。
- Hermes memory lifecycle、`MEMORY.md` / `USER.md` 内置记忆、fence escape 防御。
- 安全检查器和受控工具边界。
- Agent 事件流完整执行。
- Web API 创建会话、提交运行和读取 SSE。

## 这一版你能学到什么

这一版适合作为“Agent 工程入门到团队工作流自动化”的第一阶段练习。你可以具体学到：

- 如何从零搭建一个 Python `backend/src` 分层项目，并用 `pyproject.toml`、`uv`、`ruff`、`pytest` 管理工程质量。
- 如何用 FastAPI 设计 Web 后端，包括路由、依赖注入、后台任务、健康检查和 API 测试。
- 如何用 Server-Sent Events 把 Agent 的中间过程流式推给前端，而不是只返回最终答案。
- 如何设计一个团队工程 Workflow Loop，把规划、独立性判断、上下文、安全检查、工具调用、审查、回答和持久化拆成清晰阶段。
- 如何把已知工程任务类型代码化，让 LLM 负责上下文理解和执行辅助，而不是负责所有流程决策。
- 如何设计并行前提：只有任务互不依赖、写入范围不冲突、上下文足够独立、验证路径独立时才并行，否则串行。
- 如何把双轮审查纳入运行时：先做规范合规审查，再做代码质量审查。
- 如何用 ErrorClassifier 和 RepetitionInspector 区分可恢复错误、策略违规、环境问题和重复架构失败。
- 如何跟进快速迭代的 Agent 技术栈，把 LangGraph/LangChain Core 升级到新的稳定主线并用测试兜底。
- 如何做 Provider 抽象，让 OpenAI、DeepSeek、Ollama 走统一聊天接口。
- 如何用 SQLAlchemy Async 和 SQLite 设计最小记忆层，保存可调试的运行历史。
- 如何设计受控工具注册表，并把工具限制在 workspace 根目录内，避免越权读取或越权写入。
- 如何做 deterministic safety checks，用规则先拦截危险删除、密钥泄露和不允许的工具调用。
- 如何写不依赖真实模型网络调用的测试，用 fake provider 测 Agent 事件流。
- 如何把一个简历项目讲成“工程系统”：可观测性、安全边界、持久化、Provider 抽象和后续扩展路线。

## 后续路线

下一阶段可以继续做这些能力：

- 接入真实代码编辑：hash anchor + Tree-sitter。
- 完善团队任务工作流：任务类型注册、并行调度、依赖图和审计日志。
- 增强双轮审查：规范合规检查、代码质量检查和自动修复建议。
- 增加 FTS5 记忆检索和 `on_pre_compress` hook。
- 引入 MCP 风格工具协议。
- 做 fast/complete 双 Provider 路由和 fallback。
- 加 checkpoint replay，让每次 Agent 运行可恢复、可复盘。
- 优化 Web UI，把工具调用、TimingPoint、Snapshot 做成更直观的可视化面板。
