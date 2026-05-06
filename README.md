# Solo Agent

Solo Agent 是一个 Python 生态的 Web 版个人编程助手 MVP。它不是简单的聊天壳，而是一个可以写进简历的本地 Agent Runtime：有可观测的执行流程、受控工具调用、安全检查、Provider 抽象和 SQLite 持久化。

第一版目标是先做出一个能运行、能演示、能继续扩展的闭环：

用户在 Web 页面输入编程任务，后端启动 Agent，Agent 依次完成规划、上下文收集、安全检查、只读工具调用、流式回复和持久化记录。

## 核心能力

- `Web UI`：在浏览器里提交编程任务，并实时查看 Agent 执行进度。
- `Agent Loop`：按 `plan -> context -> inspect -> tool_call -> respond -> persist` 组织任务。
- `Provider 抽象`：支持 OpenAI-compatible API、DeepSeek 和 Ollama。
- `SQLite 记忆层`：保存 session、run、message、tool_call、timing_point 和 snapshot。
- `多轮 Session 记忆`：同一 session 的后续 run 会读取 recent messages、summary snapshot 和 FTS5 检索结果。
- `Hermes 风格记忆周期`：调用前 `prefetch_all`，回复后 `sync_all`，后台 `queue_prefetch_all`，压缩前 `on_pre_compress`。
- `记忆开关`：可按 run 控制是否保存/使用记忆，以及是否参考历史聊天记录。
- `Fence 防御`：召回记忆通过 `<memory-context>` 注入，并声明它不是新的用户输入。
- `只读工具系统`：支持列文件、读文件、搜索文本，并限制在 workspace 根目录内。
- `安全检查器`：拦截危险删除、密钥泄露、可疑出站访问和重复工具调用。
- `SSE 流式事件`：前端可以看到规划、工具调用、响应生成、持久化等中间过程。
- `LangGraph 1.x 拓扑预留`：使用新版稳定线保留可编译 graph，方便后续接 checkpoint 和更复杂编排。

更多项目背景和简历叙事见 [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)。

## 技术栈

- 后端：FastAPI
- 前端：Jinja2 + 原生 CSS/JS + Server-Sent Events
- Agent 编排：LangGraph 1.x 风格状态流，并预留 TypedDict StateGraph 拓扑
- 模型接入：OpenAI-compatible、DeepSeek、Ollama
- 持久化：SQLite + SQLAlchemy Async + aiosqlite
- 记忆检索：SQLite FTS5 sidecar index，中文短语带 LIKE 降级
- 配置管理：pydantic-settings
- 测试：pytest + pytest-asyncio
- 代码质量：Ruff
- 项目管理：uv

## 项目结构

当前目录按 Web 端 Agent 产品形态拆分，前后端边界更接近 DeerFlow 2.0 这类全栈 Agent 项目：

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
- 安全检查器和只读工具边界。
- Agent 事件流完整执行。
- Web API 创建会话、提交运行和读取 SSE。

## 这一版你能学到什么

这一版适合作为“Agent 工程入门到可展示作品”的第一阶段练习。你可以具体学到：

- 如何从零搭建一个 Python `backend/src` 分层项目，并用 `pyproject.toml`、`uv`、`ruff`、`pytest` 管理工程质量。
- 如何用 FastAPI 设计 Web 后端，包括路由、依赖注入、后台任务、健康检查和 API 测试。
- 如何用 Server-Sent Events 把 Agent 的中间过程流式推给前端，而不是只返回最终答案。
- 如何设计一个 Agent Loop，把规划、上下文、安全检查、工具调用、回答和持久化拆成清晰阶段。
- 如何跟进快速迭代的 Agent 技术栈，把 LangGraph/LangChain Core 升级到新的稳定主线并用测试兜底。
- 如何做 Provider 抽象，让 OpenAI、DeepSeek、Ollama 走统一聊天接口。
- 如何用 SQLAlchemy Async 和 SQLite 设计最小记忆层，保存可调试的运行历史。
- 如何设计只读工具注册表，并把工具限制在 workspace 根目录内，避免越权读取。
- 如何做 deterministic safety checks，用规则先拦截危险删除、密钥泄露和不允许的工具调用。
- 如何写不依赖真实模型网络调用的测试，用 fake provider 测 Agent 事件流。
- 如何把一个简历项目讲成“工程系统”：可观测性、安全边界、持久化、Provider 抽象和后续扩展路线。

## 后续路线

下一阶段可以继续做这些能力：

- 接入真实代码编辑：hash anchor + Tree-sitter。
- 增加 FTS5 记忆检索和 `on_pre_compress` hook。
- 引入 MCP 风格工具协议。
- 做 fast/complete 双 Provider 路由和 fallback。
- 加 checkpoint replay，让每次 Agent 运行可恢复、可复盘。
- 优化 Web UI，把工具调用、TimingPoint、Snapshot 做成更直观的可视化面板。
