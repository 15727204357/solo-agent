# Solo Agent V1

Solo Agent V1 是一个本地优先的 Coding Agent Harness，用来把“让 AI 写代码”这件事变成一条可观察、可验证、可回放的工程链路。

它不是简单聊天壳，也不是演示用工具选择器。用户在 Web 页面提交代码任务后，Solo Agent 会先判断任务意图，规划需要的上下文，选择受控工具，必要时生成补丁提案，运行验证，并把关键决策保存下来，方便审查、恢复和后续改进。

## 项目定位

当前 V1 关注的是 Coding Agent 的 harness 设计，而不是单纯模型调用：

- 面向真实仓库，而不是隔离的演示样例。
- 面向开发闭环，而不是只返回一段回答。
- 面向可审计执行，而不是让模型自由写文件。
- 面向可持续改进，而不是每次都从零开始猜流程。

一句话概括：Solo Agent V1 是一个本地运行的工程化 Coding Agent 运行时，核心价值在于意图路由、上下文管理、记忆系统、Skill 链路、团队子 Agent 编排和补丁审批闭环。

## 项目亮点

### 1. 统一意图路由

Solo Agent 会在执行工具前先生成 route plan，明确本轮任务属于解释问题、检查代码、修改代码、调试测试、运行质量检查、代码审查还是 Skill 管理。

route plan 会同时包含：

- 用户意图和备选意图；
- 需要收集的上下文范围；
- 候选工具和已选工具；
- Skill / Recipe 候选；
- 审批边界和验证策略；
- 风险摘要和决策痕迹；
- 是否需要重路由。

这样可以避免普通 Coding Agent 常见的“模型边聊边猜工具”。

### 2. Context Plan 上下文管理

上下文不是越多越好。Solo Agent 会先规划上下文，再决定读取文件、搜索文本、生成代码地图、做影响分析、查测试相关性、读取 Git 状态或召回记忆。

这种设计让 Agent 能围绕证据工作：知道为什么搜索、期望找到什么、没找到时如何 fallback，减少无效检索和上下文污染。

### 3. ReAct + Reflection 混合循环

执行层采用 ReAct 思路：观察上下文、调用工具、读取结果、继续下一步。

反馈层引入 Reflection 思路：测试失败、工具无结果、补丁被阻塞、上下文置信度不足时，可以触发有界重路由；team tester、supervisor、patch gate、route replay 和 eval harness 都会参与复盘。

它不是“一次调用失败就结束”，而是有清楚边界的迭代式解决问题流程。

### 4. 面向 Coding 的记忆系统

Solo Agent 使用 SQLite + FTS5/BM25 做记忆检索，适合精确召回：

- 文件路径；
- 函数名和类名；
- 报错信息；
- 配置项；
- 历史解决方案；
- 用户偏好和项目事实。

相比纯向量召回，FTS5 更适合 Coding 场景里大量精确 token、路径和错误文本的检索。长期记忆还带有候选、审批、冲突和撤销机制，避免错误信息静默污染记忆。

### 5. Skill / Recipe 渐进披露

Skill 不会一开始全部塞进 prompt。系统先用紧凑元数据做路由，只有命中后才加载完整 `SKILL.md`、Recipe 和支持文件。

Recipe 会区分自动步骤、手动步骤、阻塞原因和审批边界。Skill evolution 可以把成功经验或反复出现的问题沉淀成 Skill/Recipe 变更提案，但不会绕过审批直接修改自身能力。

### 6. 团队模式子 Agent 编排

团队模式不是随意开多个 Agent。Solo Agent 使用固定的工程团队链路：

```text
team_plan -> parallelism_gate -> team_develop -> team_test -> team_supervisor
```

`parallelism_gate` 只负责判断开发任务是否适合拆分、拆成几个 developer。每个 developer 在独立 lightweight workspace 中工作，tester 负责验证，supervisor 负责审查并合并不冲突的 diff。

主工作区不会被 developer 子 Agent 直接修改。

### 7. Verified Editing 补丁审批闭环

代码修改不会直接写入主工作区，而是进入：

```text
生成补丁提案 -> 审批边界 -> 应用补丁 -> 执行验证 -> 记录结果
```

这让 Agent 具备执行能力，同时保留用户对主仓库修改的控制权。

### 8. Route Replay 与 Eval Harness

每次运行都会保存关键事件和路由决策。后续可以回放 route epoch，检查当时为什么选了某个 intent、上下文范围和工具。

Eval harness 支持断言 expected intent、required scopes、forbidden tools、approval required、expected reroute 等字段，为后续优化路由策略提供数据闭环。

## 当前 V1 能力边界

已经具备：

- Web UI 和 SSE 事件流；
- LangGraph `StateGraph` 主运行时；
- OpenAI-compatible、DeepSeek、Ollama、fake provider；
- 统一意图路由和 Route 面板状态；
- Context Plan、上下文预算保护和压缩；
- SQLite 持久化和 FTS5/BM25 记忆检索；
- Python AST 代码智能、code map、影响分析、测试相关性；
- Skill/Recipe 路由、策略和 evolution 提案；
- team developer 隔离工作区，支持 Git worktree/overlay 和 copy fallback；
- verified editing、patch proposal 和审批边界；
- route replay 和 eval scoring；
- 后端、前端回归测试。

暂不宣称：

- Docker / VM 级真实沙箱；
- 完全自主自进化；
- 任意子 Agent 拓扑；
- 模型输出直接改主仓库；
- 外部向量数据库依赖。

## 技术栈

- 后端：Python、FastAPI、LangGraph、SQLAlchemy Async、aiosqlite
- 前端：React、TypeScript、Vite、Server-Sent Events
- 记忆：SQLite、FTS5、BM25
- 代码智能：Python AST、本地 SQLite 索引、FTS5/BM25 检索
- 测试：pytest、pytest-asyncio、Ruff、Vitest
- 项目管理：uv、npm

## 快速开始

### 1. 克隆项目

```powershell
git clone https://github.com/15727204357/solo-agent.git
cd solo-agent
```

### 2. 安装后端依赖

推荐使用 `uv`：

```powershell
uv sync --extra dev
```

如果没有 `uv`，也可以使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 3. 配置模型 Provider

本地 Ollama 示例：

```powershell
$env:SOLO_AGENT_PROVIDER="ollama"
$env:SOLO_AGENT_MODEL="llama3.1"
$env:SOLO_AGENT_BASE_URL="http://localhost:11434"
```

OpenAI-compatible 示例：

```powershell
$env:SOLO_AGENT_PROVIDER="openai"
$env:SOLO_AGENT_MODEL="gpt-4.1-mini"
$env:SOLO_AGENT_API_KEY="your-api-key"
```

DeepSeek 示例：

```powershell
$env:SOLO_AGENT_PROVIDER="deepseek"
$env:SOLO_AGENT_MODEL="deepseek-chat"
$env:SOLO_AGENT_API_KEY="your-api-key"
```

也可以复制 `.env.example` 为 `.env` 后统一配置。

### 4. 启动后端

```powershell
uv run solo-agent-web
```

打开：

```text
http://127.0.0.1:8000
```

### 5. 前端开发模式

如果需要单独开发前端：

```powershell
cd frontend
npm install
npm run dev
```

## 常用配置

```powershell
$env:SOLO_AGENT_MEMORY_ENABLED="true"
$env:SOLO_AGENT_CONVERSATION_HISTORY_ENABLED="true"
$env:SOLO_AGENT_INTENT_ROUTER_MODE="shadow_hybrid"
$env:SOLO_AGENT_INTENT_ROUTER_MODEL_TIMEOUT_SECONDS="1.5"
$env:SOLO_AGENT_SUBAGENT_ENABLED="false"
```

说明：

- `SOLO_AGENT_INTENT_ROUTER_MODE=rules`：只使用规则路由，延迟最低。
- `SOLO_AGENT_INTENT_ROUTER_MODE=shadow_hybrid`：默认推荐，模型建议只做影子记录，不直接改变执行。
- `SOLO_AGENT_INTENT_ROUTER_MODE=hybrid`：允许模型建议参与排序，但仍不能绕过 guardrail。
- `SOLO_AGENT_SUBAGENT_ENABLED=true`：允许 Plan Mode 进入团队子 Agent 链路。

## 运行测试

后端测试：

```powershell
uv run --extra dev python -m pytest -q
```

后端代码检查：

```powershell
uv run --extra dev ruff check .
```

前端测试和构建：

```powershell
cd frontend
npm test
npm run build
```

## 项目结构

```text
backend/
  src/solo_agent/      # 后端、Agent runtime、workflow、memory、tools、skills
  tests/               # 后端单元测试和集成测试
frontend/
  src/                 # React/TypeScript 前端源码
  templates/           # FastAPI 服务的 HTML 壳
  static/              # 静态资源和构建产物
docs/                  # 产品与架构文档
skills/                # 工作区 Skill 和 Recipe
data/                  # 本地运行数据
```

## 文档入口

- [工作流编排](docs/WORKFLOW_ORCHESTRATION.md)
- [团队 Coding 工作流](docs/solo_agent_coding_workflow.md)
- [并行门控](docs/workflow_parallelism.md)
- [上下文管理](docs/context-management.md)
- [工具与 Skill 生态](docs/tool_ecosystem.md)
- [代码智能与恢复](docs/code_intelligence_recovery_workflow.md)
- [项目说明](docs/PROJECT_BRIEF.md)
- [记忆层](backend/src/solo_agent/memory/README.md)

## 适合如何使用

Solo Agent V1 适合用来：

- 学习 Coding Agent Harness 的工程结构；
- 研究意图路由、上下文规划和工具治理；
- 实验本地记忆、Skill/Recipe 和 route eval；
- 构建可审计的 AI 编码工作流；
- 作为简历项目展示完整 Agent 工程闭环。
