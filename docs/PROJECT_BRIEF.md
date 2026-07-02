# Solo Agent 项目说明

Solo Agent 是一个本地优先的 Coding Agent Harness，用来处理真实仓库里的代码任务。项目重点不是模型本身，而是 harness：如何让 Agent 规划上下文、选择工具、管理记忆、使用 Skill、编排子 Agent、验证结果，并留下可审计的运行痕迹。

V1 不追求在模型能力上超过云端 Coding Agent。它的价值在于：控制链路显式、本地可运行、过程可观察、结果可测试。

## 产品定位

Solo Agent 把 Coding Agent 看作一个执行系统，而不是一次聊天补全。

一次 run 应该能回答这些问题：

- 用户到底想做什么？
- 需要哪些上下文，为什么需要？
- 哪些工具是候选，哪些被选择，哪些被拒绝或阻断？
- 哪些 Skill 或 Recipe 相关？
- 哪些步骤能自动执行，哪些必须审批？
- 什么证据证明结果可靠？
- 哪些经验值得沉淀成记忆、评测样本或 Skill 变更提案？

## 设计策略

### 意图路由作为控制中枢

意图路由是第一层决策入口。它把用户请求转成 route plan，里面包含 intent、备选意图、context plan、tool plan、skill plan、recipe plan、approval plan、verification plan、decision trace、reroute triggers、risk summary 和 next actions。

这样可以避免每个阶段都用自己的启发式重新猜任务类型。

### ReAct + Reflection 混合循环

执行层是 ReAct 风格：观察仓库上下文，通过工具行动，再读取工具结果。反馈层是 Reflection 风格：tester 反馈、supervisor 审查、outcome judge、patch gate、memory candidate、route replay 和 eval 都会对执行结果做复盘。

系统不是一直调用工具，而是能判断证据不足、补丁被阻塞、测试失败，或者某个经验应该沉淀为记忆或 Skill 提案。

### 上下文是一种需要规划的资源

上下文不是越多越好，而是要围绕证据收集。route plan 可以请求路径读取、文本搜索、代码地图、符号搜索、引用、影响分析、测试相关性、Git 范围和记忆范围。每个范围都服务于一个预期证据。

运行内 summary 和 snapshot 可以减少重复扫描。只读上下文工具可以并发，写入动作仍然受控。

### 面向 Coding 的记忆系统

记忆系统围绕精确召回设计。SQLite FTS5/BM25 比纯向量检索更适合路径、符号、报错、配置名和历史修复方案。系统会保存最近消息、压缩摘要、内置 `MEMORY.md` / `USER.md`、路由决策、审查报告、子 Agent 运行记录和候选记忆。

长期记忆不是自动写入。候选记忆必须经过治理才能成为正式记忆。

### Skill 与自我改进基础

Skill 采用渐进披露：先看紧凑元数据，真正命中后才加载完整 Skill，Recipe 和支持文件只在需要时加载。Skill evolution 是受控的自我改进链路。成功经验或阻塞模式可以生成 pending Skill/Recipe 提案，但不会绕过审批直接修改自己的能力。

这种设计让系统有积累可复用工作流的路径，同时避免静默污染长期行为。

### 团队模式

团队模式不是任意 Agent swarm，而是小型工程团队结构。`parallelism_gate` 判断计划应该由一个 developer 做，还是拆成多个 developer assignment。developer 在隔离轻量工作区里执行，tester 验证，supervisor 把不冲突的 diff 合并成 verified patch proposal。

关键边界是：developer 子 Agent 可以执行，但不能直接写主工作区。

## 当前 V1 范围

当前已经落地的能力包括：

- LangGraph `StateGraph` 运行时；
- Web UI 和 SSE 事件流；
- OpenAI-compatible、DeepSeek、Ollama 和 fake provider；
- 统一意图路由事件和前端 Route 状态；
- Context Plan、context guard 和上下文压缩；
- SQLite 持久化和 FTS5 记忆检索；
- Python 代码智能：AST 索引、code map、引用搜索、影响分析、测试相关性、FTS5/BM25 搜索；
- 有边界的工具注册表；
- verified editing 和补丁审批边界；
- Skill/Recipe 路由、策略、覆盖检查和 evolution 提案；
- team developer 工作区，支持 Git worktree/overlay 和 copy fallback；
- route replay 和 route-aware eval scoring；
- 后端和前端回归测试。

## V1 不做什么

- 不做 Docker 或虚拟机级别沙箱。
- 不开放任意子 Agent 拓扑。
- 不允许模型输出直接修改主工作区。
- 不允许 Skill 自动修改自己而不经过审批。
- 不依赖外部向量数据库。
- 不宣称已经实现完全自主自进化。

## 简历叙事

可以这样概括：

> 构建了一个本地优先的 Coding Agent Harness，围绕意图路由、上下文规划、FTS5 记忆、Skill/Recipe 渐进披露、团队子 Agent、verified editing、route replay 和 evals 形成闭环。系统结合 ReAct 式执行循环和 Reflection 式反馈机制，通过 tester feedback、supervisor review、patch gate、memory candidate 和 route eval 让 Agent 决策可解释、可回放、可持续改进。

## 工程原则

- 路由决策必须显式、可回放。
- 上下文要有范围、原因、预算和 fallback。
- 记忆必须治理，不能静默污染。
- Skill 要渐进加载，避免无关内容干扰任务。
- 子 Agent 只能在有边界的工作区里执行。
- 文件修改必须变成补丁提案，再经过审批。
- 模型建议不能绕过确定性 guardrail。
- 失败案例要沉淀为 eval、route decision 或 Skill 提案。
