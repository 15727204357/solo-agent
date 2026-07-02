# 工具与 Skill 生态

Solo Agent 的工具系统专注 Coding Agent 场景。它不是通用插件市场、浏览器自动化平台、数据分析平台或 SaaS 连接器集合。V1 支持的是检查、修改、验证、审查和沉淀代码工作流所需要的最小能力面。

## 工具分层

- 上下文工具：workspace snapshot、文件发现、受限读取、文本搜索、代码地图、符号搜索、引用、调用关系、影响分析、测试相关性。
- 编辑工具：准备编辑、预览补丁、hash-aware apply、补丁提案和受控文件操作。
- 质量工具：pytest、Ruff、build、typecheck、format check 等结构化命令。
- Git 只读工具：status、diff、show、recent log。
- 记忆工具：最近消息、摘要、内置记忆、路由决策和记忆治理。
- Skill 编排：紧凑 Skill 索引、`skill_view`、Recipe、Recipe preview/run、声明式脚本。
- 团队子 Agent：隔离工作区中的 developer assignment，以及 tester / supervisor 审查。

高风险动作，例如主工作区编辑、删除、移动文件、Skill 变更、写入型脚本、安装依赖、发布、部署和网络型命令，都必须通过提案或审批边界。

## Skill 渐进披露

Skill 不是一开始全部塞进 prompt，而是分阶段加载：

1. 路由阶段只使用紧凑元数据。
2. 显式 Skill 请求或高置信命中时，才加载完整 `SKILL.md`。
3. Recipe 和支持文件只在选中的 Skill 需要时加载。
4. 脚本只有在 metadata 声明且策略允许时才能执行。

这样可以减少 prompt 噪音，也降低无关 Skill 文本影响本轮任务的风险。

## Skill 合约

`SKILL.md` 可以声明：

- `required_tools`；
- `tool_strategy`；
- `acceptance_criteria`；
- `failure_recovery`；
- `metadata.hermes.recipes`；
- `metadata.hermes.scripts`。

路由器需要能解释 Skill 或 Recipe 为什么被选中、为什么降权、为什么阻断，或为什么只作为备选。

## Recipe 边界

声明式 Recipe 可以自动执行低风险的 read、search、git-read、test、build、lint、check 步骤。手动步骤或高风险步骤会被报告为人工动作，不会绕过审批。

Recipe policy 会拒绝不安全模板参数、疑似密钥、shell 元字符、缺少审批的写入步骤，以及引用未知工具的步骤。

## Skill Evolution

Skill evolution 是运行结束后的 Reflection，不是自动自我修改系统。

运行结束后，Solo Agent 可以分析确定性证据，例如阻塞的 Recipe 步骤、成功的安全工具序列、反复出现的恢复模式。当置信度超过阈值时，系统最多创建一个 pending `SkillChangeProposal`。

被提升的 Recipe 必须通过现有 schema 和 policy。提案必须包含安全操作，并且仍然需要审批后才能修改工作区 Skill。低置信发现只会保存在 snapshot 中。

## 团队子 Agent 工具

developer 子 Agent 只在隔离工作区中具备写入能力。它们拿到的工具注册表比 lead agent 更窄，并且会记录工具调用 ledger。supervisor 使用 ledger 和最终 diff 生成主工作区的 verified patch proposal。

这让团队模式拥有真实执行能力，同时避免主仓库被不受控写入。

## 本地环境检查

仓库里的本地包装命令，例如 `rtk`，只能看作开发者环境里的可选便利工具，不是运行时必备能力。如果包装命令不存在，就使用有边界的内置命令，并在验证说明里报告限制。
