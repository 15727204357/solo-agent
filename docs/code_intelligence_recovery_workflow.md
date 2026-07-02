# 代码智能与恢复工作流

Solo Agent 内置了本地 Python 代码智能层。它不是完整 LSP server，也不是外部向量数据库或 Docker 沙箱。它是一个有工作区边界的本地索引，用来帮助 harness 收集可解释的仓库证据。

## 代码智能工具

代码智能工具都是只读且受 workspace 边界限制的：

- `code_index_status(path=".", refresh=false)`：检查或刷新 `.solo-agent/codeintel/index.sqlite3` 下的 Python 索引。
- `code_map(path=".", max_files=500)`：返回模块、类、函数、方法、常量、import、调用边、测试、入口点、解析错误和索引元数据。
- `find_references(symbol, path=".", max_matches=100)`：返回符号定义、import 和引用。
- `analyze_impact(paths=[], symbols=[], include_tests=true)`：根据 import、引用、调用关系和测试相关性估算影响范围。
- `semantic_code_search(query, path=".", max_matches=20)`：使用本地 SQLite FTS5/BM25 加路径、符号和 token 分数做检索。
- `symbol_search`、`symbol_definition`、`call_graph`、`test_relevance`：提供更聚焦的 LSP-like 查询。

对 Python 项目，索引会包含 AST 符号、import、调用、引用、pytest 测试、fixture、marker、docstring、signature 和语法错误。

## 工作流集成

代码智能是路由和验证阶段的证据来源：

- 意图路由可以请求 code map、impact analysis、symbol、reference 或 test relevance 范围。
- `collect_context` 会 materialize 这些范围，并把 compact summary 存入 graph state。
- `select_tools` 应复用 route plan 和已有 snapshots，避免重复做任务分类。
- team mode 会把 code map 和 impact summary 传给 developer / tester。
- 如果 team plan 没有显式验证命令，tester 可以使用 impact analysis 和 test relevance 给出的建议。

## 恢复与重路由

无结果搜索和验证失败不只是最终回答里的解释，它们可以变成 reroute trigger。比如搜索无结果时扩大搜索范围，测试失败时切换到 debug-oriented context，影响分析不足时请求不同的代码智能范围。

## 事件

工作流可能发出：

- `code_index_started`
- `code_index_completed`
- `code_index_stale`
- `code_map_completed`
- `impact_analysis_completed`
- `test_relevance_completed`

## 证据产物

团队和恢复路径可以保留：

- code map summary；
- impact analysis summary；
- suggested verification commands；
- sandbox diff；
- developer tool ledger；
- pytest/ruff 输出；
- supervisor report。

这些证据会服务于补丁审批、route replay 和失败分析。

## 边界

代码智能不负责改文件。它只提供排序后的证据。编辑仍然必须经过工具注册表、verified editing、sandbox/team 边界和审批 gate。
