# Code Intelligence and Recovery Workflow

Solo Agent now has a product-oriented Python code-intelligence layer and a more
product-like long-task recovery path. Code Intelligence v2 is intentionally not
a real LSP process, external vector database, or Docker sandbox dependency. It
is a local LSP-like Python index with persistent SQLite storage, incremental
refresh, call/reference graphs, test relevance, and explainable local retrieval.

## Code Intelligence V2

The code-intelligence tools are read-only and workspace bounded:

- `code_index_status(path=".", refresh=false)` checks or refreshes the
  persistent Python index stored under `.solo-agent/codeintel/index.sqlite3`.
- `code_map(path=".", max_files=500)` returns indexed modules, classes,
  functions, methods, constants, import edges, call edges, test files,
  entrypoints, parse errors, and index metadata.
- `find_references(symbol, path=".", max_matches=100)` returns indexed
  definitions, imports, and references.
- `analyze_impact(paths=[], symbols=[], include_tests=true)` estimates affected
  files through imports, references, call graph signals, and test relevance.
- `semantic_code_search(query, path=".", max_matches=20)` uses local
  SQLite FTS5/BM25 plus path/symbol/token scoring and returns ranking reasons.
- `symbol_search`, `symbol_definition`, `call_graph`, and `test_relevance`
  expose the same index through more focused LSP-like queries.

For Python projects, the index includes AST symbols, imports, calls, references,
pytest tests, fixtures, markers, docstrings, signatures, and syntax errors.
Cross-language indexing, real LSP server integration, and embedding-backed
semantic retrieval are reserved for a later v3.

## Workflow Integration

Code tasks trigger `code_index_status`, `code_map`, and `analyze_impact` during
context collection. The compact results are stored in graph state as
`code_map_summary` and `impact_analysis` so later stages do not need to rescan
unless the run starts fresh.

Team mode passes the impact summary to the developer prompt. The tester uses
impact verification commands when the team plan does not already specify a
targeted command. These commands now come from test relevance scoring where
possible, so the agent can explain what it believes will be affected, which
tests it should run, and why.

The workflow emits:

- `code_index_started`
- `code_index_completed`
- `code_index_stale`
- `code_map_completed`
- `impact_analysis_completed`
- `test_relevance_completed`

## Interrupt And Resume

The web run-control model now distinguishes:

- `paused`: user interrupted without structured feedback.
- `awaiting_feedback`: user interrupted and supplied feedback to apply on
  resume.
- `awaiting_approval`: verified patch or skill change is waiting for approval.
- `cancelled`: user rejected or stopped the run.

`POST /api/sessions/{session_id}/runs/{run_id}/interrupt` records a
`run_interrupted` event and moves the run to `paused` or `awaiting_feedback`.

`POST /api/sessions/{session_id}/runs/{run_id}/resume` can load a checkpoint
state and continue from `team_develop`, `team_test`, or `team_supervisor` with
`checkpoint_id`, `from_node`, `human_feedback`, and `recovery_hints`.

## Sandbox Artifacts

Each isolated team run can retain:

- sandbox root
- sandbox diff
- developer tool ledger
- pytest/ruff output
- developer summary
- code map summary
- impact analysis

`GET /api/sessions/{session_id}/runs/{run_id}/artifacts` returns the latest
artifact view from checkpoint state. If the sandbox root still exists, resume
uses it as the command workspace. If it is missing, the runtime can fall back to
checkpoint state and known patch evidence instead of pretending the sandbox is
still live.
