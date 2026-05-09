# Solo Agent Workflow Orchestration

## Architecture

Solo Agent uses a single **LangGraph StateGraph** as its sole workflow orchestration engine. All execution modes — plan mode, agent serial, agent parallel, verified editing, error recovery, and checkpoint/replay — are implemented as graph nodes and conditional routes within this one graph.

## Graph Topology

The main workflow graph is built by `build_main_workflow_graph()` in `backend/src/solo_agent/workflow/graphs.py`.

```
START
  → receive_user_turn
  → memory_enabled_route
     → [enabled] load_builtin_memory → prefetch_memory → build_memory_context
     → [disabled] skip_memory
  → skill_context
  → context_guard_before_plan
  → run_mode_route
     → [plan] deep_plan → plan_quality_gate → [passed] plan_self_review → plan_response → postlude
                      → [failed] deep_plan_revision → plan_quality_gate (max 2 revisions)
     → [agent] plan → task_state → parallelism_gate
        → [serial] collect_context → inspect → select_tools → execute_tools → spec_compliance_review
        → [parallel] parallel_dispatch → wait_subagents → supervisor_review
  → propose_verified_patch (if patch needed)
     → [awaiting_approval] END
     → [no patch] subdirectory_hint → context_guard_before_respond → respond
  → postlude: sync_memory → queue_prefetch → compress_memory → persist_snapshot → END
```

### Error Recovery Routing

Every node's exception is caught and routed through:
```
classify_error → recovery_route
  → [recoverable] recovery_action → retry
  → [policy_violation] blocked_response → postlude
  → [environment_error] environment_error_response → postlude
  → [architecture_failure] architecture_failure_response → postlude
```

## Node Definitions

### Prelude Nodes
- **receive_user_turn**: Initialize run state (loop stage, run mode, memory/skill budgets)
- **load_builtin_memory**: Load MEMORY.md and USER.md files
- **prefetch_memory**: Load conversation history, summary, retrieved memories
- **build_memory_context**: Build `<memory-context>` fence block
- **skill_context**: Select and load relevant skills
- **context_guard_before_plan**: Check context budget before planning

### Plan Route
- **deep_plan**: Generate Superpowers-style implementation plan
- **plan_quality_gate**: Deterministic validation (no placeholders, concrete steps, 2-5 min granularity)
- **deep_plan_revision**: Revise plan after quality gate failure (max 1 revision)
- **plan_self_review**: LLM self-review of final plan

### Agent Route
- **plan**: Generate task plan
- **parallelism_gate**: Deterministic independence check (problem_domain, context, write_set, verification)
- **serial path**: collect_context → inspect → select_tools → execute_tools
- **parallel path**: parallel_dispatch → wait_subagents → supervisor_review

## State Model

See `backend/src/solo_agent/workflow/graph_state.py` and `backend/src/solo_agent/agent/state.py` for the complete state model.

## Checkpoint / Replay

- SQLite checkpointer is the default (`workflow_checkpointer: "sqlite"`)
- Checkpoint refs are persisted as snapshots
- API endpoints: `GET /checkpoints`, `GET /graph`, `POST /resume`, `POST /replay`
- Replay supports dry-run mode (write tools blocked)
