# Migration: LangGraph StateGraph as Sole Runtime

## What Changed

Solo Agent's workflow orchestration has been unified from multiple runtime paths to a single LangGraph StateGraph.

### Removed
- **`workflow_engine` setting** — no longer configurable; LangGraph is the only engine
- **`_run_text_provider_strategy_legacy`** — manual stage loop deleted
- **`_run_lead_agent_strategy`** — independent lead-agent runtime deleted (subagent/sandbox internals migrated to graph nodes)
- **`_run_langgraph_text_provider_strategy`** — inlined into `WorkflowRuntime.run()`
- **`parallel_dispatch_placeholder_node`** — replaced with real parallel dispatch node

### What Stays the Same
All feature layers are preserved — they're now graph nodes instead of stage loop iterations:
- Planning layer (deep plan, quality validation, self-review ✓)
- Memory layer (prefetch, build context, sync, compress ✓)
- Context management (budget evaluation, compression ✓)
- Safety inspectors (dangerous deletion, secret leak, repetition ✓)
- Tool registry (20+ workspace-bounded tools ✓)
- Verified editing (hash-anchored patches, approval flow ✓)
- Subagent dispatch (read-only fan-out) — moved to graph parallel path
- SQLite persistence (sessions, runs, messages, snapshots, patches ✓)

## Test Checklist

- [ ] All existing tests pass with no `workflow_engine` reference
- [ ] `grep -R "workflow_engine" backend/src` returns nothing
- [ ] `grep -R "_run_text_provider_strategy_legacy" backend/src` returns nothing
- [ ] `grep -R "_run_lead_agent_strategy" backend/src` returns nothing
- [ ] `grep -R "parallel_dispatch_placeholder" backend/src` returns nothing
- [ ] `build_main_workflow_graph()` is the only graph builder in use
- [ ] Plan mode runs through graph (not runtime branch)
- [ ] Agent serial/parallel are graph route decisions (not runtime if/else)

## Rollback

If regression is found, the old `WorkflowRuntime` logic is preserved in git history.
