# Workflow Engine v2 — LangGraph StateGraph Runtime

> Archived note: this page describes the original LangGraph migration design and
> contains legacy placeholder details. The active coding-agent workflow is
> documented in `docs/solo_agent_coding_workflow.md`.

## Why StateGraph

The original Solo Agent text-provider workflow used a hard-coded async stage loop (`_run_text_provider_strategy`) that manually called stage functions in sequence. This approach:

- Made conditional routing (blocked / awaiting-approval / parallel) implicit and error-prone
- Required manual `if/return` checks scattered through the stage list
- Provided no built-in pause/resume or checkpoint capability
- Could not easily support structured parallelism

The v2 engine replaces the manual stage loop with a compiled LangGraph `StateGraph`. LangGraph provides:

- **Conditional routing** via `add_conditional_edges` — routing logic is declared upfront
- **Checkpoint persistence** — in-memory or SQLite-backed state saves at each step
- **Streaming** — native `astream()` support for real-time event delivery
- **Interrupt/resume** — `interrupt()` for human-in-the-loop (future work)

## Preserved Solo Agent Mechanisms

This upgrade is **not a rewrite**. Every existing mechanism is preserved unchanged:

- **Memory**: session memory loading, builtin memory, prefetch, sync, compress, context block
- **Context**: `ContextManager`, `ContextTokenEstimator`, context guard, tool output cutoff, `SubdirectoryHintTracker`
- **Error handling**: `ErrorClassifier`, `BehaviorPolicy.classify_error()`, retry/fix/fatal/architectural categories
- **Safety**: `BehaviorPolicy`, `SecurityInspector` / `EgressInspector` / `RepetitionInspector`, Iron Law blocking
- **Verified editing**: patch proposal, hash/preview/apply, approval boundary, pytest/ruff verification
- **Parallelism gate**: `TaskCandidate`, `ParallelismDecision`, all four independence checks, fail-closed serial fallback
- **Web/SSE**: same `AgentEvent` objects, same event types, same streaming contract

## Graph Shape

The text-provider graph covers the core workflow (plan through respond). Shared prelude (receive_user_turn, memory loading, skill context) and shared postlude (sync_memory, compress_memory, persist_snapshot) remain in `WorkflowRuntime.run()` outside the graph.

```text
START
  -> plan
  -> task_state
  -> parallelism_gate
  -> route_after_parallelism_gate
       parallel -> parallel_dispatch_placeholder (placeholder node)
       serial -> collect_context
       blocked -> END
  -> collect_context
  -> inspect
  -> route_after_inspect
       blocked -> END
       continue -> select_tools
  -> select_tools
  -> execute_tools
  -> route_after_execute_tools
       awaiting_approval -> END
       continue -> propose_verified_patch
  -> propose_verified_patch
  -> route_after_patch
       awaiting_approval -> END
       continue -> subdirectory_hint
  -> subdirectory_hint
  -> context_guard_before_respond
  -> route_after_guard
       blocked -> END
       continue -> respond
  -> respond
  -> END
```

### Node Adapters

Each graph node is a thin adapter that:

1. Reconstructs `AgentState` from serialized graph state (`agent_state_from_graph_data`)
2. Calls the existing async generator stage function
3. Collects emitted `AgentEvent` dicts
4. Stores updated `AgentState` back to graph state

No stage business logic was modified. The adapter pattern ensures existing behavior is untouched.

## Graph State Architecture

Graph state (`SoloGraphState`) is a plain `dict[str, Any]` with these keys:

- `agent_state`: serialized `AgentState` snapshot (dict)
- `events`: accumulated `AgentEvent` dicts
- `error`: error info if a node adapter catches an exception

Using a dict (not a live `AgentState` dataclass) ensures safe checkpoint serialization without pickling non-serializable references.

## Router Functions

| Router | Input | Outputs |
|---|---|---|
| `route_after_parallelism_gate` | `agent_state.execution_strategy` | `parallel` / `collect_context` / `END` |
| `route_after_inspect` | `agent_state.blocked` | `select_tools` / `END` |
| `route_after_execute_tools` | `agent_state.awaiting_approval` | `propose_verified_patch` / `END` |
| `route_after_patch` | `agent_state.awaiting_approval` | `subdirectory_hint` / `END` |
| `route_after_guard` | `agent_state.blocked` | target / `END` |

## File Map

| File | Purpose |
|---|---|
| `backend/src/solo_agent/settings.py` | `workflow_engine`, `workflow_checkpointer`, `workflow_checkpoint_path` fields |
| `backend/src/solo_agent/workflow/graph_state.py` | `SoloGraphState`, serialization/deserialization helpers, reducers |
| `backend/src/solo_agent/workflow/graph_nodes.py` | Factory functions creating node adapters for each stage |
| `backend/src/solo_agent/workflow/graphs.py` | `build_text_provider_graph()` — full StateGraph construction |
| `backend/src/solo_agent/workflow/checkpoints.py` | `create_checkpointer(settings)` — memory/sqlite/none factory |
| `backend/src/solo_agent/workflow/runtime.py` | LangGraph strategy with feature-flag dispatch |
| `backend/tests/test_graph_state.py` | Unit tests for state serialization |
| `backend/tests/test_graph_nodes.py` | Unit tests for node adapters |
| `backend/tests/test_workflow_checkpoints.py` | Unit tests for checkpointer factory |
| `backend/tests/test_langgraph_graph_shape.py` | Unit tests for graph compilation and routing |
| `backend/tests/test_langgraph_workflow_runtime.py` | Integration tests for graph streaming |

## How to Enable

### Environment Variables

| Variable | Values | Default |
|---|---|---|
| `SOLO_AGENT_WORKFLOW_ENGINE` | `legacy`, `langgraph` | `legacy` |
| `SOLO_AGENT_WORKFLOW_CHECKPOINTER` | `memory`, `sqlite`, `none` | `memory` |
| `SOLO_AGENT_WORKFLOW_CHECKPOINT_PATH` | any file path | `.solo-agent/checkpoints/solo_agent_graph.sqlite3` |

### In-Memory (for testing)

```bash
SOLO_AGENT_WORKFLOW_ENGINE=langgraph SOLO_AGENT_WORKFLOW_CHECKPOINTER=memory uv run --extra dev python -m solo_agent.web.app
```

### SQLite (for persistent checkpoints)

```bash
SOLO_AGENT_WORKFLOW_ENGINE=langgraph SOLO_AGENT_WORKFLOW_CHECKPOINTER=sqlite uv run --extra dev python -m solo_agent.web.app
```

### No Checkpointing

```bash
SOLO_AGENT_WORKFLOW_ENGINE=langgraph SOLO_AGENT_WORKFLOW_CHECKPOINTER=none uv run --extra dev python -m solo_agent.web.app
```

### Fallback to Legacy

```bash
# Default behavior — no env vars needed
uv run --extra dev python -m solo_agent.web.app
```

## Current Limitations

- **Parallel dispatch is a placeholder**: The `parallel_dispatch_placeholder` node emits a skip event and falls back to `serial`. Real parallel scheduling (read-only subagents with worktree isolation) is a follow-up.
- **No LangGraph interrupt() integration**: Verified patch approval still uses the existing `awaiting_approval` flag + `if/return`. Converting to LangGraph's `interrupt()` is deferred.
- **Lead-agent strategy unchanged**: The lead-agent path (tool-calling provider with subagents) already uses LangGraph ReAct internally. It is not migrated to the new StateGraph.
- **Plan mode unchanged**: `_plan_mode_path` has its own stage logic outside the graph.
- **SQLite checkpointer requires optional dependency**: Install with `uv add langgraph-checkpoint-sqlite`.

## Follow-Up Plan

1. Replace `parallel_dispatch_placeholder` with read-only parallel scheduler
2. Add task type registry
3. Convert verified patch approval to LangGraph `interrupt()`
4. Add spec review and quality review nodes
5. Add worktree isolation for write-capable parallel tasks
6. Add LangGraph checkpoint replay API in Web
