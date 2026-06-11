# Migration: LangGraph StateGraph As Sole Runtime

Solo Agent routes all main workflow behavior through a single LangGraph
`StateGraph`.

## Current Runtime

- `workflow_engine` is no longer a product-level runtime split.
- The legacy manual text-provider stage loop is removed from active routing.
- The independent lead-agent runtime is folded into graph stages.
- Default execution is serial: `collect_context -> inspect -> select_tools`.
- Multi-agent execution is the lightweight team path, enabled only by
  `run_mode=plan` plus `subagent_enabled=true`.
- The old graph-level parallel fan-out helpers remain as legacy diagnostics,
  but `parallelism_gate` is not the main graph decision point.

## Preserved Layers

- Planning and plan mode task state.
- Memory prelude/postlude.
- Context budget checks.
- Safety inspectors.
- Tool registry and isolated command workspace.
- Verified editing and approval.
- SQLite-backed persistence and snapshots.

## Team Workflow

The team branch is:

```text
team_plan -> team_develop -> team_test -> team_supervisor
```

It keeps the implementation small:

- fixed roles: planner, developer, tester, supervisor;
- developer pool capped at two assignments;
- developer edits apply only inside the command sandbox;
- tester feedback loops back to developer at most twice;
- supervisor emits a normal pending verified patch proposal for approval.

## Validation Checklist

- `backend/src` has no active legacy workflow runtime branch.
- `build_main_workflow_graph()` is the graph builder used by runtime tests.
- `route_after_task_state()` enters team mode only for plan plus subagent.
- Team mode does not route through `parallelism_gate`.
- Developer sandbox changes leave the main workspace unchanged.
- All focused workflow, verified editing, and web API tests pass.

## Rollback

If a regression appears, the previous runtime remains recoverable from Git
history; do not reintroduce runtime branching unless a targeted rollback is
explicitly chosen.
