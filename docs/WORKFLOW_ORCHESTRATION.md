# Solo Agent Workflow Orchestration

## Architecture

Solo Agent uses one LangGraph `StateGraph` as the main workflow runtime. The
graph owns memory/skill loading, planning, serial execution, lightweight team
subagents, verified editing, error recovery, response, and persistence.

## Graph Topology

`build_main_workflow_graph()` in `backend/src/solo_agent/workflow/graphs.py`
builds the active graph:

```text
START
  -> receive_user_turn
  -> memory prelude
  -> skill_context
  -> context_guard_before_plan
  -> load_task_state
  -> plan
  -> task_state
     -> [plan + subagent_enabled] team_plan
          -> team_develop
          -> team_test
             -> [needs_fix and loop < 2] team_develop
             -> team_supervisor
                -> [patch proposal ready] END awaiting approval
                -> [no proposal] collect_context
     -> [default] collect_context
  -> inspect -> select_tools -> execute_tools
  -> spec_compliance_review -> propose_verified_patch
  -> subdirectory_hint -> context_guard_before_respond -> respond
  -> skill_evolution -> memory postlude -> persist_snapshot -> END
```

## Team Subagent Path

The team path is intentionally small and stable:

- It only runs when `run_mode=plan` and `subagent_enabled=true`.
- `team_plan` converts the lead plan/task state into a task directory and at
  most two developer assignments.
- `collect_context` builds a lightweight code map and impact analysis for code
  tasks before the team path reaches developer/tester stages.
- `team_develop` prefers a restricted coding tool loop inside the command
  workspace. The developer can read/search code, prepare and preview edits,
  apply text edits in the sandbox, run pytest/ruff, and inspect sandbox diff.
  Providers without tool calling still use the JSON patch fallback, applied
  only inside the command workspace.
- `team_test` runs targeted verification commands from the team plan in the
  command workspace, falls back to impact-analysis verification suggestions,
  and allows at most two developer feedback rounds.
- `team_supervisor` turns the final patch request into the normal pending
  verified patch proposal for the main workspace using sandbox diff and ledger
  evidence rather than trusting model-authored final JSON.

Developer subagents never write the main workspace directly. Main workspace
changes still require the existing verified editing approval flow.

## Legacy Parallelism Gate

`workflow.parallelism` and the `parallelism_gate`/`parallel_dispatch` helpers
remain available for diagnostics and tests, but they are no longer the main
subagent route. The product-facing multi-agent story is the fixed team workflow.

## Error Recovery

Graph node exceptions route through:

```text
classify_error -> recovery_route
  -> [recoverable] recovery_action -> repetition_guard
  -> [policy_violation] blocked_response
  -> [environment_error] environment_error_response
  -> [architecture_failure] architecture_failure_response
```

## Checkpoint And Replay

SQLite remains the default checkpointer. The web API exposes checkpoint,
interrupt, resume, artifacts, and replay operations. Resume can target
`team_develop`, `team_test`, or `team_supervisor` with human feedback and
recovery hints. Sandbox artifacts are retained for approval, paused, and
awaiting-feedback states so the next run can continue from the same evidence
when the sandbox still exists.
