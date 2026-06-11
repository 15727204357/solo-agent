# Workflow Parallelism Gate

The old parallelism gate is now a legacy diagnostic helper, not the primary
subagent orchestration path.

## Current Product Rule

Multi-agent execution is enabled only when both conditions are true:

- `run_mode=plan`
- `subagent_enabled=true`

When enabled, the graph follows the fixed team workflow:

```text
team_plan -> team_develop -> team_test -> team_supervisor
```

The previous `parallelism_gate -> parallel_dispatch -> wait_subagents` path is
kept for internal experiments and regression tests, but the main graph does not
route through it.

## Legacy Gate Semantics

The helper still evaluates whether structured task metadata is safe to split:

- problem-domain independence;
- context independence;
- write-set independence;
- verification independence.

This remains useful as a diagnostic signal, but the resume-project demo favors
the stable team workflow over arbitrary model-selected subagent topology.

## Team Workflow Events

The active team path emits:

- `team_plan_started`
- `team_plan_completed`
- `team_developer_started`
- `team_developer_completed`
- `team_tester_started`
- `team_tester_completed`
- `team_supervisor_started`
- `team_supervisor_completed`
- `patch_proposed`
- `patch_approval_required`

## Scope

Developer subagents can produce real code changes, but only inside the isolated
command workspace. The supervisor converts the final sandbox result into the
existing verified editing patch proposal for the main workspace, so applying
changes still requires explicit approval.
