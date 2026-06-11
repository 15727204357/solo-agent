# Solo Agent Coding Workflow

Solo Agent's team mode is a lightweight sandbox-aware coding workflow:

```text
team_plan -> team_develop -> team_test -> team_supervisor
```

## Runtime Shape

- `team_plan` turns the lead plan/task list into at most two developer assignments.
- Code tasks first build a lightweight code map and impact summary so team mode
  can pass affected files, symbols, related tests, and suggested verification
  commands into the developer/tester loop.
- `team_develop` runs a coding developer loop in the isolated command workspace when the provider supports tool calling.
- The developer loop is limited to `read_file`, `search_code`, `prepare_edit`, `preview_patch`, `apply_text_edit`, `run_pytest`, `run_ruff_check`, and `git_diff`.
- If the provider cannot call tools, team mode falls back to the existing JSON patch request path and applies that patch only inside the command workspace.
- `team_test` reviews the sandbox diff plus pytest/ruff evidence and can send structured feedback back to the developer for one more fix pass.
- `team_supervisor` reconstructs a verified patch proposal from the final sandbox files and leaves the main workspace unchanged until approval.

## Sandbox Boundary

The current implementation is lightweight local isolation, not Docker isolation.
For `sandbox_mode=isolated`, Solo Agent copies the workspace into
`.solo-agent/sandboxes/<session>/<run>/workspace` and runs developer edits and
allowed commands there. Command execution stays bounded by the local allowlist:
tests, lint/build checks, and read-only git inspection.

Because sandbox copies do not include `.git`, team-mode `git_diff` compares the
main workspace baseline against the sandbox workspace directly. This gives the
developer, tester, and supervisor a stable diff without writing to the main
checkout.

## Approval Boundary

The supervisor does not trust a model-authored patch as the merge artifact. It
rebuilds the patch request from the final sandbox file contents, then uses the
normal verified editing service to produce a pending patch proposal. The main
workspace is modified only after the user approves that proposal.

## Recovery Boundary

Long-running team work records graph snapshots plus sandbox artifacts:
sandbox root, sandbox diff, tool ledger, pytest/ruff output, developer summary,
code map summary, and impact analysis. User interrupt moves the run into
`paused` or `awaiting_feedback` instead of throwing the work away. Resume can
continue from `team_develop`, `team_test`, or `team_supervisor` with human
feedback and recovery hints.
