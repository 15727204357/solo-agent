---
name: python-backend-change
description: Workflow for Python backend changes using context reads, hash-anchored edits, pytest, and ruff as user-message SOP.
category: workflow
triggers: [python backend, fastapi, sqlalchemy, pytest, api route, repository, agent graph]
red_flags: [skip pytest, broad refactor, hidden behavior change]
required_tools: [workspace_snapshot, search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest, run_ruff_check]
metadata: {"hermes": {"recipes": [{"id": "inspect", "file": "references/recipes/inspect.yaml"}, {"id": "focused-test", "file": "references/recipes/focused-test.yaml"}, {"id": "verify", "file": "references/recipes/verify.yaml"}]}}
---

# Python Backend Change

## When to Use

Use this skill when implementing, fixing, or refactoring Python backend behavior such as API routes, services, repositories, persistence, or agent graph code.

This skill is Hermes-style user-message background: it guides backend change workflow for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or the current user request.

## Iron Law

BACKEND CHANGES NEED LOCAL CONTEXT, A SMALL EDIT, AND EXECUTED VERIFICATION.

Do not change backend behavior from memory. Read the relevant code, make the smallest safe change, and run focused pytest plus ruff when feasible.

## Red Flags

- Skipping pytest after changing behavior.
- Broad refactors around the requested fix.
- Hidden API, schema, or persistence behavior changes.
- Editing generated or migration-sensitive files without confirming the workflow.
- Assuming async, transaction, or dependency injection behavior without reading it.

## Pressure Scenarios

- The bug spans route, service, and repository layers. Trace the narrow call path and avoid redesigning the stack.
- A failing backend test suggests a fixture or database issue. Read the fixture and persistence boundary before editing production code.
- The requested fix touches an agent graph or workflow. Verify state transitions and edge behavior, not just the happy path.

## Counterexamples

- Documentation-only backend notes do not need pytest unless executable examples changed.
- A review-only backend task should report findings instead of editing files.
- A user-approved spike may prioritize exploration, but production changes still need clear verification before finalizing.

## Rationalization Traps

- "This is just Python" ignores framework, database, and async lifecycle behavior.
- "The route works if the service works" misses serialization, dependency, and error handling boundaries.
- "Ruff passed, so behavior is fine" confuses style checks with tests.
- "The test setup is annoying" is not a reason to skip focused verification.

## Tool Protocol

1. Inspect project structure with `workspace_snapshot` when the relevant layout is not known.
2. Locate relevant files with `search_text`.
3. Read only the files needed to understand the behavior boundary.
4. If editing, follow `hash-anchored-editing`.
5. Prefer adding or updating a focused test before production code when behavior changes.
6. Run focused pytest when possible, then ruff.
7. Broaden tests only when the change crosses module or integration boundaries.

## Stop Conditions

- Stop if the behavior contract is ambiguous and multiple API outcomes are plausible.
- Stop if the next edit would require a schema, migration, or data compatibility decision not requested.
- Stop if tests cannot run and the change has material backend risk.
- Stop if the fix would touch unrelated layers or refactor beyond the requested outcome.

## Verification

- Tests cover the intended behavior.
- `run_pytest` passes or failures are explained.
- `run_ruff_check` passes or failures are explained.
- API, persistence, and agent workflow side effects are considered when relevant.
- Any unverified backend risk is reported clearly.
