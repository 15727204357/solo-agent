---
name: python-backend-change
description: Workflow for Python backend changes using context reads, hash-anchored edits, pytest, and ruff.
category: workflow
triggers: [python backend, fastapi, sqlalchemy, pytest, api route, repository, agent graph]
red_flags: [skip pytest, broad refactor, hidden behavior change]
required_tools: [workspace_snapshot, search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest, run_ruff_check]
---

# Python Backend Change

## Procedure

1. Inspect project structure with `workspace_snapshot`.
2. Locate relevant files with `search_text`.
3. Read only the files needed.
4. If editing, follow `hash-anchored-editing`.
5. Run focused pytest when possible, then ruff.

## Verification

- Tests cover the intended behavior.
- `run_pytest` passes or failures are explained.
- `run_ruff_check` passes or failures are explained.
