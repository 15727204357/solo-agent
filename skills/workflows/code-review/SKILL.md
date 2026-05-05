---
name: code-review
description: Local code review workflow focused on bugs, regressions, security, and missing tests.
category: workflow
triggers: [review, code review, inspect changes, risks, regression]
red_flags: [style-only review, no line references, no tests considered]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest, run_ruff_check]
---

# Code Review

## Review Priorities

1. Correctness bugs.
2. Behavioral regressions.
3. Security or secret exposure risks.
4. Missing tests or weak verification.

## Procedure

- Inspect the relevant files or diff.
- Report findings first, ordered by severity.
- Include concrete file and line references when available.
- If no findings exist, say so and mention residual risk.

## Verification

- Findings are actionable.
- Summary does not bury risks.
