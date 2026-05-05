---
name: debug-test-failure
description: Workflow for diagnosing and fixing failing pytest or ruff checks.
category: workflow
triggers: [test failure, pytest failed, ruff failed, traceback, regression]
red_flags: [fix without reading failure, change unrelated code, silence test]
required_tools: [run_pytest, run_ruff_check, search_text, read_file, prepare_edit, preview_patch, apply_text_edit]
---

# Debug Test Failure

## Procedure

1. Reproduce or inspect the failing command output.
2. Identify the smallest failing unit.
3. Read the implementation and test around the failure.
4. Apply the smallest hash-anchored fix.
5. Rerun the failing check.

## Red Flags

- Deleting assertions instead of fixing behavior.
- Changing unrelated code to make the test pass.
- Ignoring a failing check at the end.

## Verification

- The original failure no longer appears.
- No new failure is introduced.
