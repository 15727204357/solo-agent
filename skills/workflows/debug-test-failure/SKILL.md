---
name: debug-test-failure
description: Workflow for diagnosing and fixing failing pytest or ruff checks as user-message SOP.
category: workflow
triggers: [test failure, pytest failed, ruff failed, traceback, regression]
red_flags: [fix without reading failure, change unrelated code, silence test]
required_tools: [run_pytest, run_ruff_check, search_text, read_file, prepare_edit, preview_patch, apply_text_edit]
metadata: {"hermes": {"recipes": [{"id": "failure-triage", "file": "references/recipes/failure-triage.yaml"}]}}
---

# Debug Test Failure

## When to Use

Use this skill when pytest, ruff, or another local quality check fails and the user asks to diagnose or fix it.

This skill is Hermes-style user-message background: it guides the debugging workflow for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a narrower user instruction.

## Iron Law

REPRODUCE OR READ THE FAILURE BEFORE FIXING IT.

Never change code to fix a test failure until the failure mode, target behavior, and smallest relevant unit are understood.

## Red Flags

- Deleting assertions instead of fixing behavior.
- Changing unrelated code to make the test pass.
- Ignoring a failing check at the end.
- Fixing from memory without reading the traceback.
- Silencing lint or skipping tests without user approval.

## Pressure Scenarios

- Many tests fail at once. Start with the first root-cause failure, not every symptom.
- The failure looks like an environment problem. Separate environment blockers from code defects before editing.
- A lint fix is mechanical but broad. Preview the exact diff and avoid unrelated formatting churn.

## Counterexamples

- If the user only asks for an explanation of a failure, do not edit files.
- If the failure is caused by a missing dependency or sandbox limit, report the blocker before changing code.
- If the failing check is unrelated to the user's requested change, preserve scope and ask before broad fixes.

## Rationalization Traps

- "I have seen this traceback before" is not a substitute for reading this failure.
- "Changing the test is faster" is unsafe unless the test expectation is demonstrably wrong.
- "The final suite mostly passes" does not close the task if the original failure remains.
- "A broad refactor will clean this up" is not debugging unless it is the smallest safe fix.

## Tool Protocol

1. Reproduce or inspect the failing command output.
2. Identify the smallest failing unit.
3. Read the implementation and test around the failure.
4. Apply the smallest hash-anchored fix.
5. Rerun the failing check.
6. Run adjacent checks when the fix could affect nearby behavior.

## Stop Conditions

- Stop if the failure cannot be reproduced and no reliable output is available.
- Stop if the root cause points outside the current task scope.
- Stop if the next fix would remove coverage or weaken assertions without clear evidence.
- Stop if environment, dependency, or sandbox constraints prevent verification.

## Verification

- The original failure no longer appears.
- No new failure is introduced in the focused check.
- Any broader remaining failures are identified as related or unrelated.
- The fix is the smallest change that addresses the root cause.
