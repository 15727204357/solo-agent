---
name: code-review
description: Local code review workflow focused on bugs, regressions, security, and missing tests as user-message SOP.
category: workflow
triggers: [review, code review, inspect changes, risks, regression]
red_flags: [style-only review, no line references, no tests considered]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest, run_ruff_check]
---

# Code Review

## When to Use

Use this skill when the user asks for a review, risk assessment, regression check, PR-style inspection, or evaluation of changed code.

This skill is Hermes-style user-message background: it guides review posture for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a specific review scope from the user.

## Iron Law

FINDINGS FIRST, RISKS OVER STYLE.

Prioritize correctness, regressions, security, and missing tests. Do not bury actionable findings under summaries or praise.

## Red Flags

- Style-only review.
- No concrete file or line references.
- No tests or verification considered.
- Suggesting broad rewrites instead of identifying review findings.
- Treating unclear behavior as acceptable without noting risk.

## Pressure Scenarios

- The diff is large and time is limited. Review the highest-risk paths first and state residual risk.
- The code looks clean but lacks tests for changed behavior. Report missing verification as a finding when it creates real risk.
- The user asks "LGTM?" Check for behavioral regressions before agreeing.

## Counterexamples

- If the user asks for implementation, do not stay in review mode; make the requested change instead.
- If no findings are found, say so clearly and list residual testing gaps rather than inventing issues.
- Pure formatting diffs may warrant a lighter review focused on generated or accidental behavioral changes.

## Rationalization Traps

- "This is probably fine" is not a review finding or verification.
- "I should mention style because I found no bugs" can distract from the risk assessment.
- "The author likely tested it" is not evidence unless test results are visible.
- "A big diff needs a big rewrite suggestion" is not useful unless tied to a concrete defect.

## Tool Protocol

1. Correctness bugs.
2. Behavioral regressions.
3. Security or secret exposure risks.
4. Missing tests or weak verification.

- Inspect the relevant files or diff.
- Report findings first, ordered by severity.
- Include concrete file and line references when available.
- If no findings exist, say so and mention residual risk.
- Use tests or lint when they materially improve confidence and fit the review scope.
- Keep summaries brief and secondary to findings.

## Stop Conditions

- Stop if the review scope is unclear enough that findings would target the wrong change.
- Stop if required diff or file context is unavailable.
- Stop before editing files unless the user explicitly changes the task from review to implementation.
- Stop if verification cannot be run and the remaining risk is material; report the limitation.

## Verification

- Findings are actionable.
- Summary does not bury risks.
- Each finding has a tight file and line reference when possible.
- No-finding reviews explicitly state residual risk or testing gaps.
