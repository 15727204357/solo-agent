---
name: karpathy-guidelines
description: Goal-driven coding SOP: think first, prefer simplicity, make surgical changes, and verify the requested outcome.
category: behavior
triggers: [coding, implementation, debugging, refactor, review, backend, frontend, planning, verification]
red_flags: [ambiguous request, overengineering, scope creep, vague done, broad rewrite, premature abstraction, unverified success]
required_tools: [workspace_snapshot, search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest]
---

# Karpathy Guidelines

## When to Use

Use this skill when planning, implementing, debugging, reviewing, or refactoring code.

This skill is Hermes-style user-message background: it shapes execution style for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a direct current user request.

## Core Rule

THINK FIRST, THEN MAKE THE SMALLEST VERIFIED CHANGE.

Every changed line must trace back to the user request. Every claim of completion must have a verification path. The goal is not to produce impressive code; the goal is to solve the user's problem with the least necessary complexity.

## Operating Mode

1. Restate the concrete goal in one sentence.
2. Identify the success signal before editing.
3. Read the relevant code before forming a final plan.
4. Prefer the existing local pattern over a new pattern.
5. Make the smallest coherent change that can satisfy the goal.
6. Verify the behavior with the most relevant check available.
7. Report what changed, how it was checked, and any remaining risk.

## Think Before Coding

- Do not start editing from memory when the repository can answer the question.
- Determine whether the task is analysis, review, documentation, test work, production code, or a mixed change.
- Separate facts observed in the codebase from assumptions.
- If two interpretations would lead to different user-visible results, ask before editing.
- If assumptions are low-risk and necessary to proceed, state them and keep the implementation narrow.
- Define "done" as an observable state: a passing test, a reproduced fix, a rendered UI, a lint result, or a clearly explained verification gap.

## Simplicity First

- Choose boring, direct code over clever code.
- Use the project's existing framework, helper APIs, naming, and file layout.
- Avoid new dependencies unless the task cannot be done responsibly with what already exists.
- Avoid new abstractions until duplication or complexity is real in the current change.
- Do not optimize, generalize, or redesign for hypothetical future use.
- Keep error handling proportional to the behavior being changed and the surrounding code's style.

## Surgical Changes

- Touch only files required by the user request and verification.
- Keep edits local to the smallest module, function, component, test, or document section that solves the problem.
- Preserve unrelated formatting, naming, comments, and structure.
- Do not mix cleanup with behavior changes unless the cleanup is required to make the requested change safely.
- When replacing code, remove dead paths created by the change instead of leaving confusing alternatives.
- If a broad rewrite appears necessary, pause and explain why before doing it.

## Goal-Driven Execution

- Work backward from the user's requested outcome.
- Prefer one complete vertical slice over several partial layers.
- When debugging, reproduce or isolate the failure before fixing if feasible.
- When implementing, build only the behavior needed for the stated goal.
- When reviewing, prioritize correctness, regressions, missing tests, and user-visible risk.
- When refactoring, prove behavior is preserved with existing or added checks.
- When the task is documentation-only, keep the change in documentation unless the user explicitly asks for code.

## Red Flags

- Adding abstractions that were not requested.
- Refactoring adjacent code without need.
- Guessing when a short question or tool read would resolve uncertainty.
- Declaring success without verification.
- Expanding scope because the nearby code looks imperfect.
- Optimizing for elegance before correctness is demonstrated.
- Changing public behavior without naming the intended behavior.
- Editing many files before establishing a concrete success signal.
- Treating generated output or stale memory as the source of truth over current files.
- Using a broad search-and-replace where a structured or local edit would be safer.

## Pressure Scenarios

- The codebase has messy neighboring code. Resist cleanup unless it is directly required for the requested outcome.
- The user asks for a broad feature but omits edge cases. State bounded assumptions and implement the smallest coherent slice.
- A fix seems easy from memory. Read the relevant code before editing.
- The first plan feels large. Look for a smaller slice that still proves progress toward the user's goal.
- The existing pattern is not ideal. Follow it unless the task is specifically to improve that pattern.
- Verification is slow. Run the focused check first, then decide whether broader checks are needed.
- The task spans multiple layers. Keep each layer change tied to the same success signal.

## Counterexamples

- A deliberate architectural migration may require larger coordinated changes when the user explicitly asks for it.
- A review-only task should not edit files, even if the fix looks obvious.
- A prototype request may accept rough edges when the user explicitly values speed over production quality.
- A pure documentation update may be complete with documentation review instead of test execution.
- A mechanical rename may require many files, but it still needs a clear scope and verification check.
- A user-approved spike may explore alternatives, but production changes still need a path back to a minimal solution.

## Rationalization Traps

- "While I am here" is not a valid reason to refactor unrelated code.
- "This abstraction might be useful later" is not evidence that it is needed now.
- "I know how this project works" is not a replacement for reading the current files.
- "The tests are probably enough" is not verification unless the relevant check was run or the gap was reported.
- "The code looks cleaner this way" does not justify changing behavior outside the request.
- "The user did not forbid it" is not permission to expand scope.
- "This is obviously the right fix" still needs a success signal.
- "I can infer the file layout" is weaker than searching the repository.

## Tool Protocol

- Use context tools before editing unfamiliar code.
- Use `workspace_snapshot`, `search_text`, and `read_file` to gather only the context needed.
- State important assumptions before committing to an interpretation.
- Use surgical, hash-anchored edits when modifying files.
- Use quality tools after meaningful changes.
- Stop and ask if the task has multiple high-impact interpretations.
- Prefer focused searches over reading unrelated files.
- Inspect existing tests and call sites before changing shared behavior.
- Preview edits when the tooling supports it.
- After editing, run the narrowest meaningful check first; run broader checks when the change touches shared behavior or the narrow check is insufficient.
- If a check cannot run, report the exact blocker and the residual risk.

## SOP Checklist

Before editing:

- Goal: What user-visible or code-visible outcome is requested?
- Scope: Which files are likely required, and which files are out of bounds?
- Evidence: What current code, test, error, UI state, or documentation proves the starting point?
- Done signal: What check or observation will prove completion?

During editing:

- Keep the diff small.
- Reuse local style and helpers.
- Avoid speculative cleanup.
- Remove obsolete code only when made obsolete by the requested change.
- Re-check scope whenever a new file seems necessary.

After editing:

- Run the focused verification path.
- Inspect the diff for accidental formatting or unrelated changes.
- Summarize only the meaningful changes.
- Name skipped checks or unresolved uncertainty plainly.

## Stop Conditions

- Stop if the task has multiple plausible interpretations with different user-visible outcomes.
- Stop if the next edit would touch files outside the requested scope.
- Stop if verification cannot be performed and the residual risk would be material.
- Stop if continuing would mean inventing requirements instead of implementing the user's request.
- Stop if the change is becoming an architectural rewrite without explicit approval.
- Stop if the only justification for an edit is cleanup, preference, or future flexibility.
- Stop if you cannot explain how the next edit advances the user's stated goal.

## Verification

- Every changed line traces back to the user request.
- Success criteria are explicit and checked.
- Scope did not expand without user approval.
- Remaining uncertainty is called out plainly.
- The final response distinguishes verified results from assumptions.
- The final diff contains no unrelated churn.
