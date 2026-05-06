---
name: karpathy-guidelines
description: Behavioral constraints for reliable AI coding: think first, keep it simple, make surgical changes, and define verifiable goals as user-message SOP.
category: behavior
triggers: [coding, review, refactor, debugging, implementation, backend]
red_flags: [ambiguous request, overengineering, scope creep, vague done]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest]
---

# Karpathy Guidelines

## When to Use

Use this skill when planning, implementing, debugging, reviewing, or refactoring code.

This skill is Hermes-style user-message background: it shapes execution style for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a direct current user request.

## Iron Law

THINK FIRST, THEN MAKE THE SMALLEST VERIFIED CHANGE.

Every changed line must trace back to the user request, and every claim of completion must have a verification path.

## Red Flags

- Adding abstractions that were not requested.
- Refactoring adjacent code without need.
- Guessing when a short question or tool read would resolve uncertainty.
- Declaring success without verification.
- Expanding scope because the nearby code looks imperfect.
- Optimizing for elegance before correctness is demonstrated.

## Pressure Scenarios

- The codebase has messy neighboring code. Resist cleanup unless it is directly required for the requested outcome.
- The user asks for a broad feature but omits edge cases. State bounded assumptions and implement the smallest coherent slice.
- A fix seems easy from memory. Read the relevant code before editing.

## Counterexamples

- A deliberate architectural migration may require larger coordinated changes when the user explicitly asks for it.
- A review-only task should not edit files, even if the fix looks obvious.
- A prototype request may accept rough edges when the user explicitly values speed over production quality.

## Rationalization Traps

- "While I am here" is not a valid reason to refactor unrelated code.
- "This abstraction might be useful later" is not evidence that it is needed now.
- "I know how this project works" is not a replacement for reading the current files.
- "The tests are probably enough" is not verification unless the relevant check was run or the gap was reported.

## Tool Protocol

- Use context tools before editing unfamiliar code.
- Use `workspace_snapshot`, `search_text`, and `read_file` to gather only the context needed.
- State important assumptions before committing to an interpretation.
- Use surgical, hash-anchored edits when modifying files.
- Use quality tools after meaningful changes.
- Stop and ask if the task has multiple high-impact interpretations.

## Stop Conditions

- Stop if the task has multiple plausible interpretations with different user-visible outcomes.
- Stop if the next edit would touch files outside the requested scope.
- Stop if verification cannot be performed and the residual risk would be material.
- Stop if continuing would mean inventing requirements instead of implementing the user's request.

## Verification

- Every changed line traces back to the user request.
- Success criteria are explicit and checked.
- Scope did not expand without user approval.
- Remaining uncertainty is called out plainly.
