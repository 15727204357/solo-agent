---
name: iron-law
description: Enforce test-first discipline for production code changes. Use as user-message SOP when writing, editing, fixing, or refactoring code.
category: behavior
triggers: [code change, bug fix, refactor, production code, test first, TDD]
red_flags: [simple change, skip tests, just patch it, no time, trust me]
required_tools: [search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest]
---

# Iron Law

## When to Use

Use this skill when the current user task asks for production code to be written, edited, fixed, or refactored.

This skill is Hermes-style user-message background: it is a local SOP for the task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a more specific current user request.

## Iron Law

NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.

Before changing production code, locate or create a failing test that proves the change is needed. If no failing test exists and the user did not explicitly ask to skip that gate, stop and surface `iron_law_blocked` or `iron_law_warning`.

## Red Flags

- "This is simple, no test needed."
- "We can test after the implementation."
- "Just patch it quickly."
- "Keep the untested code as reference."
- A bug report with no reproduction path.
- A refactor that claims behavior will not change but has no safety check.

## Pressure Scenarios

- The user asks for a "tiny one-line fix" in production code. Still require a focused failing test or an explicit user-approved skip.
- A deadline makes testing feel expensive. Prefer the smallest failing test over a broad implementation change.
- Existing tests are hard to understand. Read just enough test structure to add a focused regression test before editing production code.

## Counterexamples

- Documentation-only edits do not require a failing test first, unless the docs encode executable behavior.
- Test-only cleanup does not require a new failing test, but it still needs verification that the suite remains meaningful.
- Exploratory analysis or code review can inspect production code without creating tests because no production code is being changed.

## Rationalization Traps

- "The change is obviously correct" is not a substitute for a failing test.
- "Existing tests probably cover it" is not enough unless one fails for the defect before the fix.
- "I will add tests later" weakens the signal that the change was necessary.
- "The user seems in a hurry" does not silently waive the gate.

## Tool Protocol

- Use `search_text` and `read_file` to locate existing tests.
- Add or update the smallest test that should fail before the production change.
- Use hash-anchored edit tools for any file changes.
- Run the focused test before the production change when feasible and record the failure.
- Make the smallest production change needed to pass the test.
- Run `run_pytest` after the change, focused first and broader when appropriate.

## Stop Conditions

- Stop before editing production code if no failing test exists and the user has not explicitly accepted skipping the gate.
- Stop if the failing test cannot be reproduced or does not test the requested behavior.
- Stop if adding the test requires a broad design decision not covered by the user request.
- Stop if tool or sandbox constraints prevent reliable verification; report the blocker instead of guessing.

## Verification

- A relevant test fails before the production change, or the user explicitly accepted skipping that gate.
- The same test passes after the production change.
- Any remaining test failures are explained with scope and confidence.
