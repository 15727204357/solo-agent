---
name: iron-law
description: Superpowers-style TDD SOP for production code changes: no production code without a failing test first.
category: behavior
triggers: [code change, bug fix, refactor, production code, implementation, test first, TDD, regression, behavior change]
red_flags: [simple change, skip tests, just patch it, no time, trust me, test after, obvious fix, keep untested code]
required_tools: [search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest]
---

# Iron Law

## When to Use

Use this skill when the current user task asks for production code to be written, edited, fixed, or refactored.

This skill is Hermes-style user-message background: it is a local SOP for the task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a more specific current user request.

## Iron Law

NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.

Before changing production code, locate or create a failing test that proves the change is needed. The failing test is the permission slip for the production edit.

If production code is written before the failing test, delete that production code and start over from the test. Do not keep the untested implementation as a reference, commented code, or hidden fallback.

If no failing test exists, stop and surface `iron_law_blocked`. A request to "skip tests" does not unlock production edits; it only confirms the request is not ready for a production-code change.

## Non-Negotiable Loop

1. Understand the requested behavior change.
2. Find the smallest existing test location that should express the behavior.
3. Write or update exactly the focused test needed to fail for the current defect or missing behavior.
4. Run the focused test and confirm it fails for the expected reason.
5. Only then edit production code.
6. Make the smallest production change that can pass the test.
7. Run the focused test again and confirm it passes.
8. Run broader relevant checks when the change touches shared behavior, public APIs, integration points, or fragile code.
9. Report the red-green evidence in the final response.

## What Counts as Production Code

- Application logic, library logic, CLI behavior, services, handlers, models, components, templates, migrations, scripts used by users, and configuration that changes runtime behavior.
- Refactors of production code, even when intended to preserve behavior.
- Bug fixes, feature work, performance changes, and compatibility changes.
- Build or packaging changes when they affect shipped behavior.

## What Counts as a Failing Test

- A test that fails before the production edit and passes after it.
- A regression test that reproduces the reported bug.
- A characterization test for a refactor, when the goal is behavior preservation.
- A narrow integration or end-to-end check when unit-level coverage cannot observe the behavior.
- A documented focused command and failure output when the repository already has the right test but it is currently failing for the relevant reason.

The failure must be meaningful. A syntax error, broken fixture, wrong assertion, or test that fails for an unrelated reason does not satisfy the law.

## Allowed Exceptions

Exceptions must be explicit and narrow. They do not silently weaken the default rule.

- Documentation-only edits do not require a failing test first, unless the docs encode executable behavior.
- Test-only cleanup does not require a new failing test, but the affected tests must still be run when feasible.
- Review-only, planning-only, or exploratory analysis may inspect production code without adding tests because no production code is being changed.
- Pure formatting that provably does not change behavior may use formatter verification instead of a failing test.
- Emergency production-code overrides are not handled by this skill. Escalate to the human partner and do not apply the production edit in the graph run.

## Red Flags

- "This is simple, no test needed."
- "We can test after the implementation."
- "Just patch it quickly."
- "Keep the untested code as reference."
- A bug report with no reproduction path.
- A refactor that claims behavior will not change but has no safety check.
- A production diff appears before any test diff.
- A test is added after the implementation and never observed failing.
- The test fails for the wrong reason.
- The fix is validated only by reasoning.
- The implementation is broad because writing a narrow test feels awkward.
- The test is skipped, marked xfail, or loosened to make the suite pass.

## Pressure Scenarios

- The user asks for a "tiny one-line fix" in production code. Still require a focused failing test or an explicit user-approved skip.
- A deadline makes testing feel expensive. Prefer the smallest failing test over a broad implementation change.
- Existing tests are hard to understand. Read just enough test structure to add a focused regression test before editing production code.
- A failing test is hard to write. Narrow the behavior, add a characterization seam through existing public APIs, or stop and explain the blocker.
- The bug is reproduced manually. Convert the reproduction into the smallest automated test before changing production code when feasible.
- The existing suite is slow. Run a targeted test first, then broaden verification after the fix.
- The change is a refactor. Capture the behavior with existing or added tests before moving code.
- The user asks to "just make it work." The law still applies unless they explicitly approve skipping it.

## Counterexamples

- Documentation-only edits do not require a failing test first, unless the docs encode executable behavior.
- Test-only cleanup does not require a new failing test, but it still needs verification that the suite remains meaningful.
- Exploratory analysis or code review can inspect production code without creating tests because no production code is being changed.
- Generated snapshots or golden files may be updated after the failing test demonstrates the intentional behavior change.
- A build-system-only formatting change can be verified by the relevant formatter or build check if it cannot affect runtime behavior.
- A deleted unused file may not need a new failing test if usage search and existing suite provide the safety signal.

## Rationalization Traps

- "The change is obviously correct" is not a substitute for a failing test.
- "Existing tests probably cover it" is not enough unless one fails for the defect before the fix.
- "I will add tests later" weakens the signal that the change was necessary.
- "The user seems in a hurry" does not silently waive the gate.
- "I already know the fix" is not permission to write it first.
- "I need to see the implementation to know what to test" usually means the behavior is not understood yet.
- "The test would be too small" is not a problem; small focused tests are preferred.
- "The code is legacy" increases the need for a failing characterization test.
- "The test is flaky, but it failed once" is not enough unless the failure is reproducible and relevant.

## Recovery Rule

If production code was changed before a failing test:

1. Stop editing production code immediately.
2. Remove the untested production change.
3. Return to the last state where production code did not include the speculative fix.
4. Add or locate the failing test.
5. Confirm the red state.
6. Re-implement only what is needed to make that test pass.

Do not preserve the premature implementation in comments, scratch files, alternate branches inside the code, or hidden toggles.

## Tool Protocol

- Use `search_text` and `read_file` to locate existing tests.
- Add or update the smallest test that should fail before the production change.
- Use hash-anchored edit tools for any file changes.
- Run the focused test before the production change when feasible and record the failure.
- Make the smallest production change needed to pass the test.
- Run `run_pytest` after the change, focused first and broader when appropriate.
- Inspect the failure output enough to confirm the failure is the expected one.
- Keep the first failing test narrow; add broader tests only when risk requires them.
- Do not weaken assertions, skip tests, or change fixtures merely to make the suite green.
- When sandbox or tool limits block test execution, report the exact blocked command and do not pretend the red state was observed.
- A user request to skip tests does not waive the red step for production code.

## SOP Checklist

Before production code:

- Behavior: What exact behavior must change or be preserved?
- Test location: Which existing test file or test style is closest?
- Red state: Which command will prove the test fails before the fix?
- Expected failure: What assertion or error should appear?

During production code:

- Keep the implementation minimal.
- Do not broaden the fix beyond the failing test's behavior unless the user request requires it.
- Do not edit the test to match the implementation unless the test was wrong for a stated reason.
- Remove temporary diagnostics unless they are part of the requested behavior.

After production code:

- Re-run the focused test and confirm green.
- Run broader relevant tests when appropriate.
- Inspect the diff order mentally: the test explains the production change.
- Report red, green, and any broader checks.

## Stop Conditions

- Stop before editing production code if no failing test exists and the user has not explicitly accepted skipping the gate.
- Stop if the failing test cannot be reproduced or does not test the requested behavior.
- Stop if adding the test requires a broad design decision not covered by the user request.
- Stop if tool or sandbox constraints prevent reliable verification; report the blocker instead of guessing.
- Stop if the current diff contains production code created before the failing test and remove it before proceeding.
- Stop if the only available test would assert implementation details unrelated to the requested behavior.
- Stop if making the test pass requires changing the requested behavior or hiding the failure.

## Verification

- A relevant test fails before the production change.
- The same test passes after the production change.
- Any remaining test failures are explained with scope and confidence.
- The final response identifies the focused failing test and the passing command.
- If the user asked to skip tests, the final response labels the production edit as blocked rather than partially verified.
- No production code remains that was not justified by the red-green loop or an explicit exception.
