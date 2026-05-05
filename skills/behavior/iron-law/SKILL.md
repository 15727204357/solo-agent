---
name: iron-law
description: Enforce test-first discipline for production code changes. Use when writing, editing, fixing, or refactoring code.
category: behavior
triggers: [code change, bug fix, refactor, production code, test first, TDD]
red_flags: [simple change, skip tests, just patch it, no time, trust me]
required_tools: [search_text, read_file, prepare_edit, preview_patch, apply_text_edit, run_pytest]
---

# Iron Law

## The Iron Law

NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST.

If asked to change production code, first locate or create a failing test that proves the change is needed. If no failing test exists and the user did not explicitly ask to skip tests, stop and surface `iron_law_blocked` or `iron_law_warning`.

## Red Flags

- "This is simple, no test needed."
- "We can test after the implementation."
- "Just patch it quickly."
- "Keep the untested code as reference."

## Tool Protocol

- Use `search_text` and `read_file` to locate existing tests.
- Use hash-anchored edit tools for any file changes.
- Run `run_pytest` after the change.

## Verification

- The relevant test fails before the production change or the user explicitly accepts skipping that gate.
- The relevant test passes after the production change.
