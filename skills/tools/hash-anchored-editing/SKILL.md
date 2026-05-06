---
name: hash-anchored-editing
description: Safe editing protocol requiring file hash validation, patch preview, and anchored writes as user-message SOP.
category: tools
triggers: [edit file, modify code, patch, write code, refactor]
red_flags: [line number only, stale context, direct write, no preview]
required_tools: [get_file_hash, prepare_edit, preview_patch, apply_text_edit]
---

# Hash-Anchored Editing

## When to Use

Use this skill whenever an existing file may be edited, especially in a shared or concurrent workspace.

This skill is Hermes-style user-message background: it defines a safer editing SOP. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or the current user task.

## Iron Law

Never write to an existing file unless the edit is anchored to the current file hash.

## Red Flags

- Editing from stale context.
- Replacing text that appears multiple times.
- Applying a patch without preview.
- Trusting line numbers after another worker may have edited the file.
- Writing a whole file when a small anchored change would do.

## Pressure Scenarios

- Another worker is editing nearby files. Re-read and re-hash before applying edits instead of assuming the file is unchanged.
- The intended replacement is repeated in the file. Use a unique anchor or narrow context before writing.
- A quick typo fix feels safe. Still validate the file state before changing it.

## Counterexamples

- Creating a brand-new file has no prior file hash, but this project may still forbid new files for the current task.
- Reading, searching, or reviewing files does not require a hash anchor because no write occurs.
- Generated formatter output may rewrite broad regions, but it should only run after an anchored intentional source edit.

## Rationalization Traps

- "I just read the file" is not enough if another actor could have changed it.
- "The patch is obvious" is not a reason to skip preview.
- "The text only appears once in my memory" is not a substitute for checking exact current content.
- "A full overwrite is simpler" risks deleting concurrent changes.

## Tool Protocol

1. Use `get_file_hash` or `prepare_edit`.
2. Use `preview_patch` to inspect the exact change.
3. Use `apply_text_edit` only with the matching `expected_hash`.
4. If the hash changed, reread and restart the edit.

## Stop Conditions

- Stop if the file hash changed between preparation and apply.
- Stop if the patch preview contains unrelated changes.
- Stop if the target text is ambiguous or appears in multiple places without a safe anchor.
- Stop if the next write would modify files outside the current user request.

## Verification

- The edit result includes a new hash.
- The final diff contains only intended changes.
- Follow-up tests or lint are run when code changed, or the reason they were not run is reported.
