---
name: hash-anchored-editing
description: Safe editing protocol requiring file hash validation, patch preview, and anchored writes.
category: tools
triggers: [edit file, modify code, patch, write code, refactor]
red_flags: [line number only, stale context, direct write, no preview]
required_tools: [get_file_hash, prepare_edit, preview_patch, apply_text_edit]
---

# Hash-Anchored Editing

## Iron Law

Never write to an existing file unless the edit is anchored to the current file hash.

## Procedure

1. Use `get_file_hash` or `prepare_edit`.
2. Use `preview_patch` to inspect the exact change.
3. Use `apply_text_edit` only with the matching `expected_hash`.
4. If the hash changed, reread and restart the edit.

## Red Flags

- Editing from stale context.
- Replacing text that appears multiple times.
- Applying a patch without preview.

## Verification

- The edit result includes a new hash.
- Follow-up tests or lint are run when code changed.
