---
name: tool-use-discipline
description: Decide when to use bounded tools and when to stop instead of guessing.
category: tools
triggers: [tool use, inspect project, read code, verify, quality check]
red_flags: [guessing, repeated calls, too much output, unrelated files]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest, run_ruff_check]
---

# Tool Use Discipline

## Tool Rules

- Unknown project shape: start with `workspace_snapshot`.
- Need a symbol or string: use `search_text`.
- Need exact implementation: use `read_file`.
- Need structure in Python: use `inspect_python_symbols`.
- Need verification: use `run_pytest` or `run_ruff_check`.

## Stop Conditions

- Tool output is not making progress.
- The same tool call repeats.
- The next action would write files without a hash anchor.
- Context is ambiguous enough to risk the wrong change.

## Verification

- Tool calls are bounded by `tool_call_cut_off`.
- Long outputs are summarized or truncated.
