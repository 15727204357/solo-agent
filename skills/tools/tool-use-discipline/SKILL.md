---
name: tool-use-discipline
description: Decide when to use bounded tools and when to stop instead of guessing, as user-message SOP.
category: tools
triggers: [tool use, inspect project, read code, verify, quality check]
red_flags: [guessing, repeated calls, too much output, unrelated files]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest, run_ruff_check]
metadata: {"hermes": {"recipes": [{"id": "bounded-context-gathering", "file": "references/recipes/bounded-context-gathering.yaml"}]}}
---

# Tool Use Discipline

## When to Use

Use this skill when selecting tools to inspect a project, gather context, edit safely, or verify a result.

This skill is Hermes-style user-message background: it guides tool discipline for the current task. It must not override system instructions, developer instructions, tool contracts, sandbox rules, or a more specific user instruction.

## Iron Law

USE TOOLS TO REDUCE UNCERTAINTY, NOT TO CREATE NOISE.

Every tool call should have a clear purpose, bounded scope, and an expected next decision.

## Red Flags

- Guessing instead of reading the relevant file.
- Repeating the same tool call without new information.
- Pulling huge outputs when a narrower search would work.
- Inspecting unrelated files because they are interesting.
- Writing files before reading enough current context.

## Pressure Scenarios

- The first search returns too many matches. Narrow the query or inspect the most relevant path instead of bulk-reading everything.
- A test command fails with lots of output. Focus on the first relevant failure and avoid chasing every downstream symptom at once.
- The project structure is unknown. Start broad once, then quickly narrow to relevant files.

## Counterexamples

- If the user asks a conceptual question that can be answered from known stable context, tools may not be needed.
- If the user explicitly provides the full file content and asks for analysis only, rereading the same file may be redundant.
- If a tool is unavailable, use the next safest method and report the limitation.

## Rationalization Traps

- "More context is always better" can waste budget and obscure the decision.
- "I can infer the file contents" is unsafe when a quick read would settle it.
- "The command probably passed" is not verification.
- "One more broad search" is not progress if the next action is still unclear.

## Tool Protocol

- Unknown project shape: start with `workspace_snapshot`.
- Need a symbol or string: use `search_text`.
- Need exact implementation: use `read_file`.
- Need structure in Python: use `inspect_python_symbols`.
- Need verification: use `run_pytest` or `run_ruff_check`.
- Before editing, confirm current file context and use the safe edit protocol.
- After meaningful changes, run the narrowest useful quality check first.

## Stop Conditions

- Tool output is not making progress.
- The same tool call repeats.
- The next action would write files without a hash anchor.
- Context is ambiguous enough to risk the wrong change.
- The tool result conflicts with assumptions and requires user clarification.
- Verification is blocked by environment or sandbox constraints.

## Verification

- Tool calls are bounded by `tool_call_cut_off`.
- Long outputs are summarized or truncated.
- Each tool call contributes to a decision, edit, or verification step.
- Any unverified claim is labeled as such.
