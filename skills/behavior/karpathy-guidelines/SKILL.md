---
name: karpathy-guidelines
description: Behavioral constraints for reliable AI coding: think first, keep it simple, make surgical changes, and define verifiable goals.
category: behavior
triggers: [coding, review, refactor, debugging, implementation, backend]
red_flags: [ambiguous request, overengineering, scope creep, vague done]
required_tools: [workspace_snapshot, search_text, read_file, run_pytest]
---

# Karpathy Guidelines

## Rules

1. Think before coding: surface assumptions and ambiguity before acting.
2. Simplicity first: implement the smallest solution that satisfies the task.
3. Surgical changes: touch only files directly required by the task.
4. Goal-driven execution: define how success will be verified.

## Red Flags

- Adding abstractions that were not requested.
- Refactoring adjacent code without need.
- Guessing when a short question or tool read would resolve uncertainty.
- Declaring success without verification.

## Tool Protocol

- Use context tools before editing unfamiliar code.
- Use quality tools after meaningful changes.
- Stop and ask if the task has multiple high-impact interpretations.

## Verification

- Every changed line traces back to the user request.
- Success criteria are explicit and checked.
