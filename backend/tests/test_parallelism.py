from __future__ import annotations

from solo_agent.workflow.parallelism import (
    TaskCandidate,
    evaluate_independence,
    extract_task_candidates_from_text,
)


def test_all_four_conditions_pass_allows_parallel() -> None:
    tasks = [
        TaskCandidate(
            id="T1",
            title="Add provider config tests",
            domain="providers",
            read_paths=("backend/src/solo_agent/providers/",),
            write_paths=("backend/tests/test_provider_config.py",),
            verify_commands=("pytest backend/tests/test_provider_config.py -q",),
        ),
        TaskCandidate(
            id="T2",
            title="Add memory inbox tests",
            domain="memory",
            read_paths=("backend/src/solo_agent/memory/",),
            write_paths=("backend/tests/test_memory_inbox.py",),
            verify_commands=("pytest backend/tests/test_memory_inbox.py -q",),
        ),
    ]

    decision = evaluate_independence(tasks, max_parallel=3)

    assert decision.mode == "parallel"
    assert decision.allowed is True
    assert decision.max_parallel == 2
    assert all(item.passed for item in decision.conditions)
    assert decision.groups == (("T1", "T2"),)


def test_shared_write_path_forces_serial() -> None:
    tasks = [
        TaskCandidate(
            id="T1",
            title="Change runtime events",
            domain="workflow",
            read_paths=("backend/src/solo_agent/workflow/runtime.py",),
            write_paths=("backend/src/solo_agent/workflow/runtime.py",),
            verify_commands=("pytest backend/tests/test_workflow_runtime.py -q",),
        ),
        TaskCandidate(
            id="T2",
            title="Change runtime strategy",
            domain="workflow",
            read_paths=("backend/src/solo_agent/workflow/runtime.py",),
            write_paths=("backend/src/solo_agent/workflow/runtime.py",),
            verify_commands=("pytest backend/tests/test_workflow_runtime.py -q",),
        ),
    ]

    decision = evaluate_independence(tasks, max_parallel=3)

    assert decision.mode == "serial"
    assert decision.allowed is False
    assert any(
        condition.condition == "write_set_independence" and not condition.passed
        for condition in decision.conditions
    )
    assert decision.groups == (("T1",), ("T2",))


def test_missing_verification_forces_serial() -> None:
    tasks = [
        TaskCandidate(
            id="T1",
            title="Provider work",
            domain="providers",
            read_paths=("backend/src/solo_agent/providers/",),
            write_paths=("backend/src/solo_agent/providers/factory.py",),
            verify_commands=(),
        ),
        TaskCandidate(
            id="T2",
            title="Memory work",
            domain="memory",
            read_paths=("backend/src/solo_agent/memory/",),
            write_paths=("backend/src/solo_agent/memory/repository.py",),
            verify_commands=("pytest backend/tests/test_memory.py -q",),
        ),
    ]

    decision = evaluate_independence(tasks)

    assert decision.mode == "serial"
    assert any(
        condition.condition == "verification_independence" and not condition.passed
        for condition in decision.conditions
    )


def test_plan_without_structured_metadata_forces_serial() -> None:
    text = "1. Fix the app. 2. Run tests."

    tasks = extract_task_candidates_from_text(text)
    decision = evaluate_independence(tasks)

    assert len(tasks) == 1
    assert tasks[0].risk_flags == ("unstructured_plan",)
    assert decision.mode == "serial"


def test_nested_write_path_overlap_forces_serial() -> None:
    tasks = [
        TaskCandidate(
            id="T1",
            title="Refactor workflow package",
            domain="workflow",
            read_paths=("backend/src/solo_agent/workflow/",),
            write_paths=("backend/src/solo_agent/workflow/",),
            verify_commands=("pytest backend/tests/test_workflow.py -q",),
        ),
        TaskCandidate(
            id="T2",
            title="Add runtime test",
            domain="runtime",
            read_paths=("backend/src/solo_agent/workflow/runtime.py",),
            write_paths=("backend/src/solo_agent/workflow/runtime.py",),
            verify_commands=("pytest backend/tests/test_runtime.py -q",),
        ),
    ]

    decision = evaluate_independence(tasks)

    assert decision.mode == "serial"
    assert decision.conflicts
    assert any("overlaps" in item for item in decision.conflicts)


def test_extract_task_candidates_from_json_fence() -> None:
    text = '''
Plan.

```json
{
  "parallel_tasks": [
    {
      "id": "T1",
      "title": "Provider tests",
      "domain": "providers",
      "read_paths": ["backend/src/solo_agent/providers/"],
      "write_paths": ["backend/tests/test_provider_config.py"],
      "verify_commands": ["pytest backend/tests/test_provider_config.py -q"]
    },
    {
      "id": "T2",
      "title": "Memory tests",
      "domain": "memory",
      "read_paths": ["backend/src/solo_agent/memory/"],
      "write_paths": ["backend/tests/test_memory_inbox.py"],
      "verify_commands": ["pytest backend/tests/test_memory_inbox.py -q"]
    }
  ]
}
```
'''

    tasks = extract_task_candidates_from_text(text)

    assert [task.id for task in tasks] == ["T1", "T2"]
    assert tasks[0].write_paths == ("backend/tests/test_provider_config.py",)
