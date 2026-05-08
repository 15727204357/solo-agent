from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ParallelExecutionMode = Literal["parallel", "serial"]

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_GLOBAL_VERIFY_COMMANDS = {
    "pytest",
    "python -m pytest",
    "ruff check .",
    "uv run pytest",
    "uv run python -m pytest",
    "uv run --extra dev python -m pytest",
}
_SHARED_WRITE_PATHS = {
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    ".env",
    ".env.example",
}


@dataclass(frozen=True)
class TaskCandidate:
    id: str
    title: str
    description: str = ""
    domain: str = "unknown"
    read_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    verify_commands: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    needs_global_context: bool = False
    risk_flags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int = 1) -> TaskCandidate:
        task_id = str(value.get("id") or value.get("task_id") or f"T{index}")
        title = str(value.get("title") or value.get("subject") or value.get("name") or task_id)
        read_paths = _tuple_of_strings(value.get("read_paths") or value.get("reads") or value.get("context_paths"))
        write_paths = _tuple_of_strings(value.get("write_paths") or value.get("writes") or value.get("paths"))
        verify_commands = _tuple_of_strings(
            value.get("verify_commands") or value.get("verification") or value.get("test_commands")
        )
        domain = str(value.get("domain") or _infer_domain(write_paths, read_paths) or "unknown")
        return cls(
            id=task_id,
            title=title,
            description=str(value.get("description") or ""),
            domain=domain,
            read_paths=tuple(_normalize_path(item) for item in read_paths if _normalize_path(item)),
            write_paths=tuple(_normalize_path(item) for item in write_paths if _normalize_path(item)),
            verify_commands=tuple(_normalize_command(item) for item in verify_commands if str(item).strip()),
            depends_on=_tuple_of_strings(value.get("depends_on") or value.get("dependencies")),
            needs_global_context=bool(value.get("needs_global_context", False)),
            risk_flags=_tuple_of_strings(value.get("risk_flags") or value.get("risks")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "domain": self.domain,
            "read_paths": list(self.read_paths),
            "write_paths": list(self.write_paths),
            "verify_commands": list(self.verify_commands),
            "depends_on": list(self.depends_on),
            "needs_global_context": self.needs_global_context,
            "risk_flags": list(self.risk_flags),
        }


@dataclass(frozen=True)
class ConditionVerdict:
    condition: str
    passed: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "passed": self.passed,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ParallelismDecision:
    mode: ParallelExecutionMode
    allowed: bool
    reason: str
    conditions: tuple[ConditionVerdict, ...]
    tasks: tuple[TaskCandidate, ...]
    max_parallel: int = 1
    groups: tuple[tuple[str, ...], ...] = ()
    conflicts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "reason": self.reason,
            "max_parallel": self.max_parallel,
            "groups": [list(group) for group in self.groups],
            "conflicts": list(self.conflicts),
            "conditions": [item.to_dict() for item in self.conditions],
            "tasks": [item.to_dict() for item in self.tasks],
        }


def extract_task_candidates_from_text(text: str) -> tuple[TaskCandidate, ...]:
    for block in _FENCED_BLOCK_RE.findall(text or ""):
        payload = _try_json(block)
        tasks = _tasks_from_payload(payload)
        if tasks:
            return tasks

    payload = _try_json(text or "")
    tasks = _tasks_from_payload(payload)
    if tasks:
        return tasks

    return (
        TaskCandidate(
            id="T1",
            title="Unstructured task",
            description="Planner did not provide structured parallel_tasks metadata.",
            domain="unknown",
            risk_flags=("unstructured_plan",),
        ),
    )


def evaluate_independence(
    tasks: Iterable[TaskCandidate],
    *,
    max_parallel: int = 3,
) -> ParallelismDecision:
    task_tuple = tuple(tasks)
    if len(task_tuple) < 2:
        condition = ConditionVerdict(
            condition="minimum_task_count",
            passed=False,
            reason="Parallel execution requires at least two structured task candidates.",
            evidence={"task_count": len(task_tuple)},
        )
        return _serial(task_tuple, (condition,), ("fewer than two tasks",))

    max_parallel = max(1, min(int(max_parallel), len(task_tuple)))
    verdicts = (
        _check_problem_domain_independence(task_tuple),
        _check_context_independence(task_tuple),
        _check_write_set_independence(task_tuple),
        _check_verification_independence(task_tuple),
    )
    conflicts = tuple(
        conflict
        for verdict in verdicts
        for conflict in verdict.evidence.get("conflicts", [])
        if isinstance(conflict, str)
    )

    if all(verdict.passed for verdict in verdicts):
        task_ids = tuple(task.id for task in task_tuple)
        return ParallelismDecision(
            mode="parallel",
            allowed=True,
            reason="All four independence conditions passed.",
            conditions=verdicts,
            tasks=task_tuple,
            max_parallel=max_parallel,
            groups=(task_ids,),
            conflicts=(),
        )

    failed = [verdict.condition for verdict in verdicts if not verdict.passed]
    return _serial(
        task_tuple,
        verdicts,
        conflicts or tuple(f"failed condition: {condition}" for condition in failed),
    )


def _check_problem_domain_independence(tasks: tuple[TaskCandidate, ...]) -> ConditionVerdict:
    domains = [task.domain for task in tasks]
    unknown = [task.id for task in tasks if task.domain in {"", "unknown", "general"}]
    dependencies = {task.id: list(task.depends_on) for task in tasks if task.depends_on}
    related = [
        task.id
        for task in tasks
        if any(flag in {"related", "shared_root_cause", "same_subsystem"} for flag in task.risk_flags)
    ]

    conflicts: list[str] = []
    if unknown:
        conflicts.append(f"tasks missing scoped domains: {unknown}")
    if dependencies:
        conflicts.append(f"tasks declare dependencies: {dependencies}")
    if related:
        conflicts.append(f"tasks flagged as related/shared-root-cause: {related}")
    if len(set(domains)) != len(domains):
        conflicts.append(f"domains are not unique: {domains}")

    return ConditionVerdict(
        condition="problem_domain_independence",
        passed=not conflicts,
        reason="Each task has a distinct problem domain and no declared dependency."
        if not conflicts
        else "Tasks are not safely separated by problem domain.",
        evidence={"domains": domains, "dependencies": dependencies, "conflicts": conflicts},
    )


def _check_context_independence(tasks: tuple[TaskCandidate, ...]) -> ConditionVerdict:
    conflicts: list[str] = []
    for task in tasks:
        if task.needs_global_context:
            conflicts.append(f"{task.id} needs global context")
        if not task.read_paths and not task.write_paths and not task.verify_commands:
            conflicts.append(f"{task.id} has no scoped context evidence")
        if any(flag in {"needs_full_context", "requires_previous_result"} for flag in task.risk_flags):
            conflicts.append(f"{task.id} has context risk flags: {task.risk_flags}")

    return ConditionVerdict(
        condition="context_independence",
        passed=not conflicts,
        reason="Each task has enough local context to be executed independently."
        if not conflicts
        else "One or more tasks require global or previous-task context.",
        evidence={"conflicts": conflicts},
    )


def _check_write_set_independence(tasks: tuple[TaskCandidate, ...]) -> ConditionVerdict:
    conflicts: list[str] = []
    write_sets: dict[str, tuple[str, ...]] = {}

    for task in tasks:
        if not task.write_paths:
            conflicts.append(f"{task.id} has no declared write_paths")
        write_sets[task.id] = task.write_paths
        for path in task.write_paths:
            if path in _SHARED_WRITE_PATHS:
                conflicts.append(f"{task.id} writes shared coordination file {path}")

    for index, left in enumerate(tasks):
        for right in tasks[index + 1:]:
            for left_path in left.write_paths:
                for right_path in right.write_paths:
                    if _paths_overlap(left_path, right_path):
                        conflicts.append(f"{left.id}:{left_path} overlaps {right.id}:{right_path}")

    return ConditionVerdict(
        condition="write_set_independence",
        passed=not conflicts,
        reason="No task write sets overlap."
        if not conflicts
        else "One or more task write sets overlap or are unknown.",
        evidence={"write_sets": write_sets, "conflicts": conflicts},
    )


def _check_verification_independence(tasks: tuple[TaskCandidate, ...]) -> ConditionVerdict:
    conflicts: list[str] = []
    command_sets: dict[str, tuple[str, ...]] = {}

    for task in tasks:
        command_sets[task.id] = task.verify_commands
        if not task.verify_commands:
            conflicts.append(f"{task.id} has no verify_commands")
            continue
        for command in task.verify_commands:
            if _is_global_verify_command(command):
                conflicts.append(f"{task.id} uses global verification command: {command}")

    seen: dict[str, str] = {}
    for task in tasks:
        for command in task.verify_commands:
            owner = seen.get(command)
            if owner and owner != task.id:
                conflicts.append(f"{task.id} shares verification command with {owner}: {command}")
            seen[command] = task.id

    return ConditionVerdict(
        condition="verification_independence",
        passed=not conflicts,
        reason="Each task has targeted, non-overlapping verification."
        if not conflicts
        else "Verification is missing, global, or shared across tasks.",
        evidence={"verify_commands": command_sets, "conflicts": conflicts},
    )


def _serial(
    tasks: tuple[TaskCandidate, ...],
    conditions: tuple[ConditionVerdict, ...],
    conflicts: tuple[str, ...],
) -> ParallelismDecision:
    return ParallelismDecision(
        mode="serial",
        allowed=False,
        reason="Parallel execution denied; falling back to serial execution.",
        conditions=conditions,
        tasks=tasks,
        max_parallel=1,
        groups=tuple((task.id,) for task in tasks),
        conflicts=conflicts,
    )


def _tasks_from_payload(payload: Any) -> tuple[TaskCandidate, ...]:
    if not isinstance(payload, Mapping):
        return ()
    raw_tasks = payload.get("parallel_tasks") or payload.get("tasks") or payload.get("task_candidates")
    if not isinstance(raw_tasks, list):
        return ()
    tasks = [
        TaskCandidate.from_mapping(item, index=index)
        for index, item in enumerate(raw_tasks, start=1)
        if isinstance(item, Mapping)
    ]
    return tuple(tasks)


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)


def _normalize_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _normalize_command(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _infer_domain(write_paths: tuple[str, ...], read_paths: tuple[str, ...]) -> str:
    for path in (*write_paths, *read_paths):
        normalized = _normalize_path(path)
        parts = [part for part in normalized.split("/") if part]
        if "solo_agent" in parts:
            index = parts.index("solo_agent")
            if len(parts) > index + 1:
                return parts[index + 1]
        if len(parts) >= 2:
            return "/".join(parts[:2])
        if parts:
            return parts[0]
    return "unknown"


def _paths_overlap(left: str, right: str) -> bool:
    left = _normalize_path(left)
    right = _normalize_path(right)
    if not left or not right:
        return False
    return left == right or right.startswith(f"{left}/") or left.startswith(f"{right}/")


def _is_global_verify_command(command: str) -> bool:
    command = _normalize_command(command)
    if command in _GLOBAL_VERIFY_COMMANDS:
        return True
    if command.endswith(" python -m pytest") or command.endswith(" pytest"):
        return True
    return False
