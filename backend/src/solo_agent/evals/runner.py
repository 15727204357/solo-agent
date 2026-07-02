"""Small local eval harness for coding-agent behavior."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .case import EvalCase, EvalResult
from .report import eval_report_markdown
from .scoring import score_eval_case, score_route_case

EvalExecutor = Callable[[EvalCase, Path], dict[str, Any]]


def run_eval_suite(
    cases: Sequence[EvalCase],
    *,
    runtime_root: str | Path,
    executor: EvalExecutor | None = None,
) -> dict[str, Any]:
    root = Path(runtime_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in cases:
        results.append(_run_case(case, root / case.id, executor).to_dict())
    return {
        "results": results,
        "summary": {
            "case_count": len(results),
            "passed": sum(1 for result in results if result.get("passed")),
            "pass_rate": (sum(1 for result in results if result.get("passed")) / len(results)) if results else 0.0,
            "patch_applied": sum(1 for result in results if result.get("patch_applied")),
            "tests_failed": sum(int(result.get("tests_failed") or 0) for result in results),
            "tool_calls": sum(int(result.get("tool_calls") or 0) for result in results),
            "human_interventions": sum(int(result.get("human_interventions") or 0) for result in results),
            "failure_classes": sorted(
                {
                    str(result.get("failure_class") or "")
                    for result in results
                    if str(result.get("failure_class") or "")
                }
            ),
        },
        "markdown": eval_report_markdown(results),
    }


def _run_case(case: EvalCase, case_root: Path, executor: EvalExecutor | None) -> EvalResult:
    started = time.perf_counter()
    case_root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in case.initial_files.items():
        path = case_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    before = _file_hashes(case_root)
    output = executor(case, case_root) if executor else {}
    after = _file_hashes(case_root)
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    changed.extend(sorted(path for path in set(before) - set(after)))
    tests_failed = int(output.get("tests_failed") or 0)
    outcome_status = str(output.get("outcome_status") or "unknown")
    unrelated = sorted(set(changed) - set(case.expected_changed_files)) if case.expected_changed_files else []
    route_plan = output.get("route_plan") if isinstance(output.get("route_plan"), dict) else {}
    route_events = [event for event in output.get("route_events") or [] if isinstance(event, dict)]
    route_passed, route_score, route_notes = score_route_case(case, route_plan, route_events)
    passed, score, notes = score_eval_case(
        case,
        changed,
        tests_failed=tests_failed,
        outcome_status=outcome_status,
        route_plan=route_plan,
        route_events=route_events,
    )
    return EvalResult(
        case_id=case.id,
        passed=passed,
        score=score,
        changed_files=changed,
        task_type=case.task_type,
        suite_id=case.suite_id,
        patch_applied=bool(output.get("patch_applied", bool(changed))),
        unrelated_changed_files=unrelated,
        tests_passed=int(output.get("tests_passed") or 0),
        tests_failed=tests_failed,
        sandbox_violations=list(output.get("sandbox_violations") or []),
        outcome_status=outcome_status,
        iterations=int(output.get("iterations") or 0),
        tool_calls=int(output.get("tool_calls") or 0),
        human_interventions=int(output.get("human_interventions") or 0),
        failure_class=str(output.get("failure_class") or ""),
        route_passed=route_passed,
        route_score=route_score,
        route_notes=route_notes,
        duration_seconds=round(time.perf_counter() - started, 3),
        notes=notes + [str(item) for item in output.get("notes", [])],
    )


def _file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = str(hash(path.read_bytes()))
    return hashes
