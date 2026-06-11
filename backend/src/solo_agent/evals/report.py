"""Eval suite report rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def eval_report_markdown(results: Sequence[Mapping[str, object]]) -> str:
    passed = sum(1 for result in results if result.get("passed"))
    lines = [
        "# Coding Agent Eval Report",
        "",
        f"- Cases: {len(results)}",
        f"- Passed: {passed}",
        f"- Pass rate: {(passed / len(results) * 100) if results else 0:.1f}%",
        f"- Patch applied: {sum(1 for result in results if result.get('patch_applied'))}",
        f"- Tool calls: {sum(int(result.get('tool_calls') or 0) for result in results)}",
        f"- Human interventions: {sum(int(result.get('human_interventions') or 0) for result in results)}",
        "",
        "| Case | Type | Passed | Score | Outcome | Failure | Changed Files |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            "| {case} | {task_type} | {passed} | {score} | {outcome} | {failure} | {files} |".format(
                case=result.get("case_id", ""),
                task_type=result.get("task_type", ""),
                passed="yes" if result.get("passed") else "no",
                score=result.get("score", 0),
                outcome=result.get("outcome_status", "unknown"),
                failure=result.get("failure_class", ""),
                files=", ".join(str(item) for item in result.get("changed_files", []) or []),
            )
        )
    return "\n".join(lines)
