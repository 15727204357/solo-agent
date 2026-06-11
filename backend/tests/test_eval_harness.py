from __future__ import annotations

from pathlib import Path

from solo_agent.evals import EvalCase, eval_report_markdown, run_eval_suite


def test_eval_runner_scores_fixture_cases(tmp_path: Path) -> None:
    cases = [
        EvalCase(
            id="bugfix",
            user_request="Fix bug",
            initial_files={"pkg/app.py": "value = 1\n"},
            expected_changed_files=["pkg/app.py"],
        ),
        EvalCase(
            id="forbidden",
            user_request="Do not touch config",
            initial_files={"pkg/app.py": "value = 1\n", "config.toml": "x=1\n"},
            forbidden_changed_files=["config.toml"],
        ),
    ]

    def executor(case: EvalCase, root: Path) -> dict[str, object]:
        if case.id == "bugfix":
            (root / "pkg" / "app.py").write_text("value = 2\n", encoding="utf-8")
        else:
            (root / "config.toml").write_text("x=2\n", encoding="utf-8")
        return {"tests_passed": 1, "outcome_status": "passed", "iterations": 1, "tool_calls": 2}

    report = run_eval_suite(cases, runtime_root=tmp_path / "evals", executor=executor)

    assert report["summary"]["case_count"] == 2
    assert report["results"][0]["passed"] is True
    assert report["results"][1]["passed"] is False
    assert "# Coding Agent Eval Report" in eval_report_markdown(report["results"])
