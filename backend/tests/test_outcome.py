from __future__ import annotations

from types import SimpleNamespace

from solo_agent.outcome import build_evidence_timeline, judge_task_outcome


def test_outcome_judge_passes_with_diff_and_test_evidence() -> None:
    report = judge_task_outcome(
        user_input="Fix greeting bug",
        sandbox_diff="--- a/pkg/app.py\n+++ b/pkg/app.py\n",
        command_evidence=[{"command": "pytest -q tests/test_app.py", "result": {"returncode": 0}}],
    )

    assert report["status"] == "passed"
    assert report["approval_ready"] is True
    assert report["metadata"]["passed_command_count"] == 1


def test_outcome_judge_inconclusive_without_verification_evidence() -> None:
    report = judge_task_outcome(
        user_input="Fix greeting bug",
        patch_proposal={"id": "patch_1", "diff": "--- a/pkg/app.py\n+++ b/pkg/app.py\n"},
    )

    assert report["status"] == "inconclusive"
    assert report["approval_ready"] is True
    assert "No passing verification command evidence is available." in report["missing_evidence"]


def test_outcome_judge_blocks_dependency_failures() -> None:
    report = judge_task_outcome(
        user_input="Fix imports",
        sandbox_diff="--- a/pkg/app.py\n+++ b/pkg/app.py\n",
        failure_reports=[{"kind": "dependency_missing", "summary": "Missing dependency: requests"}],
    )

    assert report["status"] == "blocked"
    assert report["approval_ready"] is False


def test_evidence_timeline_aggregates_state_fields() -> None:
    state = SimpleNamespace(
        user_input="Fix app",
        plan="Plan",
        code_map_summary={"python_file_count": 1},
        impact_analysis={"related_tests": ["tests/test_app.py"]},
        sandbox_artifacts={"diff": "---"},
        failure_reports=[{"kind": "test_failure", "summary": "pytest failed"}],
        outcome_report={"status": "needs_fix", "summary": "Needs fix"},
        patch_proposal={"id": "patch_1"},
        git_artifact_proposal={"branch_name": "codex/fix-app"},
        snapshots={"sandbox_checkpoints": [{"label": "created"}]},
    )

    kinds = [item["kind"] for item in build_evidence_timeline(state)]

    assert kinds == [
        "user_request",
        "plan",
        "code_index",
        "impact_analysis",
        "sandbox_checkpoint",
        "sandbox_artifacts",
        "failure_report",
        "outcome_report",
        "patch_proposal",
        "git_artifact_proposal",
    ]
