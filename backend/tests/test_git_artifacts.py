from __future__ import annotations

from solo_agent.git_artifacts import propose_git_artifacts


def test_git_artifact_proposal_is_stable_and_auditable() -> None:
    proposal = propose_git_artifacts(
        user_input="Fix greeting bug in service",
        patch_proposal={"id": "patch_1", "summary": "Fix greeting bug"},
        outcome_report={"status": "passed", "risks": [{"severity": "low", "description": "Small patch"}]},
        evidence=[{"command": "pytest -q tests/test_service.py", "result": {"returncode": 0}}],
    )

    assert proposal["branch_name"] == "codex/fix-greeting-bug"
    assert proposal["commit_message"].startswith("Fix greeting bug")
    assert proposal["pr_title"] == "Fix greeting bug"
    assert "`pytest -q tests/test_service.py`: passed" in proposal["pr_description"]
    assert proposal["status"] == "proposal_only"
