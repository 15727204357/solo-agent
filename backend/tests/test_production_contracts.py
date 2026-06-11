from __future__ import annotations

import json
from pathlib import Path

import solo_agent.web.routes as routes_module
from fastapi.testclient import TestClient
from pydantic import ValidationError
from solo_agent.agent.actions import AgentAction, action_requires_approval
from solo_agent.evals import EvalCase, run_eval_suite
from solo_agent.settings import Settings
from solo_agent.skill_outcomes import SkillOutcomeRecord, summarize_skill_outcomes
from solo_agent.skill_schema import audit_skill_schemas, validate_skill_file
from solo_agent.web.app import create_app
from solo_agent.web.routes import get_repository, get_runner
from solo_agent.web.store import InMemorySessionRepository
from solo_agent.workflow.sandbox.workspace_backend import create_workspace_backend


class FakeRunner:
    def __init__(self, repo: InMemorySessionRepository) -> None:
        self.repo = repo

    async def run(self, session_id: str, run_id: str) -> None:
        await self.repo.mark_run_status(session_id, run_id, "completed")


def test_agent_action_schema_maps_tools_and_approval() -> None:
    read_action = AgentAction(kind="read", arguments={"path": "README.md"})
    edit_action = AgentAction(kind="edit_apply_in_sandbox", arguments={"path": "app.py"})
    final_action = AgentAction(kind="final", final_response="done")

    assert read_action.to_tool_call() == {"name": "read_file", "arguments": {"path": "README.md"}}
    assert action_requires_approval(read_action, approval_mode="confirm") is False
    assert action_requires_approval(edit_action, approval_mode="confirm") is True
    assert action_requires_approval(final_action, approval_mode="manual_only") is False
    try:
        AgentAction(kind="read", tool_name="bad;tool")
    except ValidationError:
        pass
    else:
        raise AssertionError("Expected invalid tool_name to fail validation")


def test_workspace_backend_copy_tracks_diff_and_metadata(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text("old\n", encoding="utf-8")
    backend = create_workspace_backend("copy", tmp_path, session_id="s1", run_id="r1")

    workspace = backend.prepare()
    (workspace.command_workspace_root / "pkg" / "app.py").write_text("new\n", encoding="utf-8")
    diff = backend.diff()
    metadata = backend.metadata()

    assert workspace.created is True
    assert (tmp_path / "pkg" / "app.py").read_text(encoding="utf-8") == "old\n"
    assert diff["changed_files"] == ["pkg/app.py"]
    assert metadata["kind"] == "copy"
    assert metadata["isolated"] is True
    assert backend.cleanup()["cleanup"] == "completed"


def test_skill_schema_validates_recipe_policy_and_disabled_state(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "workflows" / "python"
    recipes_dir = skill_dir / "references" / "recipes"
    recipes_dir.mkdir(parents=True)
    metadata = {"hermes": {"recipes": [{"id": "bad", "file": "references/recipes/bad.yaml"}]}}
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: Python Skill\n"
        "description: Python workflow.\n"
        "enabled: false\n"
        f"metadata: {json.dumps(metadata)}\n"
        "---\n"
        "# Python Skill\n",
        encoding="utf-8",
    )
    (recipes_dir / "bad.yaml").write_text(
        json.dumps({"id": "bad", "steps": [{"id": "edit", "tool": "apply_text_edit", "arguments": {"path": "x.py"}}]}),
        encoding="utf-8",
    )

    report = validate_skill_file(skill_file)
    audit = audit_skill_schemas(tmp_path)

    assert report["skill"]["enabled"] is False
    assert any(issue["kind"] == "auto_recipe_step_blocked" for issue in report["issues"])
    assert audit["summary"]["disabled_count"] == 1
    assert audit["summary"]["error_count"] == 1


def test_eval_metrics_capture_patch_and_intervention_counts(tmp_path: Path) -> None:
    case = EvalCase(
        id="repo-bugfix-1",
        user_request="fix bug",
        initial_files={"app.py": "old\n", "notes.md": "keep\n"},
        expected_changed_files=["app.py"],
        task_type="repo_bugfix",
        suite_id="smoke",
    )

    def executor(_case: EvalCase, root: Path) -> dict[str, object]:
        (root / "app.py").write_text("new\n", encoding="utf-8")
        (root / "notes.md").write_text("changed\n", encoding="utf-8")
        return {"tests_passed": 1, "tool_calls": 4, "human_interventions": 1, "outcome_status": "passed"}

    report = run_eval_suite([case], runtime_root=tmp_path / "evals", executor=executor)
    result = report["results"][0]

    assert result["patch_applied"] is True
    assert result["unrelated_changed_files"] == ["notes.md"]
    assert report["summary"]["tool_calls"] == 4
    assert report["summary"]["human_interventions"] == 1
    assert "| repo-bugfix-1 | repo_bugfix |" in report["markdown"]


def test_skill_outcome_summary_counts_recipe_statuses() -> None:
    summary = summarize_skill_outcomes(
        [
            SkillOutcomeRecord(skill_name="python-backend-change", recipe_status="completed", verified=True),
            SkillOutcomeRecord(skill_name="python-backend-change", recipe_status="blocked", blocked_steps=2),
        ]
    )

    skill = summary["skills"][0]
    assert skill["selected_count"] == 2
    assert skill["verified_count"] == 1
    assert skill["recipe_completed_count"] == 1
    assert skill["recipe_blocked_count"] == 1
    assert skill["blocked_step_count"] == 2


def test_web_run_metadata_accepts_production_controls(tmp_path: Path) -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)
    app.dependency_overrides[routes_module.get_settings] = lambda: Settings(workspace_root=tmp_path)

    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={"title": "Prod"}).json()["id"]
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json={
                "prompt": "fix production bug",
                "tool_loop_mode": "model",
                "approval_mode": "manual_only",
                "workspace_backend": "copy",
                "eval_suite_id": "smoke-prod",
            },
        )

    assert response.status_code == 202
    metadata = response.json()["metadata"]
    assert metadata["tool_loop_mode"] == "model"
    assert metadata["approval_mode"] == "manual_only"
    assert metadata["workspace_backend"] == "copy"
    assert metadata["eval_suite_id"] == "smoke-prod"
