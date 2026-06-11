from __future__ import annotations

import json

from solo_agent.skill_changes import SkillChangeProposal, apply_skill_change_proposal
from solo_agent.skill_coverage import SkillQualityError, assert_skill_quality, audit_skill_coverage
from solo_agent.skill_evolution import analyze_skill_evolution


def test_skill_coverage_reports_contract_recipe_and_policy_issues(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "workflows" / "bad-skill"
    recipes_dir = skill_dir / "references" / "recipes"
    scripts_dir = skill_dir / "scripts"
    recipes_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    metadata = {
        "hermes": {
            "recipes": [
                {"id": "missing", "file": "references/recipes/missing.yaml"},
                {"id": "unsafe", "file": "references/recipes/unsafe.yaml"},
            ]
        }
    }
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bad-skill\n"
        "description: Bad skill.\n"
        "required_tools: [not_a_tool]\n"
        f"metadata: {json.dumps(metadata)}\n"
        "---\n"
        "# Bad Skill\n",
        encoding="utf-8",
    )
    (recipes_dir / "unsafe.yaml").write_text(
        json.dumps(
            {
                "id": "unsafe",
                "name": "Unsafe",
                "steps": [{"id": "edit", "tool": "apply_text_edit", "arguments": {"path": "app.py"}}],
            }
        ),
        encoding="utf-8",
    )
    (recipes_dir / "orphan.yaml").write_text(
        json.dumps({"id": "orphan", "steps": [{"id": "snapshot", "tool": "workspace_snapshot", "arguments": {}}]}),
        encoding="utf-8",
    )
    (scripts_dir / "inspect.py").write_text("print('ok')\n", encoding="utf-8")

    report = audit_skill_coverage(tmp_path)

    issue_kinds = {issue.kind for issue in report.issues}
    assert "missing_contract_field" in issue_kinds
    assert "unknown_required_tool" in issue_kinds
    assert "missing_recipe_file" in issue_kinds
    assert "auto_recipe_step_blocked" in issue_kinds
    assert "orphan_recipe_file" in issue_kinds
    assert "undeclared_script_file" in issue_kinds
    assert report.summary["error_count"] >= 3


def test_skill_coverage_matrix_marks_common_flows_covered(tmp_path) -> None:
    _write_matrix_skill(
        tmp_path,
        "workflows/python-backend-change",
        "python-backend-change",
        ["inspect", "focused-test", "verify"],
        ["workspace_snapshot", "prepare_edit", "run_pytest"],
    )
    _write_matrix_skill(
        tmp_path,
        "workflows/debug-test-failure",
        "debug-test-failure",
        ["failure-triage"],
        ["search_text", "prepare_edit", "run_pytest"],
    )
    _write_matrix_skill(
        tmp_path,
        "workflows/code-review",
        "code-review",
        ["review-context"],
        ["workspace_snapshot", "run_pytest", "git_diff"],
    )
    _write_matrix_skill(
        tmp_path,
        "tools/hash-anchored-editing",
        "hash-anchored-editing",
        ["manual-hash-edit"],
        ["prepare_edit"],
    )
    _write_matrix_skill(
        tmp_path,
        "tools/tool-use-discipline",
        "tool-use-discipline",
        ["bounded-context-gathering"],
        ["workspace_snapshot", "run_pytest"],
    )

    report = audit_skill_coverage(tmp_path)

    statuses = {scenario.id: scenario.status for scenario in report.scenarios}
    assert statuses == {
        "python-backend-change": "covered",
        "debug-test-failure": "covered",
        "code-review": "covered",
        "hash-anchored-editing": "covered",
        "tool-use-discipline": "covered",
    }
    assert report.summary["covered_scenario_count"] == 5


def test_skill_coverage_sees_approved_evolution_recipe(tmp_path) -> None:
    _write_matrix_skill(
        tmp_path,
        "workflows/python-backend-change",
        "python-backend-change",
        ["inspect"],
        ["workspace_snapshot", "prepare_edit", "run_pytest"],
    )
    snapshot = {
        "session_id": "session-1",
        "run_id": "run-1",
        "user_input": "/skill python-backend-change run tests",
        "selected_skills": [
            {"name": "python-backend-change", "path": "skills/workflows/python-backend-change/SKILL.md"}
        ],
        "recipe_runs": [],
        "tool_calls": [
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "search_text", "arguments": {"query": "AgentState"}, "result": "ok"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ],
        "snapshots": {"explicit_skill_requests": ["python-backend-change"]},
        "response": "Done.",
    }
    evolution = analyze_skill_evolution(snapshot, workspace_root=tmp_path)
    assert evolution.proposal_payload is not None
    proposal = SkillChangeProposal(
        session_id=snapshot["session_id"],
        run_id=snapshot["run_id"],
        action=evolution.proposal_payload["action"],
        skill_name=evolution.proposal_payload["skill_name"],
        target_paths=evolution.proposal_payload["target_paths"],
        operations=evolution.proposal_payload["operations"],
    )

    applied = apply_skill_change_proposal(proposal, tmp_path)
    report = audit_skill_coverage(tmp_path)
    skill = next(item for item in report.skills if item.name == "python-backend-change")

    assert applied.status == "applied"
    assert "evolution-run-1" in skill.declared_recipes
    assert not any(
        issue.kind == "orphan_recipe_file" and issue.skill_name == "python-backend-change"
        for issue in report.issues
    )


def test_assert_skill_quality_passes_clean_skill_set(tmp_path) -> None:
    _write_clean_matrix(tmp_path)

    report = assert_skill_quality(tmp_path)

    assert report.summary["covered_scenario_count"] == 5
    assert report.summary["error_count"] == 0


def test_assert_skill_quality_blocks_clear_issues(tmp_path) -> None:
    _write_matrix_skill(
        tmp_path,
        "workflows/python-backend-change",
        "python-backend-change",
        ["inspect", "focused-test", "verify"],
        ["workspace_snapshot", "not_a_tool", "run_pytest"],
    )
    recipe_path = tmp_path / "skills" / "workflows" / "python-backend-change" / "references" / "recipes" / "focused-test.yaml"
    recipe_path.write_text(
        json.dumps(
            {
                "id": "focused-test",
                "name": "Focused test",
                "steps": [{"id": "edit", "tool": "apply_text_edit", "arguments": {"path": "app.py"}}],
            }
        ),
        encoding="utf-8",
    )
    _write_matrix_skill(
        tmp_path,
        "workflows/debug-test-failure",
        "debug-test-failure",
        ["failure-triage"],
        ["workspace_snapshot", "prepare_edit", "run_pytest"],
    )

    try:
        assert_skill_quality(tmp_path)
    except SkillQualityError as exc:
        blocker_kinds = {issue.kind for issue in exc.blockers}
    else:
        raise AssertionError("Expected the skill quality gate to fail")

    assert "unknown_required_tool" in blocker_kinds
    assert "auto_recipe_step_blocked" in blocker_kinds
    assert "scenario_not_covered" in blocker_kinds


def test_assert_skill_quality_allows_warning_only_report(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "reference" / "notes-only"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: notes-only\n"
        "description: Warning-only skill.\n"
        "---\n"
        "# Notes Only\n",
        encoding="utf-8",
    )

    report = assert_skill_quality(tmp_path, scenarios=[])

    assert report.summary["warning_count"] > 0
    assert report.summary["error_count"] == 0


def _write_matrix_skill(
    tmp_path,
    relative_dir: str,
    name: str,
    recipe_ids: list[str],
    required_tools: list[str],
) -> None:
    skill_dir = tmp_path / "skills" / relative_dir
    recipes_dir = skill_dir / "references" / "recipes"
    recipes_dir.mkdir(parents=True)
    recipes = [{"id": recipe_id, "file": f"references/recipes/{recipe_id}.yaml"} for recipe_id in recipe_ids]
    metadata = {"hermes": {"recipes": recipes}}
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {name} workflow.\n"
        f"required_tools: {json.dumps(required_tools)}\n"
        f"metadata: {json.dumps(metadata)}\n"
        "---\n"
        f"# {name}\n\n"
        "## Tool Protocol\n\n"
        "- Gather bounded context.\n"
        "- Verify with quality checks.\n\n"
        "## Stop Conditions\n\n"
        "- Stop when behavior is ambiguous.\n\n"
        "## Verification\n\n"
        "- Required checks pass.\n",
        encoding="utf-8",
    )
    for recipe_id in recipe_ids:
        (recipes_dir / f"{recipe_id}.yaml").write_text(
            json.dumps(
                {
                    "id": recipe_id,
                    "name": recipe_id,
                    "steps": [{"id": "snapshot", "tool": "workspace_snapshot", "arguments": {}}],
                }
            ),
            encoding="utf-8",
        )


def _write_clean_matrix(tmp_path) -> None:
    _write_matrix_skill(
        tmp_path,
        "workflows/python-backend-change",
        "python-backend-change",
        ["inspect", "focused-test", "verify"],
        ["workspace_snapshot", "prepare_edit", "run_pytest"],
    )
    _write_matrix_skill(
        tmp_path,
        "workflows/debug-test-failure",
        "debug-test-failure",
        ["failure-triage"],
        ["search_text", "prepare_edit", "run_pytest"],
    )
    _write_matrix_skill(
        tmp_path,
        "workflows/code-review",
        "code-review",
        ["review-context"],
        ["workspace_snapshot", "run_pytest"],
    )
    _write_matrix_skill(
        tmp_path,
        "tools/hash-anchored-editing",
        "hash-anchored-editing",
        ["manual-hash-edit"],
        ["prepare_edit"],
    )
    _write_matrix_skill(
        tmp_path,
        "tools/tool-use-discipline",
        "tool-use-discipline",
        ["bounded-context-gathering"],
        ["workspace_snapshot", "run_pytest"],
    )
