from __future__ import annotations

import json

from solo_agent.skill_changes import SkillChangeProposal, apply_skill_change_proposal
from solo_agent.skill_evolution import analyze_skill_evolution
from solo_agent.skill_recipes import parse_structured_recipe_text, recipe_from_payload
from solo_agent.tools import create_default_registry


def test_skill_evolution_proposes_recipe_for_successful_safe_sequence(tmp_path) -> None:
    _write_skill(tmp_path)
    snapshot = _snapshot(
        tool_calls=[
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "search_text", "arguments": {"query": "AgentState"}, "result": "ok"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ]
    )

    result = analyze_skill_evolution(snapshot, workspace_root=tmp_path)

    assert result.proposal_payload is not None
    assert result.candidates[0].change_kind == "add_recipe"
    assert result.candidates[0].confidence >= 0.72
    operation = result.candidates[0].proposed_operations[0]
    assert operation.action == "write_file"
    assert operation.path == "workflows/python-backend-change/references/recipes/evolution-run-1.yaml"
    assert "pytest" in (operation.content or "")
    assert len(result.candidates[0].proposed_operations) == 2
    assert result.candidates[0].proposed_operations[1].action == "patch"
    assert result.proposal_payload["evolution"]["recipe_id"] == "evolution-run-1"
    assert result.proposal_payload["evolution"]["validated"] is True

    recipe_payload = parse_structured_recipe_text(operation.content or "")
    parsed = recipe_from_payload(
        recipe_payload,
        skill={"name": "python-backend-change", "path": "skills/workflows/python-backend-change/SKILL.md"},
        source_file=operation.path,
    )
    assert parsed.id == "evolution-run-1"
    assert parsed.steps[-1].tool == "run_command"
    assert "args_json" not in (operation.content or "")


def test_skill_evolution_proposes_recovery_for_blocked_recipe_step(tmp_path) -> None:
    _write_skill(tmp_path)
    snapshot = _snapshot(
        recipe_runs=[
            {
                "recipe": {"id": "inspect", "skill_name": "python-backend-change"},
                "steps": [{"id": "install", "tool": "run_command", "status": "blocked", "reason": "install"}],
            }
        ],
        tool_calls=[],
    )

    result = analyze_skill_evolution(snapshot, workspace_root=tmp_path)

    assert result.proposal_payload is not None
    candidate = result.candidates[0]
    assert candidate.change_kind == "update_recovery"
    assert candidate.confidence > 0.8
    assert "evolution-recovery-run-1.yaml" in candidate.proposed_operations[0].path
    assert candidate.proposed_operations[1].action == "patch"


def test_skill_evolution_rejects_secret_like_snapshot() -> None:
    snapshot = _snapshot(
        tool_calls=[
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "read_file", "arguments": {"path": ".env"}, "result": "API_KEY=secret"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ]
    )

    result = analyze_skill_evolution(snapshot)

    assert result.proposal_payload is None
    assert result.skipped_reason == "unsafe_snapshot"


def test_skill_evolution_respects_low_confidence_and_max_proposals(tmp_path) -> None:
    _write_skill(tmp_path)
    snapshot = _snapshot(
        tool_calls=[
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "search_text", "arguments": {"query": "AgentState"}, "result": "ok"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ]
    )

    low_confidence = analyze_skill_evolution(snapshot, min_confidence=0.99, workspace_root=tmp_path)
    disabled = analyze_skill_evolution(snapshot, max_proposals=0, workspace_root=tmp_path)

    assert low_confidence.proposal_payload is None
    assert low_confidence.skipped_reason == "below_confidence_threshold"
    assert disabled.proposal_payload is None
    assert disabled.skipped_reason == "max_proposals_is_zero"


def test_skill_evolution_skips_when_metadata_patch_is_not_safe(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "workflows" / "python-backend-change"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: python-backend-change\n---\n# Skill\n", encoding="utf-8")
    snapshot = _snapshot(
        tool_calls=[
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "search_text", "arguments": {"query": "AgentState"}, "result": "ok"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ]
    )

    result = analyze_skill_evolution(snapshot, workspace_root=tmp_path)

    assert result.proposal_payload is None
    assert result.skipped_reason == "promotion_validation_failed_schema_or_metadata_patch"
    assert result.candidates
    assert result.candidates[0].proposed_operations == []


def test_skill_evolution_approved_recipe_is_discoverable(tmp_path) -> None:
    _write_skill(tmp_path)
    snapshot = _snapshot(
        tool_calls=[
            {"name": "workspace_snapshot", "arguments": {}, "result": "ok"},
            {"name": "search_text", "arguments": {"query": "AgentState"}, "result": "ok"},
            {"name": "run_pytest", "arguments": {"target": "backend/tests"}, "result": {"returncode": 0}},
        ]
    )
    result = analyze_skill_evolution(snapshot, workspace_root=tmp_path)
    assert result.proposal_payload is not None
    proposal = SkillChangeProposal(
        session_id=snapshot["session_id"],
        run_id=snapshot["run_id"],
        action=result.proposal_payload["action"],
        skill_name=result.proposal_payload["skill_name"],
        target_paths=result.proposal_payload["target_paths"],
        operations=result.proposal_payload["operations"],
    )

    applied = apply_skill_change_proposal(proposal, tmp_path)
    registry = create_default_registry(tmp_path)
    listed = registry.call("skill_recipe_list", {"skill_name": "python-backend-change"})

    assert applied.status == "applied"
    recipe_ids = {recipe["id"] for recipe in listed["result"]["recipes"]}
    assert "evolution-run-1" in recipe_ids


def _write_skill(tmp_path) -> None:
    skill_dir = tmp_path / "skills" / "workflows" / "python-backend-change"
    recipes_dir = skill_dir / "references" / "recipes"
    skill_dir.mkdir(parents=True)
    recipes_dir.mkdir(parents=True)
    metadata = {"hermes": {"recipes": [{"id": "inspect", "file": "references/recipes/inspect.yaml"}]}}
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: python-backend-change\n"
        "description: Python backend workflow.\n"
        f"metadata: {json.dumps(metadata)}\n"
        "---\n"
        "# Python Backend Change\n",
        encoding="utf-8",
    )
    (recipes_dir / "inspect.yaml").write_text(
        json.dumps(
            {
                "id": "inspect",
                "name": "Inspect",
                "steps": [{"id": "snapshot", "tool": "workspace_snapshot", "arguments": {}}],
            }
        ),
        encoding="utf-8",
    )


def _snapshot(**overrides):
    base = {
        "session_id": "session-1",
        "run_id": "run-1",
        "user_input": "/skill python-backend-change run tests",
        "selected_skills": [
            {
                "name": "python-backend-change",
                "path": "skills/workflows/python-backend-change/SKILL.md",
            }
        ],
        "recipe_runs": [],
        "tool_calls": [],
        "snapshots": {"explicit_skill_requests": ["python-backend-change"]},
        "response": "Done.",
    }
    base.update(overrides)
    return base
