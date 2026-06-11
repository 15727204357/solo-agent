from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from solo_agent.agent import AgentDeps, AgentSettings, run_agent_events
from solo_agent.providers import ChatMessage, ProviderChunk
from solo_agent.skill_changes import SkillChangeProposal, apply_skill_change_proposal
from solo_agent.skill_coverage import assert_skill_quality, audit_skill_coverage
from solo_agent.tools import create_default_registry

CORE_SCENARIOS = [
    (
        "python-backend-change",
        "/skill python-backend-change change backend behavior, inspect, run pytest and ruff",
        "inspect",
    ),
    (
        "debug-test-failure",
        "/skill debug-test-failure triage the failing pytest and verify the fix",
        "failure-triage",
    ),
    (
        "code-review",
        "/skill code-review review this change and run the review context recipe",
        "review-context",
    ),
    (
        "hash-anchored-editing",
        "/skill hash-anchored-editing prepare a hash anchored edit",
        "manual-hash-edit",
    ),
    (
        "tool-use-discipline",
        "/skill tool-use-discipline gather bounded context and run checks",
        "bounded-context-gathering",
    ),
]


class OutcomeEvalProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.seen_messages: list[list[ChatMessage]] = []

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        self.seen_messages.append(messages)
        system = messages[0].content
        if system.startswith("You are Solo Agent, a transparent"):
            yield ProviderChunk(content="1. Use the selected skill recipe\n2. Feed tool results into the response")
            return
        yield ProviderChunk(content="Completed with the collected tool results.")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.seen_messages.append(messages)
        return "记住用户偏好中文。"


class OutcomeEvalRegistry:
    def __init__(self, *, skill_path_root: str = "skills") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.skill_path_root = skill_path_root.rstrip("/")
        self.recipes = {
            "python-backend-change": "inspect",
            "debug-test-failure": "failure-triage",
            "code-review": "review-context",
            "hash-anchored-editing": "manual-hash-edit",
            "tool-use-discipline": "bounded-context-gathering",
        }

    def list_tools(self, visibility: str = "model") -> list[dict[str, str]]:
        return [
            {"name": "skills_list", "category": "skill"},
            {"name": "skill_view", "category": "skill"},
            {"name": "skill_recipe_list", "category": "skill"},
            {"name": "skill_recipe_preview", "category": "skill"},
            {"name": "skill_recipe_run", "category": "skill"},
            {"name": "workspace_snapshot", "category": "context"},
            {"name": "search_text", "category": "context"},
            {"name": "run_pytest", "category": "quality"},
            {"name": "run_ruff_check", "category": "quality"},
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "skills_list":
            return {"ok": True, "tool": name, "result": {"skills": self._skills()}}
        if name == "skill_view":
            skill_name = str(arguments["name"])
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "name": skill_name,
                    "path": self._skill_path(skill_name),
                    "content": f"# Full contract: {skill_name}\n\n## Tool Protocol\n\n- Use the declared recipe.",
                },
            }
        if name == "skill_recipe_list":
            skill_name = str(arguments["skill_name"])
            recipe_id = self.recipes[skill_name]
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "recipes": [
                        {
                            "id": recipe_id,
                            "name": recipe_id,
                            "skill_name": skill_name,
                            "description": "Deterministic outcome eval recipe.",
                            "matched": True,
                            "priority": 50,
                            "run_policy": "auto",
                        }
                    ],
                    "policy": {"execution_boundary": "safe deterministic eval"},
                },
            }
        if name == "skill_recipe_preview":
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "recipe": {"id": arguments["recipe_id"], "skill_name": arguments["skill_name"]},
                    "runnable_steps": 1,
                    "manual_steps": 0,
                    "run_policy": "auto",
                    "steps": [{"id": "snapshot", "tool": "workspace_snapshot", "auto_executable": True}],
                },
            }
        if name == "skill_recipe_run":
            return {
                "ok": True,
                "tool": name,
                "result": {
                    "run_id": "recipe_run_1",
                    "recipe": {"id": arguments["recipe_id"], "skill_name": arguments["skill_name"]},
                    "status": "completed",
                    "steps": [{"id": "snapshot", "tool": "workspace_snapshot", "status": "completed", "result": "ok"}],
                    "executed_steps": 1,
                    "blocked_steps": 0,
                },
            }
        if name == "run_pytest":
            return {"ok": True, "tool": name, "result": {"returncode": 0, "stdout": "passed"}}
        if name == "run_ruff_check":
            return {"ok": True, "tool": name, "result": {"returncode": 0, "stdout": "clean"}}
        return {"ok": True, "tool": name, "result": f"{name} ok"}

    def _skills(self) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "description": f"{name} deterministic skill.",
                "category": "workflow",
                "path": self._skill_path(name),
                "required_tools": ["workspace_snapshot", "run_pytest"],
            }
            for name in self.recipes
        ]

    def _skill_path(self, skill_name: str) -> str:
        return f"{self.skill_path_root}/workflows/{skill_name}/SKILL.md"


@pytest.mark.asyncio
@pytest.mark.parametrize(("skill_name", "user_input", "recipe_id"), CORE_SCENARIOS)
async def test_core_skill_outcome_eval_uses_progressive_disclosure_and_recipes(
    skill_name: str,
    user_input: str,
    recipe_id: str,
) -> None:
    registry = OutcomeEvalRegistry()
    provider = OutcomeEvalProvider()

    events = [
        event
        async for event in run_agent_events(
            "session-1",
            f"run-{skill_name}",
            user_input,
            deps=AgentDeps(provider=provider, tool_registry=registry),
            settings=AgentSettings(provider="ollama", model="fake-model", max_tool_calls=12, tool_call_cut_off=12),
        )
    ]

    event_types = [event.type for event in events]
    planner_prompt = provider.seen_messages[0][1].content
    responder_prompt = next(message[1].content for message in provider.seen_messages if "<tool-results>" in message[1].content)
    preview_calls = [arguments for name, arguments in registry.calls if name == "skill_recipe_preview"]
    run_calls = [arguments for name, arguments in registry.calls if name == "skill_recipe_run"]

    assert "<skills-index>" in planner_prompt
    assert "<skill-context>" in planner_prompt
    assert f"Full contract: {skill_name}" in planner_prompt
    assert "<skill-recipes>" in planner_prompt
    assert "skill_view_loaded" in event_types
    assert "skill_recipe_selected" in event_types
    assert "skill_recipe_previewed" in event_types
    assert "skill_subflow_completed" in event_types
    assert "response_completed" in event_types
    assert any(call["skill_name"] == skill_name and call["recipe_id"] == recipe_id for call in preview_calls)
    assert any(call["skill_name"] == skill_name and call["recipe_id"] == recipe_id for call in run_calls)
    assert "<tool-results>" in responder_prompt
    assert "skill_recipe_run" in responder_prompt


@pytest.mark.asyncio
async def test_outcome_eval_plain_task_keeps_full_skill_body_hidden() -> None:
    registry = OutcomeEvalRegistry()
    provider = OutcomeEvalProvider()

    events = [
        event
        async for event in run_agent_events(
            "session-1",
            "run-plain",
            "Summarize what this workspace contains.",
            deps=AgentDeps(provider=provider, tool_registry=registry),
            settings=AgentSettings(provider="ollama", model="fake-model", max_tool_calls=8, tool_call_cut_off=8),
        )
    ]

    planner_prompt = provider.seen_messages[0][1].content
    event_types = [event.type for event in events]

    assert "<skills-index>" in planner_prompt
    assert "<skill-context>" not in planner_prompt
    assert "<skill-recipes>" not in planner_prompt
    assert "skill_view_loaded" not in event_types
    assert not any(name == "skill_view" for name, _ in registry.calls)
    assert "response_completed" in event_types


@pytest.mark.asyncio
async def test_outcome_eval_approved_evolution_recipe_is_discoverable(tmp_path: Path) -> None:
    _write_single_python_skill(tmp_path)
    registry = OutcomeEvalRegistry(skill_path_root="skills")
    provider = OutcomeEvalProvider()

    events = [
        event
        async for event in run_agent_events(
            "session-1",
            "run-approval",
            "/skill python-backend-change run pytest for the backend change",
            deps=AgentDeps(provider=provider, tool_registry=registry),
            settings=AgentSettings(
                provider="ollama",
                model="fake-model",
                workspace_root=tmp_path,
                max_tool_calls=12,
                tool_call_cut_off=12,
            ),
        )
    ]

    proposal_event = next(event for event in events if event.type == "skill_evolution_proposed")
    proposal_payload = proposal_event.data["proposal"]
    recipe_path = (
        tmp_path
        / "skills"
        / "workflows"
        / "python-backend-change"
        / "references"
        / "recipes"
        / "evolution-run-approval.yaml"
    )
    assert not recipe_path.exists()

    proposal = SkillChangeProposal(
        session_id=proposal_payload["session_id"],
        run_id=proposal_payload["run_id"],
        action=proposal_payload["action"],
        skill_name=proposal_payload["skill_name"],
        target_paths=proposal_payload["target_paths"],
        operations=proposal_payload["operations"],
    )
    applied = apply_skill_change_proposal(proposal.model_copy(update={"status": "approved"}), tmp_path)

    report = assert_skill_quality(
        tmp_path,
        scenarios=[
            {
                "id": "python-backend-change",
                "expected_skill": "python-backend-change",
                "expected_recipes": ["inspect", "evolution-run-approval"],
                "required_tool_categories": ["context", "edit", "quality"],
                "verification": "approved evolution recipe is declared and valid",
            }
        ],
    )
    recipe_index = create_default_registry(tmp_path).call(
        "skill_recipe_list",
        {"skill_name": "python-backend-change", "max_entries": 20},
    )

    assert applied.status == "applied"
    assert recipe_path.exists()
    assert report.summary["error_count"] == 0
    assert not any(issue.kind == "orphan_recipe_file" for issue in audit_skill_coverage(tmp_path).issues)
    assert any(recipe["id"] == "evolution-run-approval" for recipe in recipe_index["result"]["recipes"])


def _write_single_python_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "workflows" / "python-backend-change"
    recipes_dir = skill_dir / "references" / "recipes"
    recipes_dir.mkdir(parents=True)
    metadata = {"hermes": {"recipes": [{"id": "inspect", "file": "references/recipes/inspect.yaml"}]}}
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: python-backend-change\n"
        "description: Python backend workflow.\n"
        f"required_tools: {json.dumps(['workspace_snapshot', 'prepare_edit', 'run_pytest'])}\n"
        f"metadata: {json.dumps(metadata)}\n"
        "---\n"
        "# Python Backend Change\n\n"
        "## Tool Protocol\n\n"
        "- Gather bounded context.\n"
        "- Verify with pytest.\n\n"
        "## Stop Conditions\n\n"
        "- Stop when the target change is ambiguous.\n\n"
        "## Verification\n\n"
        "- Focused pytest passes.\n",
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
