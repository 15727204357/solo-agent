from __future__ import annotations

import asyncio

import solo_agent.web.routes as routes_module
import solo_agent.web.runner as runner_module
from fastapi.testclient import TestClient
from solo_agent.agent.events import AgentEvent
from solo_agent.settings import Settings
from solo_agent.skill_changes import SkillChangeOperation, SkillChangeProposal
from solo_agent.verified_editing import PatchEdit, PatchProposal
from solo_agent.web.app import create_app
from solo_agent.web.routes import get_repository, get_runner
from solo_agent.web.runner import AgentRunner
from solo_agent.web.store import InMemorySessionRepository


class FakeRunner:
    def __init__(self, repo: InMemorySessionRepository) -> None:
        self.repo = repo

    async def run(self, session_id: str, run_id: str) -> None:
        await self.repo.append_event(session_id, run_id, "completed", "fake done", {})
        await self.repo.mark_run_status(session_id, run_id, "completed")


def test_web_api_session_run_and_events() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        health = client.get("/api/health")
        session = client.post("/api/sessions", json={"title": "Demo"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={
                "prompt": "hello",
                "memory_enabled": False,
                "conversation_history_enabled": False,
            },
        )
        stream_url = run.json()["stream_url"]
        events = client.get(stream_url)
        messages = client.get(f"/api/sessions/{session_id}/messages")

    assert health.status_code == 200
    assert session.status_code == 201
    assert run.status_code == 202
    assert run.json()["metadata"]["memory_enabled"] is False
    assert run.json()["metadata"]["conversation_history_enabled"] is False
    assert "fake done" in events.text
    assert messages.status_code == 200
    assert messages.json()["items"][0]["run_id"] == run.json()["id"]
    assert messages.json()["items"][0]["content"] == "hello"


def test_web_api_patch_reject_cancels_run() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={"title": "Demo"}).json()["id"]
        run = client.post(f"/api/sessions/{session_id}/runs", json={"prompt": "fix bug"}).json()
        patch = _proposal(session_id, run["id"])
        asyncio.run(repo.create_patch_proposal(patch))
        listed = client.get(f"/api/sessions/{session_id}/runs/{run['id']}/patches")
        rejected = client.post(f"/api/sessions/{session_id}/runs/{run['id']}/patches/{patch.id}/reject")

    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == patch.id
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert asyncio.run(repo.get_run(session_id, run["id"])).status == "cancelled"


def test_web_api_patch_approve_applies_and_verifies(monkeypatch) -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)
    registry = FakeApprovalRegistry()
    monkeypatch.setattr(routes_module, "create_default_registry", lambda _root: registry)

    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={"title": "Demo"}).json()["id"]
        run = client.post(f"/api/sessions/{session_id}/runs", json={"prompt": "fix bug"}).json()
        patch = _proposal(session_id, run["id"])
        asyncio.run(repo.create_patch_proposal(patch))
        approved = client.post(f"/api/sessions/{session_id}/runs/{run['id']}/patches/{patch.id}/approve")

    assert approved.status_code == 200
    assert approved.json()["ok"] is True
    assert approved.json()["verification"]["pytest"]["returncode"] == 0
    assert [name for name, _ in registry.calls] == ["apply_text_edit", "run_pytest", "run_ruff_check"]
    assert asyncio.run(repo.get_run(session_id, run["id"])).status == "completed"


def test_web_api_skill_change_approve_applies_to_workspace(tmp_path) -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)
    app.dependency_overrides[routes_module.get_settings] = lambda: Settings(workspace_root=tmp_path)

    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={"title": "Demo"}).json()["id"]
        run = client.post(f"/api/sessions/{session_id}/runs", json={"prompt": "save workflow"}).json()
        proposal = _skill_change(session_id, run["id"])
        asyncio.run(repo.create_skill_change_proposal(proposal))
        listed = client.get(f"/api/sessions/{session_id}/runs/{run['id']}/skill-changes")
        approved = client.post(f"/api/sessions/{session_id}/runs/{run['id']}/skill-changes/{proposal.id}/approve")

    assert listed.status_code == 200
    assert listed.json()["items"][0]["id"] == proposal.id
    assert approved.status_code == 200
    assert approved.json()["ok"] is True
    assert (tmp_path / "skills" / "workflows" / "new-flow" / "SKILL.md").read_text(encoding="utf-8").startswith("---")
    assert asyncio.run(repo.get_run(session_id, run["id"])).status == "completed"


def test_web_api_cancel_running_run_marks_cancelled_and_stream_closes() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    async def setup() -> tuple[str, str]:
        session = await repo.create_session("Cancel", None)
        run = await repo.create_run(session.id, "long task")
        await repo.mark_run_status(session.id, run.id, "running")
        return session.id, run.id

    session_id, run_id = asyncio.run(setup())
    with TestClient(app) as client:
        cancelled = client.post(f"/api/sessions/{session_id}/runs/{run_id}/cancel")
        events = client.get(f"/api/sessions/{session_id}/runs/{run_id}/events")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert "Run cancelled by user" in events.text
    assert asyncio.run(repo.get_run(session_id, run_id)).status == "cancelled"


def test_web_api_cancel_completed_run_is_idempotent() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    async def setup() -> tuple[str, str]:
        session = await repo.create_session("Cancel", None)
        run = await repo.create_run(session.id, "done task")
        await repo.mark_run_status(session.id, run.id, "completed")
        return session.id, run.id

    session_id, run_id = asyncio.run(setup())
    with TestClient(app) as client:
        cancelled = client.post(f"/api/sessions/{session_id}/runs/{run_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "completed"


def test_web_api_interrupt_marks_paused_or_awaiting_feedback() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    async def setup() -> tuple[str, str]:
        session = await repo.create_session("Interrupt", None)
        run = await repo.create_run(session.id, "long team task")
        await repo.mark_run_status(session.id, run.id, "running")
        return session.id, run.id

    session_id, run_id = asyncio.run(setup())
    with TestClient(app) as client:
        paused = client.post(f"/api/sessions/{session_id}/runs/{run_id}/interrupt", json={"reason": "pause"})
        await_feedback = client.post(
            f"/api/sessions/{session_id}/runs/{run_id}/interrupt",
            json={"feedback": {"message": "retry from tests"}},
        )

    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert await_feedback.status_code == 200
    assert await_feedback.json()["status"] == "awaiting_feedback"
    events = asyncio.run(repo.list_run_events(session_id, run_id))
    assert [event.type for event in events].count("run_interrupted") == 2


def test_web_api_artifacts_and_resume_use_checkpoint_state(monkeypatch, tmp_path) -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    captured: dict[str, object] = {}
    settings = Settings(workspace_root=tmp_path, provider="ollama", model="fake-model")
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)
    app.dependency_overrides[routes_module.get_settings] = lambda: settings

    async def fake_run_agent_events(
        session_id,
        run_id,
        user_input,
        deps=None,
        settings=None,
        initial_state=None,
        resume_from_node=None,
    ):
        captured["initial_state"] = initial_state
        captured["resume_from_node"] = resume_from_node
        captured["settings"] = settings
        captured["registry_root"] = getattr(deps.tool_registry, "command_workspace_root", None)
        yield AgentEvent(type="run_completed", session_id=session_id, run_id=run_id, node="workflow", data={})

    monkeypatch.setattr(routes_module, "run_agent_events", fake_run_agent_events)

    async def setup() -> tuple[str, str]:
        session = await repo.create_session("Resume", None)
        run = await repo.create_run(session.id, "fix code", metadata={"run_mode": "agent"})
        state_snapshot = {
            "user_input": "fix code",
            "sandbox_artifacts": {
                "sandbox_root": str(tmp_path / ".tmp" / "runs" / "run-1"),
                "diff": "--- a/pkg/app.py\n+++ b/pkg/app.py\n",
                "tool_ledger": [{"tool": "apply_text_edit", "ok": True}],
            },
            "code_map_summary": {"python_file_count": 1},
            "impact_analysis": {"related_tests": ["tests/test_app.py"]},
        }
        await repo.append_event(
            session.id,
            run.id,
            "persist_snapshot_completed",
            "snapshot",
            {"data": {"snapshot": {"loop_stage": "team_test", "state_snapshot": state_snapshot}}},
        )
        await repo.mark_run_status(session.id, run.id, "paused")
        return session.id, run.id

    session_id, run_id = asyncio.run(setup())
    with TestClient(app) as client:
        artifacts = client.get(f"/api/sessions/{session_id}/runs/{run_id}/artifacts")
        resumed = client.post(
            f"/api/sessions/{session_id}/runs/{run_id}/resume",
            json={
                "checkpoint_id": "event:1",
                "from_node": "team_test",
                "human_feedback": {"message": "pytest failed; fix assertion"},
            },
        )

    assert artifacts.status_code == 200
    assert artifacts.json()["sandbox_artifacts"]["tool_ledger"][0]["tool"] == "apply_text_edit"
    assert artifacts.json()["impact_analysis"]["related_tests"] == ["tests/test_app.py"]
    assert resumed.status_code == 200
    assert captured["resume_from_node"] == "team_test"
    assert captured["initial_state"]["sandbox_artifacts"]["diff"].startswith("--- a/pkg/app.py")
    assert captured["settings"].human_feedback == {"message": "pytest failed; fix assertion"}


def test_web_api_memory_inbox_approve_reject_and_revoke() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)
    now = "2026-01-01T00:00:00+00:00"
    repo._memory_candidates["cand_1"] = {
        "id": "cand_1",
        "target": "user",
        "content": "User prefers concise responses",
        "confidence": 0.8,
        "status": "pending",
        "duplicate_of_id": None,
        "conflict_ids": [],
        "safety_flags": [],
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "decided_at": None,
    }
    repo._memory_entries["mem_old"] = {
        "id": "mem_old",
        "target": "user",
        "content": "User prefers verbose responses",
        "source_candidate_id": None,
        "confidence": 1.0,
        "supersedes_id": None,
        "status": "active",
        "metadata": {},
        "created_at": now,
        "updated_at": now,
        "revoked_at": None,
    }
    repo._memory_candidates["cand_2"] = {
        **repo._memory_candidates["cand_1"],
        "id": "cand_2",
        "content": "User prefers brief responses instead",
        "conflict_ids": ["mem_old"],
    }

    with TestClient(app) as client:
        inbox = client.get("/api/memory/inbox")
        conflict = client.post("/api/memory/candidates/cand_2/approve", json={"resolution": "add"})
        approved = client.post("/api/memory/candidates/cand_2/approve", json={"resolution": "replace"})
        rejected = client.post("/api/memory/candidates/cand_1/reject", json={"reason": "no"})
        entries = client.get("/api/memory/entries?include_inactive=true")
        revoked = client.post(f"/api/memory/entries/{approved.json()['entry']['id']}/revoke", json={"reason": "old"})

    assert inbox.status_code == 200
    assert len(inbox.json()["items"]) == 2
    assert conflict.status_code == 409
    assert approved.status_code == 200
    assert approved.json()["entry"]["status"] == "active"
    assert repo._memory_entries["mem_old"]["status"] == "superseded"
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert entries.status_code == 200
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"


class FakeApprovalRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if name == "apply_text_edit":
            return {"ok": True, "result": {"changed": True, "path": arguments["path"]}}
        if name == "run_pytest":
            return {"ok": True, "result": {"returncode": 0, "output": "pytest ok"}}
        if name == "run_ruff_check":
            return {"ok": True, "result": {"returncode": 0, "output": "ruff ok"}}
        raise AssertionError(name)


def _proposal(session_id: str, run_id: str) -> PatchProposal:
    return PatchProposal(
        id="patch_test",
        session_id=session_id,
        run_id=run_id,
        status="pending",
        summary="Fix demo",
        diff="--- app.py\n+++ app.py\n@@\n-old\n+new",
        edits=[
            PatchEdit(
                path="app.py",
                expected_hash="hash-1",
                old_text="old",
                new_text="new",
                diff="--- app.py\n+++ app.py\n@@\n-old\n+new",
            )
        ],
    )


def _skill_change(session_id: str, run_id: str) -> SkillChangeProposal:
    return SkillChangeProposal(
        id="skillchg_test",
        session_id=session_id,
        run_id=run_id,
        action="create",
        skill_name="new-flow",
        target_paths=["workflows/new-flow/SKILL.md"],
        diff="--- /dev/null\n+++ b/skills/workflows/new-flow/SKILL.md\n",
        operations=[
            SkillChangeOperation(
                action="create",
                path="workflows/new-flow/SKILL.md",
                content="---\nname: new-flow\n---\n# New Flow\n",
            )
        ],
    )


def test_web_api_run_mode_defaults_to_agent() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Mode Test"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "hello"},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "agent"
    assert run.json()["metadata"]["subagent_policy"] == "off"
    assert run.json()["metadata"]["subagent_enabled"] is False


def test_web_api_run_mode_explicit_plan() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Plan Mode Test"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "design a feature", "run_mode": "plan"},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "plan"
    assert run.json()["metadata"]["subagent_policy"] == "auto"
    assert run.json()["metadata"]["subagent_enabled"] is True


def test_web_api_run_mode_explicit_agent() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Agent Mode Test"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "fix bug", "run_mode": "agent"},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "agent"
    assert run.json()["metadata"]["subagent_policy"] == "off"
    assert run.json()["metadata"]["subagent_enabled"] is False


def test_web_api_create_run_accepts_legacy_subagent_enabled_true_as_auto_policy() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Subagent Toggle"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "inspect independent tasks", "run_mode": "agent", "subagent_enabled": True},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "agent"
    assert run.json()["metadata"]["subagent_policy"] == "auto"
    assert run.json()["metadata"]["subagent_enabled"] is False


def test_web_api_create_run_accepts_legacy_subagent_enabled_false_as_off_policy() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Subagent Off"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "inspect independent tasks", "run_mode": "plan", "subagent_enabled": False},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "plan"
    assert run.json()["metadata"]["subagent_policy"] == "off"
    assert run.json()["metadata"]["subagent_enabled"] is False


def test_web_api_create_run_plan_subagent_policy_off_disables_task_exposure() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Subagent Default"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "hello", "run_mode": "plan", "subagent_policy": "off"},
        )

    assert run.status_code == 202
    assert run.json()["metadata"]["run_mode"] == "plan"
    assert run.json()["metadata"]["subagent_policy"] == "off"
    assert run.json()["metadata"]["subagent_enabled"] is False


def test_web_api_run_mode_invalid_rejected() -> None:
    app = create_app()
    repo = InMemorySessionRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    app.dependency_overrides[get_runner] = lambda: FakeRunner(repo)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={"title": "Invalid Mode"})
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"prompt": "something", "run_mode": "invalid"},
        )

    assert run.status_code == 422


def test_agent_runner_passes_workflow_settings(monkeypatch, tmp_path) -> None:
    repo = InMemorySessionRepository()
    captured: dict[str, object] = {}
    settings = Settings(
        workspace_root=tmp_path,
        subagent_policy="auto",
        subagent_enabled=False,
        max_concurrent_subagents=4,
        subagent_timeout_seconds=120,
        sandbox_mode="local",
        workflow_runtime_root=".tmp/runs",
        provider="ollama",
        model="fake-model",
    )

    class FakeProvider:
        name = "fake"
        model = "fake"

    async def fake_run_agent_events(session_id, run_id, user_input, deps=None, settings=None):
        captured["settings"] = settings
        captured["deps_settings"] = deps.settings
        yield AgentEvent(
            type="run_completed",
            session_id=session_id,
            run_id=run_id,
            node="workflow",
            data={},
        )

    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    def fake_create_registry(_root, **kwargs):
        captured["registry_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(runner_module, "create_default_registry", fake_create_registry)
    monkeypatch.setattr(runner_module, "create_provider_from_settings", lambda agent_settings: FakeProvider())
    monkeypatch.setattr(runner_module, "run_agent_events", fake_run_agent_events)

    async def run() -> None:
        session = await repo.create_session("Runner", None)
        created = await repo.create_run(session.id, "hello", metadata={"run_mode": "agent"})
        await AgentRunner(repo).run(session.id, created.id)

    asyncio.run(run())

    agent_settings = captured["settings"]
    assert captured["deps_settings"] is agent_settings
    assert agent_settings.subagent_policy == "off"
    assert agent_settings.subagent_enabled is False
    assert agent_settings.max_concurrent_subagents == 4
    assert agent_settings.subagent_timeout_seconds == 120
    assert agent_settings.sandbox_mode == "local"
    assert agent_settings.workflow_runtime_root == ".tmp/runs"
    assert captured["registry_kwargs"] == {
        "is_plan_mode": False,
        "subagent_enabled": False,
        "codeintel_max_files": 2_000,
        "codeintel_max_file_bytes": 512_000,
        "codeintel_index_ttl_seconds": 30,
    }


def test_agent_runner_retains_sandbox_when_feedback_pauses_run(monkeypatch, tmp_path) -> None:
    repo = InMemorySessionRepository()
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    settings = Settings(
        workspace_root=tmp_path,
        sandbox_mode="isolated",
        workflow_runtime_root=".tmp/runs",
        provider="ollama",
        model="fake-model",
    )

    class FakeProvider:
        name = "fake"
        model = "fake"

    async def fake_run_agent_events(session_id, run_id, user_input, deps=None, settings=None):
        await repo.mark_run_status(session_id, run_id, "awaiting_feedback")
        yield AgentEvent(type="team_tester_completed", session_id=session_id, run_id=run_id, node="team_test", data={})

    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_module, "create_provider_from_settings", lambda agent_settings: FakeProvider())
    monkeypatch.setattr(runner_module, "run_agent_events", fake_run_agent_events)

    async def run() -> tuple[str, str]:
        session = await repo.create_session("Runner", None)
        created = await repo.create_run(session.id, "fix app.py", metadata={"run_mode": "plan"})
        await AgentRunner(repo).run(session.id, created.id)
        return session.id, created.id

    session_id, run_id = asyncio.run(run())
    events = asyncio.run(repo.list_run_events(session_id, run_id))

    assert asyncio.run(repo.get_run(session_id, run_id)).status == "awaiting_feedback"
    assert any(event.type == "sandbox_created" for event in events)
    assert any(event.type == "sandbox_retained" for event in events)


def test_agent_runner_exposes_task_for_plan_auto(monkeypatch, tmp_path) -> None:
    repo = InMemorySessionRepository()
    captured: dict[str, object] = {}
    settings = Settings(workspace_root=tmp_path, provider="ollama", model="fake-model")

    class FakeProvider:
        name = "fake"
        model = "fake"

    async def fake_run_agent_events(session_id, run_id, user_input, deps=None, settings=None):
        captured["settings"] = settings
        yield AgentEvent(type="run_completed", session_id=session_id, run_id=run_id, node="workflow", data={})

    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    def fake_create_registry(_root, **kwargs):
        captured["registry_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(runner_module, "create_default_registry", fake_create_registry)
    monkeypatch.setattr(runner_module, "create_provider_from_settings", lambda agent_settings: FakeProvider())
    monkeypatch.setattr(runner_module, "run_agent_events", fake_run_agent_events)

    async def run() -> None:
        session = await repo.create_session("Runner", None)
        created = await repo.create_run(
            session.id,
            "hello",
            metadata={"run_mode": "plan", "subagent_policy": "auto", "subagent_enabled": True},
        )
        await AgentRunner(repo).run(session.id, created.id)

    asyncio.run(run())

    agent_settings = captured["settings"]
    assert agent_settings.run_mode == "plan"
    assert agent_settings.subagent_policy == "auto"
    assert agent_settings.subagent_enabled is True
    assert captured["registry_kwargs"]["is_plan_mode"] is True
    assert captured["registry_kwargs"]["subagent_enabled"] is True
    assert captured["registry_kwargs"]["command_workspace_root"]
    assert captured["registry_kwargs"]["sandbox_network_policy"] == "deny"


def test_agent_runner_hides_task_for_plan_off(monkeypatch, tmp_path) -> None:
    repo = InMemorySessionRepository()
    captured: dict[str, object] = {}
    settings = Settings(workspace_root=tmp_path, provider="ollama", model="fake-model")

    class FakeProvider:
        name = "fake"
        model = "fake"

    async def fake_run_agent_events(session_id, run_id, user_input, deps=None, settings=None):
        captured["settings"] = settings
        yield AgentEvent(type="run_completed", session_id=session_id, run_id=run_id, node="workflow", data={})

    monkeypatch.setattr(runner_module, "get_settings", lambda: settings)
    def fake_create_registry(_root, **kwargs):
        captured["registry_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(runner_module, "create_default_registry", fake_create_registry)
    monkeypatch.setattr(runner_module, "create_provider_from_settings", lambda agent_settings: FakeProvider())
    monkeypatch.setattr(runner_module, "run_agent_events", fake_run_agent_events)

    async def run() -> None:
        session = await repo.create_session("Runner", None)
        created = await repo.create_run(
            session.id,
            "hello",
            metadata={"run_mode": "plan", "subagent_policy": "off", "subagent_enabled": False},
        )
        await AgentRunner(repo).run(session.id, created.id)

    asyncio.run(run())

    agent_settings = captured["settings"]
    assert agent_settings.run_mode == "plan"
    assert agent_settings.subagent_policy == "off"
    assert agent_settings.subagent_enabled is False
    assert captured["registry_kwargs"]["is_plan_mode"] is True
    assert captured["registry_kwargs"]["subagent_enabled"] is False
    assert captured["registry_kwargs"]["command_workspace_root"]
    assert captured["registry_kwargs"]["sandbox_network_policy"] == "deny"
