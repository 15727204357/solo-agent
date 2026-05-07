from __future__ import annotations

import asyncio

import solo_agent.web.routes as routes_module
from fastapi.testclient import TestClient
from solo_agent.verified_editing import PatchEdit, PatchProposal
from solo_agent.web.app import create_app
from solo_agent.web.routes import get_repository, get_runner
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
