from __future__ import annotations

from fastapi.testclient import TestClient

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
