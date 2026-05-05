"""HTTP routes for the Solo Agent Web MVP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from solo_agent.settings import Settings, get_settings
from solo_agent.web.events import encode_sse
from solo_agent.web.runner import AgentRunner
from solo_agent.web.store import SessionRepository, SQLiteSessionRepository
from solo_agent.web.templates import templates

router = APIRouter()

_settings = get_settings()
_repository = SQLiteSessionRepository(_settings.database_url, memory_root=_settings.workspace_root)
_runner = AgentRunner(_repository)


class CreateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    workspace_path: str | None = Field(default=None, max_length=500)


class CreateRunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    memory_enabled: bool | None = None
    conversation_history_enabled: bool | None = None


def get_repository() -> SessionRepository:
    return _repository


def get_runner() -> AgentRunner:
    return _runner


async def parse_body(request: Request, model: type[BaseModel]) -> BaseModel:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        data = dict(form)
    return model.model_validate(data)


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HTMLResponse:
    sessions = await repo.list_sessions()
    return templates().TemplateResponse(
        request,
        "index.html",
        {
            "app_name": settings.app_name,
            "environment": settings.environment,
            "sessions": sessions,
        },
    )


@router.get("/api/health")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    return {
        "ok": True,
        "service": "solo-agent-web",
        "environment": settings.environment,
        "workspace_root": str(settings.workspace_root),
    }


@router.post("/api/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    body = await parse_body(request, CreateSessionRequest)
    title = (body.title or "New coding session").strip() or "New coding session"
    session = await repo.create_session(title=title, workspace_path=body.workspace_path)
    return session.to_public_dict()


@router.get("/api/sessions")
async def list_sessions(
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    sessions = await repo.list_sessions()
    return {"items": [session.to_public_dict() for session in sessions]}


@router.get("/api/sessions/{session_id}")
async def get_session(
    session_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    runs = await repo.list_runs(session_id)
    summary = await repo.get_summary(session_id)
    message_count = await repo.count_messages(session_id)
    return {
        **session.to_public_dict(),
        "message_count": message_count,
        "summary": summary,
        "runs": [run.to_public_dict() for run in runs],
    }


@router.get("/api/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    limit: int = 50,
) -> dict[str, object]:
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    bounded_limit = max(1, min(limit, 200))
    return {"items": await repo.list_messages(session_id, limit=bounded_limit)}


@router.post("/api/sessions/{session_id}/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    runner: Annotated[AgentRunner, Depends(get_runner)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    session = await repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    body = await parse_body(request, CreateRunRequest)
    memory_enabled = settings.memory_enabled if body.memory_enabled is None else body.memory_enabled
    conversation_history_enabled = (
        settings.conversation_history_enabled
        if body.conversation_history_enabled is None
        else body.conversation_history_enabled
    )
    run = await repo.create_run(
        session_id=session_id,
        prompt=body.prompt.strip(),
        metadata={
            "memory_enabled": memory_enabled,
            "conversation_history_enabled": conversation_history_enabled,
        },
    )
    background_tasks.add_task(runner.run, session_id, run.id)
    return {
        **run.to_public_dict(),
        "stream_url": f"/api/sessions/{session_id}/runs/{run.id}/events",
    }


@router.get("/api/sessions/{session_id}/runs/{run_id}/events")
async def stream_run_events(
    session_id: str,
    run_id: str,
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    async def event_stream() -> AsyncIterator[str]:
        async for event in repo.iter_run_events(
            session_id=session_id,
            run_id=run_id,
            heartbeat_seconds=settings.event_heartbeat_seconds,
        ):
            if await request.is_disconnected():
                break
            if event is None:
                yield ": heartbeat\n\n"
            else:
                yield encode_sse(event)
                if event.type in {"completed", "failed", "cancelled", "run_completed"}:
                    break
            await asyncio.sleep(0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
