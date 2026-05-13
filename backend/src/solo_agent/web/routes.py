"""HTTP routes for the Solo Agent Web MVP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from solo_agent.agent.graph import run_agent_events
from solo_agent.memory import MemoryGovernanceError
from solo_agent.settings import Settings, get_settings
from solo_agent.tools import create_default_registry
from solo_agent.verified_editing import apply_approved_patch
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
    run_mode: Literal["agent", "plan"] | None = None
    subagent_enabled: bool | None = None


class UpdateMemoryCandidateRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=2200)
    target: str | None = Field(default=None, pattern="^(memory|user|skill)$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, object] | None = None


class ApproveMemoryCandidateRequest(BaseModel):
    resolution: str = Field(default="add", pattern="^(add|replace|merge)$")
    content: str | None = Field(default=None, min_length=1, max_length=8000)


class MemoryDecisionRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


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
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


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


@router.get("/api/memory/inbox")
async def list_memory_inbox(
    repo: Annotated[SessionRepository, Depends(get_repository)],
    candidate_status: str | None = Query(default=None, alias="status"),
    target: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    bounded_limit = max(1, min(limit, 200))
    status_value = candidate_status or "pending"
    return {
        "items": await repo.list_memory_candidates(
            status=status_value,
            target=target,
            limit=bounded_limit,
        )
    }


@router.patch("/api/memory/candidates/{candidate_id}")
async def update_memory_candidate(
    candidate_id: str,
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    body = await parse_body(request, UpdateMemoryCandidateRequest)
    try:
        candidate = await repo.update_memory_candidate(
            candidate_id,
            content=body.content,
            target=body.target,
            confidence=body.confidence,
            metadata=body.metadata,
        )
    except MemoryGovernanceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory candidate not found")
    return candidate


@router.post("/api/memory/candidates/{candidate_id}/approve")
async def approve_memory_candidate(
    candidate_id: str,
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    body = await parse_body(request, ApproveMemoryCandidateRequest)
    try:
        return await repo.approve_memory_candidate(
            candidate_id,
            resolution=body.resolution,
            content=body.content,
        )
    except MemoryGovernanceError as exc:
        detail = str(exc)
        code = status.HTTP_409_CONFLICT if detail != "Memory candidate not found" else status.HTTP_404_NOT_FOUND
        raise HTTPException(status_code=code, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/api/memory/candidates/{candidate_id}/reject")
async def reject_memory_candidate(
    candidate_id: str,
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    body = await parse_body(request, MemoryDecisionRequest)
    candidate = await repo.reject_memory_candidate(candidate_id, reason=body.reason)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory candidate not found")
    return candidate


@router.get("/api/memory/entries")
async def list_memory_entries(
    repo: Annotated[SessionRepository, Depends(get_repository)],
    target: str | None = None,
    include_inactive: bool = False,
    limit: int = 100,
) -> dict[str, object]:
    bounded_limit = max(1, min(limit, 200))
    return {
        "items": await repo.list_memory_entries(
            target=target,
            include_inactive=include_inactive,
            limit=bounded_limit,
        )
    }


@router.post("/api/memory/entries/{entry_id}/revoke")
async def revoke_memory_entry(
    entry_id: str,
    request: Request,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    body = await parse_body(request, MemoryDecisionRequest)
    entry = await repo.revoke_memory_entry(entry_id, reason=body.reason)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory entry not found")
    return entry


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
            "run_mode": body.run_mode or "agent",
            "subagent_enabled": (
                settings.subagent_enabled
                if body.subagent_enabled is None
                else body.subagent_enabled
            ),
        },
    )
    background_tasks.add_task(runner.run, session_id, run.id)
    return {
        **run.to_public_dict(),
        "stream_url": f"/api/sessions/{session_id}/runs/{run.id}/events",
    }


@router.get("/api/sessions/{session_id}/runs/{run_id}/patches")
async def list_run_patches(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    patches = await repo.list_patch_proposals(session_id, run_id)
    return {"items": [patch.to_public_dict() for patch in patches]}


@router.get("/api/sessions/{session_id}/runs/{run_id}/patches/{patch_id}")
async def get_run_patch(
    session_id: str,
    run_id: str,
    patch_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    patch = await repo.get_patch_proposal(session_id, run_id, patch_id)
    if patch is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patch proposal not found")
    return patch.to_public_dict()


@router.post("/api/sessions/{session_id}/runs/{run_id}/patches/{patch_id}/reject")
async def reject_run_patch(
    session_id: str,
    run_id: str,
    patch_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    patch = await repo.get_patch_proposal(session_id, run_id, patch_id)
    if patch is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patch proposal not found")
    if patch.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Patch proposal is not pending")

    updated = await repo.update_patch_proposal(
        session_id,
        run_id,
        patch_id,
        status="rejected",
        error="rejected by user",
        decided=True,
    )
    await repo.append_event(
        session_id,
        run_id,
        "patch_rejected",
        "Patch proposal rejected by user.",
        {"patch_id": patch_id},
    )
    await repo.mark_run_status(session_id, run_id, "cancelled")
    return (updated or patch).to_public_dict()


@router.post("/api/sessions/{session_id}/runs/{run_id}/patches/{patch_id}/approve")
async def approve_run_patch(
    session_id: str,
    run_id: str,
    patch_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    patch = await repo.get_patch_proposal(session_id, run_id, patch_id)
    if patch is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patch proposal not found")
    if patch.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Patch proposal is not pending")

    await repo.update_patch_proposal(session_id, run_id, patch_id, status="approved", decided=True)
    await repo.append_event(
        session_id,
        run_id,
        "patch_apply_started",
        "Applying approved patch proposal.",
        {"patch_id": patch_id},
    )
    registry = create_default_registry(settings.workspace_root)

    async def call_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        return registry.call(name, arguments)

    applied = await apply_approved_patch(patch.model_copy(update={"status": "approved"}), call_tool=call_tool)
    updated = await repo.update_patch_proposal(
        session_id,
        run_id,
        patch_id,
        status=applied.status,
        apply_results=[dict(item) for item in applied.apply_results],
        verification=applied.verification.model_dump(mode="json") if applied.verification else None,
        error=applied.error,
    )
    if applied.apply_results and applied.status != "failed":
        await repo.append_event(
            session_id,
            run_id,
            "patch_applied",
            "Patch proposal applied to the workspace.",
            {"patch_id": patch_id, "apply_results": applied.apply_results},
        )
    if applied.verification is not None:
        await repo.append_event(
            session_id,
            run_id,
            "verification_started",
            "Running pytest and ruff verification.",
            {"patch_id": patch_id},
        )
        await repo.append_event(
            session_id,
            run_id,
            "verification_completed",
            "Patch verification completed.",
            {"patch_id": patch_id, "verification": applied.verification.model_dump(mode="json")},
        )
    await repo.mark_run_status(
        session_id,
        run_id,
        "completed" if applied.status == "applied" else "failed",
    )
    public = (updated or applied).to_public_dict()
    public["ok"] = applied.status == "applied"
    return public

@router.get("/api/sessions/{session_id}/runs/{run_id}/events/history")
async def list_run_event_history(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    limit: int = 300,
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    bounded_limit = max(1, min(limit, 1000))
    events = await repo.list_run_events(session_id, run_id, limit=bounded_limit)

    return {
        "items": [event.to_dict() for event in events],
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
                if event.type in {"completed", "failed", "cancelled", "run_completed", "patch_approval_required"}:
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


# ---------------------------------------------------------------------------
# Checkpoint & Replay API
# ---------------------------------------------------------------------------


@router.get("/api/sessions/{session_id}/runs/{run_id}/checkpoints")
async def list_run_checkpoints(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> list[dict[str, object]]:
    checkpoints = await repo.list_checkpoints(session_id, run_id)
    return [dict(c) for c in (checkpoints or [])]


@router.get("/api/sessions/{session_id}/runs/{run_id}/graph")
async def get_run_graph_state(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    checkpoint_id: str | None = Query(None),
) -> dict[str, object]:
    state = await repo.get_graph_snapshot(session_id, run_id, checkpoint_id=checkpoint_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Graph state not found")
    return dict(state)


class ResumeRunRequest(BaseModel):
    approval: Literal["approved", "rejected"] | None = None
    recovery_hints: dict[str, object] | None = None


@router.post("/api/sessions/{session_id}/runs/{run_id}/resume")
async def resume_run(
    session_id: str,
    run_id: str,
    body: ResumeRunRequest,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    request: Request,
) -> StreamingResponse:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if body.approval == "rejected":
        await repo.mark_run_status(session_id, run_id, "cancelled")
        return StreamingResponse(
            iter([encode_sse({"type": "cancelled", "data": {"reason": "Rejected by user"}})]),
            media_type="text/event-stream",
        )

    async def event_stream() -> AsyncIterator[str]:
        prompt = run.prompt if hasattr(run, "prompt") else ""
        async for event in run_agent_events(
            session_id=session_id,
            run_id=run_id,
            user_input=prompt if isinstance(prompt, str) else "",
        ):
            if await request.is_disconnected():
                break
            yield encode_sse(event)
            if event.type in {"completed", "failed", "cancelled", "run_completed"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class ReplayRunRequest(BaseModel):
    checkpoint_id: str | None = None
    from_node: str | None = None
    dry_run: bool = True


@router.post("/api/sessions/{session_id}/runs/{run_id}/replay")
async def replay_run(
    session_id: str,
    run_id: str,
    body: ReplayRunRequest,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    request: Request,
) -> StreamingResponse:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    async def event_stream() -> AsyncIterator[str]:
        prompt = run.prompt if hasattr(run, "prompt") else ""
        async for event in run_agent_events(
            session_id=session_id,
            run_id=f"{run_id}_replay",
            user_input=prompt if isinstance(prompt, str) else "",
        ):
            if await request.is_disconnected():
                break
            yield encode_sse(event)
            if event.type in {"completed", "failed", "cancelled", "run_completed"}:
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
