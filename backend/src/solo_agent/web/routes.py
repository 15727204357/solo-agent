"""HTTP routes for the Solo Agent Web MVP."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from solo_agent.agent import AgentDeps, AgentSettings
from solo_agent.agent.graph import run_agent_events
from solo_agent.memory import MemoryGovernanceError
from solo_agent.settings import Settings, get_settings
from solo_agent.skill_changes import apply_skill_change_proposal
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
    tool_loop_mode: Literal["heuristic", "model"] | None = None
    intent_router_mode: Literal["rules", "shadow_hybrid", "hybrid"] | None = None
    approval_mode: Literal["confirm", "manual_only"] | None = None
    workspace_backend: Literal["local", "copy", "docker"] | None = None
    eval_suite_id: str | None = Field(default=None, max_length=120)
    subagent_policy: Literal["off", "auto"] | None = None
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
    run_mode = body.run_mode or "agent"
    tool_loop_mode = body.tool_loop_mode or settings.tool_loop_mode
    intent_router_mode = body.intent_router_mode or settings.intent_router_mode
    approval_mode = body.approval_mode or settings.approval_mode
    workspace_backend = body.workspace_backend or settings.workspace_backend
    subagent_policy = _resolve_subagent_policy(
        run_mode=run_mode,
        requested_policy=body.subagent_policy,
        legacy_enabled=body.subagent_enabled,
        settings=settings,
    )
    resolved_subagent_enabled = run_mode == "plan" and subagent_policy == "auto"

    run = await repo.create_run(
        session_id=session_id,
        prompt=body.prompt.strip(),
        metadata={
            "memory_enabled": memory_enabled,
            "conversation_history_enabled": conversation_history_enabled,
            "run_mode": run_mode,
            "tool_loop_mode": tool_loop_mode,
            "intent_router_mode": intent_router_mode,
            "approval_mode": approval_mode,
            "workspace_backend": workspace_backend,
            "eval_suite_id": body.eval_suite_id or settings.eval_suite_id,
            "subagent_policy": subagent_policy,
            "subagent_enabled": resolved_subagent_enabled,
        },
    )
    background_tasks.add_task(runner.run, session_id, run.id)
    return {
        **run.to_public_dict(),
        "stream_url": f"/api/sessions/{session_id}/runs/{run.id}/events",
    }


@router.post("/api/sessions/{session_id}/runs/{run_id}/cancel")
async def cancel_run(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.status in {"completed", "failed", "cancelled"}:
        return run.to_public_dict()

    await repo.append_event(
        session_id,
        run_id,
        "cancelled",
        "Run cancelled by user",
        {"data": {"reason": "user_cancelled"}},
    )
    await repo.mark_run_status(session_id, run_id, "cancelled")
    updated = await repo.get_run(session_id, run_id)
    return (updated or run).to_public_dict()


def _resolve_subagent_policy(
    *,
    run_mode: str,
    requested_policy: str | None,
    legacy_enabled: bool | None,
    settings: Settings,
) -> str:
    if requested_policy in {"off", "auto"}:
        return requested_policy
    if legacy_enabled is True:
        return "auto"
    if legacy_enabled is False:
        return "off"
    if run_mode == "plan":
        return settings.subagent_policy
    return "off"


def _state_snapshot_from_graph_payload(snapshot: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(snapshot, dict):
        return {}
    data = snapshot.get("data")
    if not isinstance(data, dict):
        return {}
    state = data.get("state_snapshot")
    if isinstance(state, dict):
        return dict(state)
    nested = data.get("snapshot")
    if isinstance(nested, dict) and isinstance(nested.get("state_snapshot"), dict):
        return dict(nested["state_snapshot"])
    return {}


def _sandbox_root_from_state(state: dict[str, object]) -> str | None:
    artifacts = state.get("sandbox_artifacts")
    if isinstance(artifacts, dict) and artifacts.get("sandbox_root"):
        return str(artifacts["sandbox_root"])
    snapshots = state.get("snapshots")
    if isinstance(snapshots, dict):
        nested = snapshots.get("sandbox_artifacts")
        if isinstance(nested, dict) and nested.get("sandbox_root"):
            return str(nested["sandbox_root"])
    return None


def _agent_settings_from_run(
    settings: Settings,
    run: object,
    *,
    resume_from_node: str | None = None,
    recovery_hints: dict[str, object] | None = None,
    human_feedback: dict[str, object] | None = None,
) -> AgentSettings:
    metadata = dict(getattr(run, "metadata", {}) or {})
    run_mode = str(metadata.get("run_mode", "agent"))
    tool_loop_mode = str(metadata.get("tool_loop_mode", settings.tool_loop_mode))
    intent_router_mode = str(metadata.get("intent_router_mode", settings.intent_router_mode))
    approval_mode = str(metadata.get("approval_mode", settings.approval_mode))
    workspace_backend = str(metadata.get("workspace_backend", settings.workspace_backend))
    return AgentSettings(
        provider=settings.provider,
        workspace_root=settings.workspace_root,
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout_seconds=settings.timeout_seconds,
        temperature=settings.temperature,
        plan_max_tokens=settings.plan_max_tokens,
        response_max_tokens=settings.response_max_tokens,
        max_tool_calls=settings.max_tool_calls,
        tool_call_cut_off=settings.tool_call_cut_off,
        tool_output_max_bytes=settings.tool_output_max_bytes,
        context_file_limit=settings.context_file_limit,
        context_search_limit=settings.context_search_limit,
        memory_enabled=bool(metadata.get("memory_enabled", settings.memory_enabled)),
        conversation_history_enabled=bool(
            metadata.get("conversation_history_enabled", settings.conversation_history_enabled)
        ),
        verified_editing_enabled=settings.verified_editing_enabled,
        patch_max_tokens=settings.patch_max_tokens,
        run_mode=run_mode,
        tool_loop_mode=tool_loop_mode if tool_loop_mode in {"heuristic", "model"} else settings.tool_loop_mode,
        intent_router_mode=(
            intent_router_mode
            if intent_router_mode in {"rules", "shadow_hybrid", "hybrid"}
            else settings.intent_router_mode
        ),
        intent_router_max_epochs=settings.intent_router_max_epochs,
        intent_router_model_timeout_seconds=settings.intent_router_model_timeout_seconds,
        approval_mode=approval_mode if approval_mode in {"confirm", "manual_only"} else settings.approval_mode,
        workspace_backend=workspace_backend if workspace_backend in {"local", "copy", "docker"} else settings.workspace_backend,
        eval_suite_id=str(metadata.get("eval_suite_id") or settings.eval_suite_id or "") or None,
        is_plan_mode=run_mode == "plan",
        subagent_policy=str(metadata.get("subagent_policy", settings.subagent_policy)),
        subagent_enabled=bool(metadata.get("subagent_enabled", False)),
        sandbox_mode=settings.sandbox_mode,
        sandbox_retain_on_failure=settings.sandbox_retain_on_failure,
        sandbox_network_policy=settings.sandbox_network_policy,
        sandbox_command_timeout_seconds=settings.sandbox_command_timeout_seconds,
        sandbox_max_output_bytes=settings.sandbox_max_output_bytes,
        sandbox_max_commands_per_run=settings.sandbox_max_commands_per_run,
        sandbox_max_changed_files=settings.sandbox_max_changed_files,
        sandbox_max_workspace_bytes=settings.sandbox_max_workspace_bytes,
        codeintel_max_files=settings.codeintel_max_files,
        codeintel_max_file_bytes=settings.codeintel_max_file_bytes,
        codeintel_index_ttl_seconds=settings.codeintel_index_ttl_seconds,
        outcome_judge_enabled=settings.outcome_judge_enabled,
        outcome_judge_provider_mode=settings.outcome_judge_provider_mode,
        eval_runtime_root=settings.eval_runtime_root,
        git_artifacts_enabled=settings.git_artifacts_enabled,
        resume_from_node=resume_from_node,
        recovery_hints=dict(recovery_hints or {}),
        human_feedback=dict(human_feedback or {}),
    )


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
        verification_plan=applied.verification_plan.model_dump(mode="json"),
        verification=applied.verification.model_dump(mode="json") if applied.verification else None,
        stop_gate=applied.stop_gate.model_dump(mode="json"),
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
            "Running planned patch verification.",
            {"patch_id": patch_id, "verification_plan": applied.verification_plan.model_dump(mode="json")},
        )
        await repo.append_event(
            session_id,
            run_id,
            "verification_completed",
            "Patch verification completed.",
            {
                "patch_id": patch_id,
                "verification": applied.verification.model_dump(mode="json"),
                "stop_gate": applied.stop_gate.model_dump(mode="json"),
            },
        )
    await repo.mark_run_status(
        session_id,
        run_id,
        "completed" if applied.status == "applied" else "failed",
    )
    public = (updated or applied).to_public_dict()
    public["ok"] = applied.status == "applied"
    return public


@router.get("/api/sessions/{session_id}/runs/{run_id}/skill-changes")
async def list_run_skill_changes(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    proposals = await repo.list_skill_change_proposals(session_id, run_id)
    return {"items": [proposal.to_public_dict() for proposal in proposals]}


@router.get("/api/sessions/{session_id}/runs/{run_id}/skill-changes/{proposal_id}")
async def get_run_skill_change(
    session_id: str,
    run_id: str,
    proposal_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    proposal = await repo.get_skill_change_proposal(session_id, run_id, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill change proposal not found")
    return proposal.to_public_dict()


@router.post("/api/sessions/{session_id}/runs/{run_id}/skill-changes/{proposal_id}/reject")
async def reject_run_skill_change(
    session_id: str,
    run_id: str,
    proposal_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    proposal = await repo.get_skill_change_proposal(session_id, run_id, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill change proposal not found")
    if proposal.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Skill change proposal is not pending")
    updated = await repo.update_skill_change_proposal(
        session_id,
        run_id,
        proposal_id,
        status="rejected",
        error="rejected by user",
        decided=True,
    )
    await repo.append_event(
        session_id,
        run_id,
        "skill_change_rejected",
        "Skill change proposal rejected by user.",
        {"proposal_id": proposal_id},
    )
    await repo.mark_run_status(session_id, run_id, "cancelled")
    return (updated or proposal).to_public_dict()


@router.post("/api/sessions/{session_id}/runs/{run_id}/skill-changes/{proposal_id}/approve")
async def approve_run_skill_change(
    session_id: str,
    run_id: str,
    proposal_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    proposal = await repo.get_skill_change_proposal(session_id, run_id, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill change proposal not found")
    if proposal.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Skill change proposal is not pending")

    await repo.update_skill_change_proposal(session_id, run_id, proposal_id, status="approved", decided=True)
    applied = apply_skill_change_proposal(proposal.model_copy(update={"status": "approved"}), settings.workspace_root)
    updated = await repo.update_skill_change_proposal(
        session_id,
        run_id,
        proposal_id,
        status=applied.status,
        apply_results=[dict(item) for item in applied.apply_results],
        error=applied.error,
    )
    await repo.append_event(
        session_id,
        run_id,
        "skill_change_applied" if applied.status == "applied" else "failed",
        "Skill change proposal applied." if applied.status == "applied" else "Skill change proposal failed.",
        {"proposal_id": proposal_id, "apply_results": applied.apply_results, "error": applied.error},
    )
    await repo.mark_run_status(session_id, run_id, "completed" if applied.status == "applied" else "failed")
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


@router.get("/api/sessions/{session_id}/runs/{run_id}/skill-recipe-runs")
async def list_run_skill_recipe_runs(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    limit: int = 300,
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    events = await repo.list_run_events(session_id, run_id, limit=max(1, min(limit, 1000)))
    items = [
        _event_payload(event)
        for event in events
        if getattr(event, "type", "") == "skill_subflow_completed"
    ]
    return {"items": items}


@router.get("/api/sessions/{session_id}/runs/{run_id}/skill-recipe-runs/{recipe_run_id}")
async def get_run_skill_recipe_run(
    session_id: str,
    run_id: str,
    recipe_run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    limit: int = 1000,
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    events = await repo.list_run_events(session_id, run_id, limit=max(1, min(limit, 2000)))
    for event in events:
        payload = _event_payload(event)
        if getattr(event, "type", "") == "skill_subflow_completed" and str(payload.get("run_id") or "") == recipe_run_id:
            return payload
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill recipe run not found")


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
                if event.type in {
                    "completed",
                    "failed",
                    "cancelled",
                    "run_completed",
                    "patch_approval_required",
                    "skill_change_approval_required",
                }:
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
    checkpoint_id: str | None = None
    from_node: Literal["team_develop", "team_test", "team_supervisor"] | None = None
    recovery_hints: dict[str, object] | None = None
    human_feedback: dict[str, object] | None = None


class InterruptRunRequest(BaseModel):
    reason: str | None = None
    feedback: dict[str, object] | None = None


@router.post("/api/sessions/{session_id}/runs/{run_id}/resume")
async def resume_run(
    session_id: str,
    run_id: str,
    body: ResumeRunRequest,
    repo: Annotated[SessionRepository, Depends(get_repository)],
    settings: Annotated[Settings, Depends(get_settings)],
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

    snapshot = await repo.get_graph_snapshot(session_id, run_id, checkpoint_id=body.checkpoint_id)
    initial_state = _state_snapshot_from_graph_payload(snapshot)
    sandbox_root = _sandbox_root_from_state(initial_state)
    registry_kwargs: dict[str, object] = {"is_plan_mode": run.metadata.get("run_mode") == "plan"}
    if sandbox_root:
        registry_kwargs.update({
            "command_workspace_root": sandbox_root,
            "sandbox_mode": "copy",
            "sandbox_network_policy": settings.sandbox_network_policy,
            "sandbox_command_timeout_seconds": settings.sandbox_command_timeout_seconds,
            "sandbox_max_output_bytes": settings.sandbox_max_output_bytes,
            "sandbox_max_changed_files": settings.sandbox_max_changed_files,
            "sandbox_max_workspace_bytes": settings.sandbox_max_workspace_bytes,
        })
    registry = create_default_registry(settings.workspace_root, **registry_kwargs)
    agent_settings = _agent_settings_from_run(
        settings,
        run,
        resume_from_node=body.from_node,
        recovery_hints=body.recovery_hints or {},
        human_feedback=body.human_feedback or {},
    )
    deps = AgentDeps(tool_registry=registry, safety_inspector=registry, settings=agent_settings)
    await repo.mark_run_status(session_id, run_id, "running")

    async def event_stream() -> AsyncIterator[str]:
        prompt = run.prompt if hasattr(run, "prompt") else ""
        awaiting_approval = False
        async for event in run_agent_events(
            session_id=session_id,
            run_id=run_id,
            user_input=prompt if isinstance(prompt, str) else "",
            deps=deps,
            settings=agent_settings,
            initial_state=initial_state,
            resume_from_node=body.from_node,
        ):
            if await request.is_disconnected():
                break
            await repo.append_event(session_id, run_id, event.type, event.message, event.to_dict())
            if event.type in {"patch_approval_required", "skill_change_approval_required"}:
                awaiting_approval = True
                await repo.mark_run_status(session_id, run_id, "awaiting_approval")
            elif event.type == "run_completed":
                await repo.mark_run_status(session_id, run_id, "awaiting_approval" if awaiting_approval else "completed")
            elif event.type in {"error", "failed"}:
                await repo.mark_run_status(session_id, run_id, "failed")
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


@router.post("/api/sessions/{session_id}/runs/{run_id}/interrupt")
async def interrupt_run(
    session_id: str,
    run_id: str,
    body: InterruptRunRequest,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    if run.status in {"completed", "failed", "cancelled"}:
        return run.to_public_dict()
    new_status = "awaiting_feedback" if body.feedback else "paused"
    await repo.append_event(
        session_id,
        run_id,
        "run_interrupted",
        "Run interrupted by user",
        {"reason": body.reason or "user_interrupted", "feedback": body.feedback or {}},
    )
    await repo.mark_run_status(session_id, run_id, new_status)
    updated = await repo.get_run(session_id, run_id)
    return (updated or run).to_public_dict()


@router.get("/api/sessions/{session_id}/runs/{run_id}/artifacts")
async def get_run_artifacts(
    session_id: str,
    run_id: str,
    repo: Annotated[SessionRepository, Depends(get_repository)],
) -> dict[str, object]:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    snapshot = await repo.get_graph_snapshot(session_id, run_id)
    state = _state_snapshot_from_graph_payload(snapshot)
    artifacts = dict(state.get("sandbox_artifacts") or {})
    return {
        "run_id": run_id,
        "sandbox_artifacts": artifacts,
        "code_map_summary": dict(state.get("code_map_summary") or {}),
        "impact_analysis": dict(state.get("impact_analysis") or {}),
        "outcome_report": dict(state.get("outcome_report") or {}),
        "failure_reports": list(state.get("failure_reports") or []),
        "evidence_timeline": list(state.get("evidence_timeline") or []),
        "git_artifact_proposal": dict(state.get("git_artifact_proposal") or {}),
        "eval_report": dict(state.get("eval_report") or {}),
    }


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
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> StreamingResponse:
    run = await repo.get_run(session_id, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    snapshot = await repo.get_graph_snapshot(session_id, run_id, checkpoint_id=body.checkpoint_id)
    initial_state = _state_snapshot_from_graph_payload(snapshot)
    replay_settings = _agent_settings_from_run(
        settings,
        run,
        resume_from_node=body.from_node,
        recovery_hints={"dry_run": body.dry_run, "replay": True},
    )
    registry = create_default_registry(settings.workspace_root, is_plan_mode=run.metadata.get("run_mode") == "plan")
    deps = AgentDeps(tool_registry=registry, safety_inspector=registry, settings=replay_settings)

    async def event_stream() -> AsyncIterator[str]:
        prompt = run.prompt if hasattr(run, "prompt") else ""
        async for event in run_agent_events(
            session_id=session_id,
            run_id=f"{run_id}_replay",
            user_input=prompt if isinstance(prompt, str) else "",
            deps=deps,
            settings=replay_settings,
            initial_state=initial_state,
            resume_from_node=body.from_node,
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


def _event_payload(event: object) -> dict[str, object]:
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict):
        return payload
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        return data
    return {}
