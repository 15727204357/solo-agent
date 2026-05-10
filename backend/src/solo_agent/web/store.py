"""Session repository interfaces and in-memory implementation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

from solo_agent.memory import SQLiteMemoryRepository, init_sqlite_memory
from solo_agent.memory.models import MessageRole, RunStatus
from solo_agent.verified_editing import PatchProposal
from solo_agent.web.models import RunEvent, RunRecord, SessionRecord, new_id, utc_now

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
RUN_STATUSES = {*TERMINAL_STATUSES, "running", "queued", "awaiting_approval"}


class SessionRepository(ABC):
    @abstractmethod
    async def create_session(self, title: str, workspace_path: str | None) -> SessionRecord:
        raise NotImplementedError

    @abstractmethod
    async def list_sessions(self) -> list[SessionRecord]:
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionRecord | None:
        raise NotImplementedError

    @abstractmethod
    async def create_run(
        self,
        session_id: str,
        prompt: str,
        metadata: dict[str, object] | None = None,
    ) -> RunRecord:
        raise NotImplementedError

    @abstractmethod
    async def list_runs(self, session_id: str) -> list[RunRecord]:
        raise NotImplementedError

    @abstractmethod
    async def get_run(self, session_id: str, run_id: str) -> RunRecord | None:
        raise NotImplementedError

    async def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, object]]:
        raise NotImplementedError

    async def count_messages(self, session_id: str) -> int:
        raise NotImplementedError

    async def get_summary(self, session_id: str) -> dict[str, object] | None:
        raise NotImplementedError

    async def create_patch_proposal(self, proposal: PatchProposal) -> PatchProposal:
        raise NotImplementedError

    async def list_patch_proposals(self, session_id: str, run_id: str) -> list[PatchProposal]:
        raise NotImplementedError

    async def get_patch_proposal(self, session_id: str, run_id: str, patch_id: str) -> PatchProposal | None:
        raise NotImplementedError

    async def update_patch_proposal(
        self,
        session_id: str,
        run_id: str,
        patch_id: str,
        *,
        status: str | None = None,
        apply_results: list[dict[str, object]] | None = None,
        verification: dict[str, object] | None = None,
        error: str | None = None,
        decided: bool = False,
    ) -> PatchProposal | None:
        raise NotImplementedError

    async def list_memory_candidates(
        self,
        *,
        status: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    async def update_memory_candidate(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        target: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        raise NotImplementedError

    async def approve_memory_candidate(
        self,
        candidate_id: str,
        *,
        resolution: str = "add",
        content: str | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError

    async def reject_memory_candidate(self, candidate_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        raise NotImplementedError

    async def list_memory_entries(
        self,
        *,
        target: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        raise NotImplementedError

    async def revoke_memory_entry(self, entry_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        raise NotImplementedError

    @abstractmethod
    async def mark_run_status(self, session_id: str, run_id: str, status: str) -> None:
        raise NotImplementedError

    async def list_checkpoints(self, session_id: str, run_id: str) -> list[dict[str, object]]:
        raise NotImplementedError

    async def get_graph_snapshot(
        self, session_id: str, run_id: str, *, checkpoint_id: str | None = None
    ) -> dict[str, object] | None:
        raise NotImplementedError

    @abstractmethod
    async def append_event(
        self,
        session_id: str,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        raise NotImplementedError

    @abstractmethod
    async def iter_run_events(
        self,
        session_id: str,
        run_id: str,
        heartbeat_seconds: int,
    ) -> AsyncIterator[RunEvent | None]:
        raise NotImplementedError

    async def list_run_events(
            self,
            session_id: str,
            run_id: str,
            limit: int = 300,
    ) -> list[RunEvent]:
        raise NotImplementedError


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._runs: dict[str, RunRecord] = {}
        self._session_runs: dict[str, list[str]] = defaultdict(list)
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._messages: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._patches: dict[str, PatchProposal] = {}
        self._run_patches: dict[str, list[str]] = defaultdict(list)
        self._memory_candidates: dict[str, dict[str, object]] = {}
        self._memory_entries: dict[str, dict[str, object]] = {}

    async def create_session(self, title: str, workspace_path: str | None) -> SessionRecord:
        async with self._lock:
            session = SessionRecord(title=title, workspace_path=workspace_path)
            self._sessions[session.id] = session
            return session

    async def list_sessions(self) -> list[SessionRecord]:
        async with self._lock:
            return sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def create_run(
        self,
        session_id: str,
        prompt: str,
        metadata: dict[str, object] | None = None,
    ) -> RunRecord:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            run = RunRecord(session_id=session_id, prompt=prompt, metadata=metadata or {})
            self._runs[run.id] = run
            self._session_runs[session_id].append(run.id)
            self._messages[session_id].append(
                {
                    "id": f"msg_{len(self._messages[session_id]) + 1}",
                    "run_id": run.id,
                    "role": "user",
                    "content": prompt,
                    "sequence": len(self._messages[session_id]) + 1,
                }
            )
            self._sessions[session_id].updated_at = utc_now()
            return run

    async def list_runs(self, session_id: str) -> list[RunRecord]:
        async with self._lock:
            run_ids = self._session_runs.get(session_id, [])
            return [self._runs[run_id] for run_id in reversed(run_ids) if run_id in self._runs]

    async def get_run(self, session_id: str, run_id: str) -> RunRecord | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                return None
            return run

    async def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, object]]:
        async with self._lock:
            return list(self._messages.get(session_id, []))[-limit:]

    async def count_messages(self, session_id: str) -> int:
        async with self._lock:
            return len(self._messages.get(session_id, []))

    async def get_summary(self, session_id: str) -> dict[str, object] | None:
        return None

    async def create_patch_proposal(self, proposal: PatchProposal) -> PatchProposal:
        async with self._lock:
            self._patches[proposal.id] = proposal
            self._run_patches[proposal.run_id].append(proposal.id)
            if proposal.session_id in self._sessions:
                self._sessions[proposal.session_id].updated_at = utc_now()
            return proposal

    async def list_patch_proposals(self, session_id: str, run_id: str) -> list[PatchProposal]:
        async with self._lock:
            return [
                self._patches[patch_id]
                for patch_id in self._run_patches.get(run_id, [])
                if patch_id in self._patches and self._patches[patch_id].session_id == session_id
            ]

    async def get_patch_proposal(self, session_id: str, run_id: str, patch_id: str) -> PatchProposal | None:
        async with self._lock:
            proposal = self._patches.get(patch_id)
            if proposal is None or proposal.session_id != session_id or proposal.run_id != run_id:
                return None
            return proposal

    async def update_patch_proposal(
        self,
        session_id: str,
        run_id: str,
        patch_id: str,
        *,
        status: str | None = None,
        apply_results: list[dict[str, object]] | None = None,
        verification: dict[str, object] | None = None,
        error: str | None = None,
        decided: bool = False,
    ) -> PatchProposal | None:
        async with self._lock:
            proposal = self._patches.get(patch_id)
            if proposal is None or proposal.session_id != session_id or proposal.run_id != run_id:
                return None
            payload = proposal.model_dump(mode="json")
            payload.update(
                {
                    key: value
                    for key, value in {
                        "status": status,
                        "apply_results": apply_results,
                        "verification": verification,
                        "error": error,
                    }.items()
                    if value is not None
                }
            )
            proposal = PatchProposal.model_validate(payload)
            self._patches[patch_id] = proposal
            return proposal

    async def list_memory_candidates(
        self,
        *,
        status: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        async with self._lock:
            items = list(self._memory_candidates.values())
            if status is not None:
                items = [item for item in items if item.get("status") == status]
            if target is not None:
                items = [item for item in items if item.get("target") == target]
            return sorted(items, key=lambda item: str(item.get("created_at", "")), reverse=True)[:limit]

    async def update_memory_candidate(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        target: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        async with self._lock:
            candidate = self._memory_candidates.get(candidate_id)
            if candidate is None:
                return None
            if content is not None:
                candidate["content"] = content
            if target is not None:
                candidate["target"] = target
            if confidence is not None:
                candidate["confidence"] = confidence
            if metadata:
                candidate["metadata"] = {**dict(candidate.get("metadata") or {}), **metadata}
            candidate["updated_at"] = utc_now().isoformat()
            return dict(candidate)

    async def approve_memory_candidate(
        self,
        candidate_id: str,
        *,
        resolution: str = "add",
        content: str | None = None,
    ) -> dict[str, object]:
        async with self._lock:
            candidate = self._memory_candidates.get(candidate_id)
            if candidate is None:
                raise ValueError("Memory candidate not found")
            if candidate.get("status") != "pending":
                raise ValueError("Candidate is not pending")
            conflict_ids = list(candidate.get("conflict_ids") or [])
            if conflict_ids and resolution == "add":
                raise ValueError("conflict_requires_resolution")
            now = utc_now().isoformat()
            if resolution in {"replace", "merge"}:
                for entry_id in conflict_ids:
                    if entry_id in self._memory_entries:
                        self._memory_entries[entry_id]["status"] = "superseded"
                        self._memory_entries[entry_id]["updated_at"] = now
            entry = {
                "id": new_id("mem"),
                "target": candidate.get("target"),
                "content": content or candidate.get("content", ""),
                "source_candidate_id": candidate_id,
                "confidence": candidate.get("confidence", 0.0),
                "supersedes_id": conflict_ids[0] if conflict_ids else None,
                "status": "active",
                "metadata": {"resolution": resolution},
                "created_at": now,
                "updated_at": now,
                "revoked_at": None,
            }
            candidate["content"] = entry["content"]
            candidate["status"] = "approved"
            candidate["updated_at"] = now
            candidate["decided_at"] = now
            self._memory_entries[str(entry["id"])] = entry
            return {"candidate": dict(candidate), "entry": dict(entry)}

    async def reject_memory_candidate(self, candidate_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        async with self._lock:
            candidate = self._memory_candidates.get(candidate_id)
            if candidate is None:
                return None
            candidate["status"] = "rejected"
            candidate["decided_at"] = utc_now().isoformat()
            if reason:
                candidate["metadata"] = {**dict(candidate.get("metadata") or {}), "rejection_reason": reason}
            return dict(candidate)

    async def list_memory_entries(
        self,
        *,
        target: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        async with self._lock:
            items = list(self._memory_entries.values())
            if target is not None:
                items = [item for item in items if item.get("target") == target]
            if not include_inactive:
                items = [item for item in items if item.get("status") == "active"]
            return sorted(items, key=lambda item: str(item.get("updated_at", "")), reverse=True)[:limit]

    async def revoke_memory_entry(self, entry_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        async with self._lock:
            entry = self._memory_entries.get(entry_id)
            if entry is None:
                return None
            now = utc_now().isoformat()
            entry["status"] = "revoked"
            entry["revoked_at"] = now
            entry["updated_at"] = now
            if reason:
                entry["metadata"] = {**dict(entry.get("metadata") or {}), "revocation_reason": reason}
            return dict(entry)

    async def mark_run_status(self, session_id: str, run_id: str, status: str) -> None:
        if status not in RUN_STATUSES:
            return
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                return
            now = utc_now()
            run.status = status
            run.updated_at = now
            if status in TERMINAL_STATUSES:
                run.completed_at = now
            if session_id in self._sessions:
                self._sessions[session_id].updated_at = now

    async def append_event(
        self,
        session_id: str,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        async with self._lock:
            sequence = len(self._events[run_id]) + 1
            event = RunEvent(
                session_id=session_id,
                run_id=run_id,
                sequence=sequence,
                type=event_type,
                message=message,
                payload=payload or {},
            )
            self._events[run_id].append(event)
            subscribers = list(self._subscribers.get(run_id, set()))
            if event_type == "response_completed":
                response = _agent_event_data(payload or {}).get("response")
                if isinstance(response, str):
                    self._messages[session_id].append(
                        {
                            "id": f"msg_{len(self._messages[session_id]) + 1}",
                            "run_id": run_id,
                            "role": "assistant",
                            "content": response,
                            "sequence": len(self._messages[session_id]) + 1,
                        }
                    )

        for queue in subscribers:
            await queue.put(event)
        return event

    async def iter_run_events(
        self,
        session_id: str,
        run_id: str,
        heartbeat_seconds: int,
    ) -> AsyncIterator[RunEvent | None]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                return
            replay = list(self._events.get(run_id, []))
            self._subscribers[run_id].add(queue)

        try:
            for event in replay:
                yield event
                if event.type in TERMINAL_STATUSES or event.type == "run_completed":
                    return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except TimeoutError:
                    yield None
                    continue

                yield event
                if event.type in TERMINAL_STATUSES or event.type == "run_completed":
                    return
        finally:
            async with self._lock:
                self._subscribers.get(run_id, set()).discard(queue)

    async def list_run_events(
            self,
            session_id: str,
            run_id: str,
            limit: int = 300,
    ) -> list[RunEvent]:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.session_id != session_id:
                return []
            return list(self._events.get(run_id, []))[-limit:]


class SQLiteSessionRepository(SessionRepository):
    """Web repository backed by SQLite memory tables plus in-process SSE queues."""

    def __init__(self, database_url: str | Path, *, memory_root: str | Path | None = None) -> None:
        self._database_url = str(database_url)
        self._memory_root = memory_root
        self._memory: SQLiteMemoryRepository | None = None
        self._init_lock = asyncio.Lock()
        self._event_lock = asyncio.Lock()
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)

    async def _repo(self) -> SQLiteMemoryRepository:
        if self._memory is not None:
            return self._memory
        async with self._init_lock:
            if self._memory is None:
                self._memory = await init_sqlite_memory(
                    self._database_url,
                    memory_root=self._memory_root,
                )
            return self._memory

    async def create_session(self, title: str, workspace_path: str | None) -> SessionRecord:
        repo = await self._repo()
        record = await repo.create_session(title=title, workspace_path=workspace_path)
        return _session_from_memory(record)

    async def list_sessions(self) -> list[SessionRecord]:
        repo = await self._repo()
        records = await repo.list_sessions()
        return [_session_from_memory(record) for record in records]

    async def get_session(self, session_id: str) -> SessionRecord | None:
        repo = await self._repo()
        record = await repo.get_session(session_id)
        return _session_from_memory(record) if record else None

    async def create_run(
        self,
        session_id: str,
        prompt: str,
        metadata: dict[str, object] | None = None,
    ) -> RunRecord:
        repo = await self._repo()
        record = await repo.create_run(
            session_id=session_id,
            provider=None,
            model=None,
            metadata={"prompt": prompt, **dict(metadata or {})},
        )
        await repo.append_message(
            session_id=session_id,
            run_id=record.id,
            role=MessageRole.USER,
            content=prompt,
            metadata={"source": "web"},
        )
        return _run_from_memory(record)

    async def list_runs(self, session_id: str) -> list[RunRecord]:
        repo = await self._repo()
        records = await repo.list_runs(session_id)
        return [_run_from_memory(record) for record in records]

    async def get_run(self, session_id: str, run_id: str) -> RunRecord | None:
        repo = await self._repo()
        record = await repo.get_run(run_id)
        if record is None or record.session_id != session_id:
            return None
        return _run_from_memory(record)

    async def list_messages(self, session_id: str, limit: int = 50) -> list[dict[str, object]]:
        repo = await self._repo()
        records = await repo.list_messages(session_id, limit=limit)
        return [_message_to_dict(record) for record in records]

    async def count_messages(self, session_id: str) -> int:
        repo = await self._repo()
        return await repo.count_messages(session_id)

    async def get_summary(self, session_id: str) -> dict[str, object] | None:
        repo = await self._repo()
        summary = await repo.get_latest_summary(session_id)
        if summary is None:
            return None
        return {
            "id": summary.id,
            "label": summary.label,
            "summary": summary.data.get("summary", ""),
            "created_at": summary.created_at.isoformat(),
        }

    async def memory_repository(self) -> SQLiteMemoryRepository:
        return await self._repo()

    async def mark_run_status(self, session_id: str, run_id: str, status: str) -> None:
        if status not in RUN_STATUSES:
            return
        repo = await self._repo()
        if hasattr(repo, "set_run_status"):
            await repo.set_run_status(run_id=run_id, status=status)
        elif status in TERMINAL_STATUSES:
            mapped = RunStatus.COMPLETED if status == "completed" else RunStatus.FAILED
            await repo.complete_run(run_id=run_id, status=mapped)

    async def create_patch_proposal(self, proposal: PatchProposal) -> PatchProposal:
        repo = await self._repo()
        record = await repo.create_patch_proposal(
            session_id=proposal.session_id,
            run_id=proposal.run_id,
            patch_id=proposal.id,
            summary=proposal.summary,
            edits=[edit.model_dump(mode="json") for edit in proposal.edits],
            diff=proposal.diff,
            status=proposal.status,
        )
        return _patch_from_memory(record)

    async def list_patch_proposals(self, session_id: str, run_id: str) -> list[PatchProposal]:
        repo = await self._repo()
        records = await repo.list_patch_proposals(session_id=session_id, run_id=run_id)
        return [_patch_from_memory(record) for record in records]

    async def get_patch_proposal(self, session_id: str, run_id: str, patch_id: str) -> PatchProposal | None:
        repo = await self._repo()
        record = await repo.get_patch_proposal(patch_id)
        if record is None or record.session_id != session_id or record.run_id != run_id:
            return None
        return _patch_from_memory(record)

    async def update_patch_proposal(
        self,
        session_id: str,
        run_id: str,
        patch_id: str,
        *,
        status: str | None = None,
        apply_results: list[dict[str, object]] | None = None,
        verification: dict[str, object] | None = None,
        error: str | None = None,
        decided: bool = False,
    ) -> PatchProposal | None:
        repo = await self._repo()
        record = await repo.update_patch_proposal(
            patch_id=patch_id,
            status=status,
            apply_results=apply_results,
            verification=verification,
            error=error,
            decided=decided,
        )
        if record is None or record.session_id != session_id or record.run_id != run_id:
            return None
        return _patch_from_memory(record)

    async def list_memory_candidates(
        self,
        *,
        status: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        repo = await self._repo()
        return await repo.list_memory_candidates(status=status, target=target, limit=limit)

    async def update_memory_candidate(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        target: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        repo = await self._repo()
        return await repo.update_memory_candidate(
            candidate_id,
            content=content,
            target=target,
            confidence=confidence,
            metadata=metadata,
        )

    async def approve_memory_candidate(
        self,
        candidate_id: str,
        *,
        resolution: str = "add",
        content: str | None = None,
    ) -> dict[str, object]:
        repo = await self._repo()
        return await repo.approve_memory_candidate(candidate_id, resolution=resolution, content=content)

    async def reject_memory_candidate(self, candidate_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        repo = await self._repo()
        return await repo.reject_memory_candidate(candidate_id, reason=reason)

    async def list_memory_entries(
        self,
        *,
        target: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        repo = await self._repo()
        return await repo.list_memory_entries(target=target, include_inactive=include_inactive, limit=limit)

    async def revoke_memory_entry(self, entry_id: str, *, reason: str | None = None) -> dict[str, object] | None:
        repo = await self._repo()
        return await repo.revoke_memory_entry(entry_id, reason=reason)

    async def list_checkpoints(self, session_id: str, run_id: str) -> list[dict[str, object]]:
        repo = await self._repo()
        records = await repo.list_checkpoint_refs(run_id=run_id)
        return [
            {
                "checkpoint_id": r.data.get("checkpoint_id", ""),
                "node_name": r.data.get("node_name", ""),
                "step_number": r.data.get("step_number", 0),
                "created_at": r.created_at.isoformat(),
                "snapshot_id": r.id,
            }
            for r in records
        ]

    async def get_graph_snapshot(
        self, session_id: str, run_id: str, *, checkpoint_id: str | None = None
    ) -> dict[str, object] | None:
        repo = await self._repo()
        if checkpoint_id:
            records = await repo.list_checkpoint_refs(run_id=run_id, limit=1)
            for r in records:
                if r.data.get("checkpoint_id") == checkpoint_id:
                    return {"checkpoint_id": checkpoint_id, "run_id": run_id, "data": r.data}
            return None
        records = await repo.list_graph_timeline(run_id=run_id, limit=1)
        if not records:
            return None
        return {"snapshot_id": records[-1].id, "type": records[-1].snapshot_type, "data": records[-1].data}

    async def append_event(
        self,
        session_id: str,
        run_id: str,
        event_type: str,
        message: str,
        payload: dict[str, object] | None = None,
    ) -> RunEvent:
        repo = await self._repo()
        payload = payload or {}
        agent_data = _agent_event_data(payload)
        if event_type == "response_completed" and isinstance(agent_data.get("response"), str):
            await repo.append_message(
                session_id=session_id,
                run_id=run_id,
                role=MessageRole.ASSISTANT,
                content=str(agent_data["response"]),
                metadata={"source": "agent"},
            )
        elif event_type == "tool_call_completed":
            await repo.append_tool_call(
                run_id=run_id,
                name=str(agent_data.get("name", "unknown")),
                arguments=dict(agent_data.get("arguments") or {}),
                result=str(agent_data.get("result", "")),
                status="completed",
            )
        elif event_type in {
            "plan_completed",
            "context_completed",
            "persist_started",
            "persist_snapshot_completed",
        }:
            await repo.append_snapshot(
                session_id=session_id,
                run_id=run_id,
                label=event_type,
                data=dict(agent_data or payload),
            )
        await repo.append_timing_point(
            run_id=run_id,
            label=event_type,
            category="agent_event",
            metadata={"message": message},
        )

        async with self._event_lock:
            sequence = len(self._events[run_id]) + 1
            event = RunEvent(
                session_id=session_id,
                run_id=run_id,
                sequence=sequence,
                type=event_type,
                message=message,
                payload=payload,
            )
            self._events[run_id].append(event)
            subscribers = list(self._subscribers.get(run_id, set()))

        for queue in subscribers:
            await queue.put(event)
        return event

    async def iter_run_events(
        self,
        session_id: str,
        run_id: str,
        heartbeat_seconds: int,
    ) -> AsyncIterator[RunEvent | None]:
        queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        async with self._event_lock:
            replay = list(self._events.get(run_id, []))
            self._subscribers[run_id].add(queue)

        try:
            for event in replay:
                yield event
                if event.type in TERMINAL_STATUSES or event.type == "run_completed":
                    return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except TimeoutError:
                    yield None
                    continue

                yield event
                if event.type in TERMINAL_STATUSES or event.type == "run_completed":
                    return
        finally:
            async with self._event_lock:
                self._subscribers.get(run_id, set()).discard(queue)

    async def list_run_events(
            self,
            session_id: str,
            run_id: str,
            limit: int = 300,
    ) -> list[RunEvent]:
        async with self._event_lock:
            return [
                       event
                       for event in self._events.get(run_id, [])
                       if event.session_id == session_id
                   ][-limit:]


def _session_from_memory(record: object) -> SessionRecord:
    return SessionRecord(
        id=str(record.id),
        title=record.title or "New coding session",
        workspace_path=record.workspace_path,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _run_from_memory(record: object) -> RunRecord:
    metadata = getattr(record, "metadata_", None) or {}
    return RunRecord(
        id=str(record.id),
        session_id=str(record.session_id),
        prompt=str(metadata.get("prompt", "")),
        metadata=dict(metadata),
        status=str(record.status),
        created_at=record.started_at,
        updated_at=record.completed_at or record.started_at,
        completed_at=record.completed_at,
    )


def _patch_from_memory(record: object) -> PatchProposal:
    return PatchProposal(
        id=str(record.id),
        session_id=str(record.session_id),
        run_id=str(record.run_id),
        status=str(record.status),
        summary=str(record.summary or ""),
        diff=str(record.diff or ""),
        edits=list(record.edits or []),
        apply_results=list(record.apply_results or []),
        verification=record.verification,
        error=record.error,
        created_at=record.created_at.isoformat() if record.created_at else None,
        updated_at=record.updated_at.isoformat() if record.updated_at else None,
        decided_at=record.decided_at.isoformat() if record.decided_at else None,
    )


def _message_to_dict(record: object) -> dict[str, object]:
    return {
        "id": str(record.id),
        "session_id": str(record.session_id),
        "run_id": str(record.run_id) if record.run_id else None,
        "role": str(record.role),
        "content": str(record.content),
        "sequence": int(record.sequence),
        "created_at": record.created_at.isoformat(),
    }


def _agent_event_data(payload: dict[str, object]) -> dict[str, object]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload

