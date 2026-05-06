"""Session repository interfaces and in-memory implementation."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path

from solo_agent.memory import SQLiteMemoryRepository, init_sqlite_memory
from solo_agent.memory.models import MessageRole, RunStatus
from solo_agent.web.models import RunEvent, RunRecord, SessionRecord, utc_now

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


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

    @abstractmethod
    async def mark_run_status(self, session_id: str, run_id: str, status: str) -> None:
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


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionRecord] = {}
        self._runs: dict[str, RunRecord] = {}
        self._session_runs: dict[str, list[str]] = defaultdict(list)
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._subscribers: dict[str, set[asyncio.Queue[RunEvent]]] = defaultdict(set)
        self._messages: dict[str, list[dict[str, object]]] = defaultdict(list)

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

    async def mark_run_status(self, session_id: str, run_id: str, status: str) -> None:
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
        repo = await self._repo()
        if status in TERMINAL_STATUSES:
            mapped = RunStatus.COMPLETED if status == "completed" else RunStatus.FAILED
            await repo.complete_run(run_id=run_id, status=mapped)

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
