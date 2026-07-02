from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from .database import create_memory_engine, create_session_factory, init_database
from .governance import MemoryGovernanceError, MemoryGovernanceService
from .models import (
    MemoryCandidateRecord,
    MemoryEntryRecord,
    MessageRecord,
    MessageRole,
    PatchProposalRecord,
    RunRecord,
    RunStatus,
    SessionRecord,
    SessionType,
    SkillChangeProposalRecord,
    SnapshotRecord,
    SnapshotType,
    TimingPointRecord,
    ToolCallRecord,
    ToolCallStatus,
    utc_now,
)


def _new_id() -> str:
    return str(uuid4())


def _value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return value


def _merge_metadata(existing: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    if updates:
        merged.update(updates)
    return merged


def _fts_query(query: str) -> str:
    terms = [
        term.replace('"', "").strip()
        for term in query.split()
        if term.replace('"', "").strip()
    ]
    return " OR ".join(f'"{term}"' for term in terms[:8])


def _message_dict(message: MessageRecord) -> dict[str, Any]:
    return {
        "id": message.id,
        "run_id": message.run_id,
        "role": message.role,
        "content": message.content,
        "sequence": message.sequence,
        "created_at": message.created_at.isoformat(),
    }


def _memory_candidate_dict(record: MemoryCandidateRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "target": record.target,
        "content": record.content,
        "source_session_id": record.source_session_id,
        "source_run_id": record.source_run_id,
        "source_message_id": record.source_message_id,
        "source_excerpt": record.source_excerpt,
        "confidence": record.confidence,
        "status": record.status,
        "duplicate_of_id": record.duplicate_of_id,
        "conflict_ids": list(record.conflict_ids or []),
        "safety_flags": list(record.safety_flags or []),
        "metadata": dict(record.metadata_ or {}),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "decided_at": record.decided_at.isoformat() if record.decided_at else None,
    }


def _memory_entry_dict(record: MemoryEntryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "target": record.target,
        "content": record.content,
        "source_candidate_id": record.source_candidate_id,
        "confidence": record.confidence,
        "supersedes_id": record.supersedes_id,
        "status": record.status,
        "metadata": dict(record.metadata_ or {}),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "revoked_at": record.revoked_at.isoformat() if record.revoked_at else None,
    }


def _snapshot_summary(snapshot: SnapshotRecord | None) -> str:
    if snapshot is None:
        return ""
    return str(snapshot.data.get("summary", ""))


def _extract_preference_lines(payload: dict[str, Any]) -> list[str]:
    def flatten(value: Any) -> str:
        if isinstance(value, dict):
            return "\n".join(flatten(item) for item in value.values())
        if isinstance(value, list | tuple | set):
            return "\n".join(flatten(item) for item in value)
        return str(value)

    text = flatten(payload)
    markers = ("偏好", "prefer", "preference", "记住")
    seen: set[str] = set()
    insights: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" -'\"")
        lowered = line.lower()
        if not line or not any(marker in lowered for marker in markers):
            continue
        if line in seen:
            continue
        seen.add(line)
        insights.append(line[:240])
        if len(insights) >= 10:
            break
    return insights


class SQLiteMemoryRepository:
    """Async persistence boundary for Solo Agent session memory."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        memory_root: str | Path | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._engine = engine
        self.memory_root = Path(memory_root or Path.cwd()).resolve()
        self._prefetch_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._governance = MemoryGovernanceService(session_factory, memory_root=self.memory_root)

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> SQLiteMemoryRepository:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @classmethod
    def from_engine(
        cls,
        engine: AsyncEngine,
        *,
        memory_root: str | Path | None = None,
    ) -> SQLiteMemoryRepository:
        return cls(create_session_factory(engine), memory_root=memory_root, engine=engine)

    @classmethod
    def from_url(
        cls,
        database_path: str,
        *,
        echo: bool = False,
        memory_root: str | Path | None = None,
    ) -> SQLiteMemoryRepository:
        engine = create_memory_engine(database_path, echo=echo)
        return cls.from_engine(engine, memory_root=memory_root)

    async def create_session(
        self,
        *,
        title: str | None = None,
        session_type: SessionType | str = SessionType.CHAT,
        workspace_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        now = utc_now()
        record = SessionRecord(
            id=_new_id(),
            title=title,
            session_type=_value(session_type),
            workspace_path=workspace_path,
            metadata_=metadata or {},
            created_at=now,
            updated_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
            return record

    async def list_sessions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
        session_type: SessionType | str | None = None,
    ) -> Sequence[SessionRecord]:
        statement = select(SessionRecord)
        if not include_archived:
            statement = statement.where(SessionRecord.archived_at.is_(None))
        if session_type is not None:
            statement = statement.where(SessionRecord.session_type == _value(session_type))
        statement = statement.order_by(SessionRecord.updated_at.desc()).limit(limit).offset(offset)

        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def get_session(self, session_id: str) -> SessionRecord | None:
        async with self._session_factory() as session:
            return await session.get(SessionRecord, session_id)

    async def create_run(
        self,
        *,
        session_id: str,
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        now = utc_now()
        record = RunRecord(
            id=_new_id(),
            session_id=session_id,
            provider=provider,
            model=model,
            status=RunStatus.RUNNING.value,
            metadata_=metadata or {},
            started_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def list_runs(self, session_id: str, *, limit: int = 50, offset: int = 0) -> Sequence[RunRecord]:
        statement = (
            select(RunRecord)
            .where(RunRecord.session_id == session_id)
            .order_by(RunRecord.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self._session_factory() as session:
            return await session.get(RunRecord, run_id)

    async def set_run_status(
        self,
        *,
        run_id: str,
        status: RunStatus | str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord | None:
        now = utc_now()
        async with self._session_factory() as session:
            record = await session.get(RunRecord, run_id)
            if record is None:
                return None
            record.status = _value(status)
            record.error = error
            record.metadata_ = _merge_metadata(record.metadata_, metadata)
            if record.status in {
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                record.completed_at = now
            else:
                record.completed_at = None
            await self._touch_session(session, record.session_id, now)
            await session.commit()
            return record

    async def list_messages(
        self,
        session_id: str,
        *,
        limit: int = 50,
        before_sequence: int | None = None,
        roles: Sequence[MessageRole | str] | None = None,
    ) -> Sequence[MessageRecord]:
        statement = select(MessageRecord).where(MessageRecord.session_id == session_id)
        if before_sequence is not None:
            statement = statement.where(MessageRecord.sequence < before_sequence)
        if roles:
            statement = statement.where(MessageRecord.role.in_([_value(role) for role in roles]))
        statement = statement.order_by(MessageRecord.sequence.desc()).limit(limit)

        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return list(reversed(result.all()))

    async def count_messages(self, session_id: str) -> int:
        statement = select(func.count(MessageRecord.id)).where(MessageRecord.session_id == session_id)
        async with self._session_factory() as session:
            return int(await session.scalar(statement) or 0)

    async def search_memory(
        self,
        *,
        session_id: str,
        query: str,
        limit: int = 5,
        include_builtin: bool = True,
    ) -> list[dict[str, Any]]:
        cleaned = _fts_query(query)
        if not cleaned:
            return []

        session_condition = "session_id IN (:session_id, :builtin_session)" if include_builtin else "session_id = :session_id"
        statement = text(
            f"""
            SELECT message_id AS id, session_id, role, source, content,
                   bm25(messages_fts) AS rank
            FROM messages_fts
            WHERE messages_fts MATCH :query AND {session_condition}
            ORDER BY rank
            LIMIT :limit
            """
        )
        async with self._session_factory() as session:
            rows = await session.execute(
                statement,
                {
                    "query": cleaned,
                    "session_id": session_id,
                    "builtin_session": "__builtin__",
                    "limit": limit,
                },
            )
            results = [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "source": row.source,
                    "content": row.content,
                    "rank": row.rank,
                }
                for row in rows
            ]
            if results:
                return results

            like_rows = await session.execute(
                text(
                    f"""
                    SELECT message_id AS id, session_id, role, source, content, 0.0 AS rank
                    FROM messages_fts
                    WHERE {session_condition} AND content LIKE :like_query
                    LIMIT :limit
                    """
                ),
                {
                    "session_id": session_id,
                    "builtin_session": "__builtin__",
                    "like_query": f"%{query.strip()}%",
                    "limit": limit,
                },
            )
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "role": row.role,
                    "source": row.source,
                    "content": row.content,
                    "rank": row.rank,
                }
                for row in like_rows
            ]

    async def get_latest_summary(self, session_id: str) -> SnapshotRecord | None:
        statement = (
            select(SnapshotRecord)
            .where(
                SnapshotRecord.session_id == session_id,
                SnapshotRecord.snapshot_type == SnapshotType.SUMMARY.value,
            )
            .order_by(SnapshotRecord.created_at.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            return await session.scalar(statement)

    async def get_context_stats(self, session_id: str) -> dict[str, Any]:
        """读取最近一次压缩写入的上下文统计信息。"""

        summary = await self.get_latest_summary(session_id)
        if summary is None:
            return {}
        metadata = dict(summary.metadata_ or {})
        stats = metadata.get("context_stats")
        return dict(stats) if isinstance(stats, dict) else metadata

    async def append_or_update_summary_snapshot(
        self,
        *,
        session_id: str,
        summary: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        async with self._session_factory() as session:
            await session.execute(
                delete(SnapshotRecord).where(
                    SnapshotRecord.session_id == session_id,
                    SnapshotRecord.snapshot_type == SnapshotType.SUMMARY.value,
                    SnapshotRecord.label == "conversation_summary",
                )
            )
            now = utc_now()
            record = SnapshotRecord(
                id=_new_id(),
                session_id=session_id,
                run_id=run_id,
                label="conversation_summary",
                snapshot_type=SnapshotType.SUMMARY.value,
                data={"summary": summary},
                metadata_=metadata or {},
                created_at=now,
            )
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def prefetch_all(
        self,
        *,
        session_id: str,
        query: str,
        limit: int = 5,
        recent_limit: int = 12,
        include_history: bool = True,
    ) -> dict[str, Any]:
        cache_key = (session_id, query)
        cached = self._prefetch_cache.pop(cache_key, None)
        builtin = await self.load_builtin_memory()
        await self._index_builtin_memory(builtin)
        summary = await self.get_latest_summary(session_id)
        if summary is not None:
            await self._index_summary(session_id, summary)
        recent = await self.list_messages(session_id, limit=recent_limit) if include_history else []
        retrieved = cached.get("retrieved_memories", []) if cached else await self.search_memory(
            session_id=session_id,
            query=query,
            limit=limit,
        )
        return {
            "builtin_memory": builtin,
            "summary": _snapshot_summary(summary),
            "recent_messages": [_message_dict(message) for message in recent],
            "retrieved_memories": retrieved,
            "cache_hit": cached is not None,
        }

    async def sync_all(
        self,
        *,
        session_id: str,
        run_id: str,
        user_input: str,
        assistant_response: str,
    ) -> dict[str, Any]:
        existing = await self.list_messages(session_id, limit=5)
        has_user = any(message.run_id == run_id and message.role == MessageRole.USER.value for message in existing)
        has_assistant = any(
            message.run_id == run_id and message.role == MessageRole.ASSISTANT.value for message in existing
        )
        if not has_user:
            await self.append_message(
                session_id=session_id,
                run_id=run_id,
                role=MessageRole.USER,
                content=user_input,
                metadata={"source": "sync_all"},
            )
        if assistant_response and not has_assistant:
            await self.append_message(
                session_id=session_id,
                run_id=run_id,
                role=MessageRole.ASSISTANT,
                content=assistant_response,
                metadata={"source": "sync_all"},
            )
        return {"synced": True, "run_id": run_id}

    async def queue_prefetch_all(
        self,
        *,
        session_id: str,
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        self._prefetch_cache[(session_id, query)] = {
            "retrieved_memories": await self.search_memory(
                session_id=session_id,
                query=query,
                limit=limit,
            )
        }
        return {"queued": True, "session_id": session_id}

    async def on_pre_compress(
        self,
        *,
        session_id: str,
        payload: dict[str, Any],
        run_id: str | None = None,
    ) -> dict[str, Any]:
        insights = _extract_preference_lines(payload)
        warnings: list[str] = []
        candidates: list[MemoryCandidateRecord] = []
        try:
            candidates = await self._governance.submit_from_pre_compress(
                session_id=session_id,
                run_id=run_id,
                payload=payload,
                insights=insights,
            )
        except (MemoryGovernanceError, OSError) as exc:
            warnings.append(str(exc))
        return {
            "insights": insights,
            "session_id": session_id,
            "warnings": warnings,
            "candidates": [_memory_candidate_dict(candidate) for candidate in candidates],
        }

    async def load_builtin_memory(self) -> dict[str, str]:
        await self._governance.ensure_seeded_builtin_entries()
        return {
            "memory": (self.memory_root / "MEMORY.md").read_text(encoding="utf-8"),
            "user": (self.memory_root / "USER.md").read_text(encoding="utf-8"),
        }

    async def list_memory_candidates(
        self,
        *,
        status: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        records = await self._governance.list_candidates(status=status, target=target, limit=limit)
        return [_memory_candidate_dict(record) for record in records]

    async def update_memory_candidate(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        target: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        record = await self._governance.update_candidate(
            candidate_id,
            content=content,
            target=target,
            confidence=confidence,
            metadata=metadata,
        )
        return _memory_candidate_dict(record) if record else None

    async def approve_memory_candidate(
        self,
        candidate_id: str,
        *,
        resolution: str = "add",
        content: str | None = None,
    ) -> dict[str, Any]:
        candidate, entry = await self._governance.approve_candidate(
            candidate_id,
            resolution=resolution,
            content=content,
        )
        await self._index_builtin_memory(await self.load_builtin_memory())
        return {
            "candidate": _memory_candidate_dict(candidate),
            "entry": _memory_entry_dict(entry),
        }

    async def reject_memory_candidate(self, candidate_id: str, *, reason: str | None = None) -> dict[str, Any] | None:
        record = await self._governance.reject_candidate(candidate_id, reason=reason)
        return _memory_candidate_dict(record) if record else None

    async def list_memory_entries(
        self,
        *,
        target: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        records = await self._governance.list_entries(target=target, include_inactive=include_inactive, limit=limit)
        return [_memory_entry_dict(record) for record in records]

    async def revoke_memory_entry(self, entry_id: str, *, reason: str | None = None) -> dict[str, Any] | None:
        record = await self._governance.revoke_entry(entry_id, reason=reason)
        await self._index_builtin_memory(await self.load_builtin_memory())
        return _memory_entry_dict(record) if record else None

    async def append_message(
        self,
        *,
        session_id: str,
        role: MessageRole | str,
        content: str,
        run_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> MessageRecord:
        now = created_at or utc_now()
        async with self._session_factory() as session:
            sequence = await self._next_message_sequence(session, session_id)
            record = MessageRecord(
                id=_new_id(),
                session_id=session_id,
                run_id=run_id,
                role=_value(role),
                content=content,
                name=name,
                sequence=sequence,
                metadata_=metadata or {},
                created_at=now,
            )
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def append_tool_call(
        self,
        *,
        run_id: str,
        name: str,
        arguments: dict[str, Any] | None = None,
        message_id: str | None = None,
        status: ToolCallStatus | str = ToolCallStatus.RUNNING,
        result: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> ToolCallRecord:
        now = started_at or utc_now()
        record = ToolCallRecord(
            id=_new_id(),
            run_id=run_id,
            message_id=message_id,
            name=name,
            arguments=arguments or {},
            result=result,
            status=_value(status),
            error=error,
            metadata_=metadata or {},
            started_at=now,
            completed_at=completed_at,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session_for_run(session, run_id, now)
            await session.commit()
            return record

    async def append_timing_point(
        self,
        *,
        run_id: str,
        label: str,
        category: str | None = None,
        offset_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> TimingPointRecord:
        now = created_at or utc_now()
        record = TimingPointRecord(
            id=_new_id(),
            run_id=run_id,
            label=label,
            category=category,
            offset_ms=offset_ms,
            metadata_=metadata or {},
            created_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session_for_run(session, run_id, now)
            await session.commit()
            return record

    async def append_snapshot(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        label: str | None = None,
        data: dict[str, Any] | None = None,
        snapshot_type: SnapshotType | str = SnapshotType.CHECKPOINT,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        now = created_at or utc_now()
        record = SnapshotRecord(
            id=_new_id(),
            session_id=session_id,
            run_id=run_id,
            label=label,
            snapshot_type=_value(snapshot_type),
            data=data or {},
            metadata_=metadata or {},
            created_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def create_patch_proposal(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        patch_id: str | None = None,
        summary: str = "",
        edits: list[dict[str, Any]] | None = None,
        diff: str = "",
        status: str = "pending",
        verification_plan: dict[str, Any] | None = None,
        stop_gate: dict[str, Any] | None = None,
        proposal: Any | None = None,
    ) -> PatchProposalRecord:
        if proposal is not None:
            session_id = proposal.session_id
            run_id = proposal.run_id
            patch_id = proposal.id
            summary = proposal.summary
            edits = [edit.model_dump(mode="json") for edit in proposal.edits]
            diff = proposal.diff
            status = proposal.status
            verification_plan = proposal.verification_plan.model_dump(mode="json")
            stop_gate = proposal.stop_gate.model_dump(mode="json")
        if session_id is None or run_id is None or patch_id is None:
            raise ValueError("session_id, run_id, and patch_id are required")
        now = utc_now()
        record = PatchProposalRecord(
            id=patch_id,
            session_id=session_id,
            run_id=run_id,
            status=status,
            summary=summary,
            edits=edits or [],
            diff=diff,
            verification_plan=verification_plan or {},
            stop_gate=stop_gate or {},
            apply_results=[],
            verification=None,
            created_at=now,
            updated_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def list_patch_proposals(self, *, session_id: str, run_id: str) -> Sequence[PatchProposalRecord]:
        statement = (
            select(PatchProposalRecord)
            .where(PatchProposalRecord.session_id == session_id, PatchProposalRecord.run_id == run_id)
            .order_by(PatchProposalRecord.created_at.desc())
        )
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def get_patch_proposal(self, patch_id: str) -> PatchProposalRecord | None:
        async with self._session_factory() as session:
            return await session.get(PatchProposalRecord, patch_id)

    async def update_patch_proposal(
        self,
        *,
        patch_id: str,
        status: str | None = None,
        apply_results: list[dict[str, Any]] | None = None,
        verification: dict[str, Any] | None = None,
        verification_plan: dict[str, Any] | None = None,
        stop_gate: dict[str, Any] | None = None,
        error: str | None = None,
        decided: bool = False,
    ) -> PatchProposalRecord | None:
        now = utc_now()
        async with self._session_factory() as session:
            record = await session.get(PatchProposalRecord, patch_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if apply_results is not None:
                record.apply_results = apply_results
            if verification is not None:
                record.verification = verification
            if verification_plan is not None:
                record.verification_plan = verification_plan
            if stop_gate is not None:
                record.stop_gate = stop_gate
            record.error = error
            record.updated_at = now
            if decided:
                record.decided_at = now
            await self._touch_session(session, record.session_id, now)
            await session.commit()
            return record

    async def create_skill_change_proposal(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        proposal_id: str | None = None,
        action: str = "",
        skill_name: str = "",
        target_paths: list[str] | None = None,
        diff: str = "",
        operations: list[dict[str, Any]] | None = None,
        status: str = "pending",
        proposal: Any | None = None,
    ) -> SkillChangeProposalRecord:
        if proposal is not None:
            session_id = proposal.session_id
            run_id = proposal.run_id
            proposal_id = proposal.id
            action = proposal.action
            skill_name = proposal.skill_name
            target_paths = list(proposal.target_paths)
            diff = proposal.diff
            operations = [operation.model_dump(mode="json") for operation in proposal.operations]
            status = proposal.status
        if session_id is None or run_id is None or proposal_id is None:
            raise ValueError("session_id, run_id, and proposal_id are required")
        now = utc_now()
        record = SkillChangeProposalRecord(
            id=proposal_id,
            session_id=session_id,
            run_id=run_id,
            status=status,
            action=action,
            skill_name=skill_name,
            target_paths=target_paths or [],
            diff=diff,
            operations=operations or [],
            apply_results=[],
            created_at=now,
            updated_at=now,
        )
        async with self._session_factory() as session:
            session.add(record)
            await self._touch_session(session, session_id, now)
            await session.commit()
            return record

    async def list_skill_change_proposals(self, *, session_id: str, run_id: str) -> Sequence[SkillChangeProposalRecord]:
        statement = (
            select(SkillChangeProposalRecord)
            .where(SkillChangeProposalRecord.session_id == session_id, SkillChangeProposalRecord.run_id == run_id)
            .order_by(SkillChangeProposalRecord.created_at.desc())
        )
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def get_skill_change_proposal(self, proposal_id: str) -> SkillChangeProposalRecord | None:
        async with self._session_factory() as session:
            return await session.get(SkillChangeProposalRecord, proposal_id)

    async def update_skill_change_proposal(
        self,
        *,
        proposal_id: str,
        status: str | None = None,
        apply_results: list[dict[str, Any]] | None = None,
        error: str | None = None,
        decided: bool = False,
    ) -> SkillChangeProposalRecord | None:
        now = utc_now()
        async with self._session_factory() as session:
            record = await session.get(SkillChangeProposalRecord, proposal_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if apply_results is not None:
                record.apply_results = apply_results
            record.error = error
            record.updated_at = now
            if decided:
                record.decided_at = now
            await self._touch_session(session, record.session_id, now)
            await session.commit()
            return record

    async def save_route_decision(
        self,
        *,
        session_id: str,
        run_id: str,
        node: str,
        route_name: str,
        selected: str,
        reason: str,
        evidence: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        return await self.append_snapshot(
            session_id=session_id,
            run_id=run_id,
            label=f"route:{node}:{selected}",
            snapshot_type=SnapshotType.ROUTE_DECISION,
            data={
                "node": node,
                "route_name": route_name,
                "selected": selected,
                "reason": reason,
                "evidence": evidence or {},
            },
            created_at=created_at,
        )

    async def save_graph_snapshot(
        self,
        *,
        session_id: str,
        run_id: str,
        label: str,
        node_name: str | None = None,
        state_snapshot: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        return await self.append_snapshot(
            session_id=session_id,
            run_id=run_id,
            label=label,
            snapshot_type=SnapshotType.GRAPH_SNAPSHOT,
            data={
                "node_name": node_name,
                "state_snapshot": state_snapshot or {},
            },
            created_at=created_at,
        )

    async def save_checkpoint_ref(
        self,
        *,
        session_id: str,
        run_id: str,
        checkpoint_id: str,
        node_name: str,
        step_number: int,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        return await self.append_snapshot(
            session_id=session_id,
            run_id=run_id,
            label=f"checkpoint:{node_name}:{step_number}",
            snapshot_type=SnapshotType.CHECKPOINT_REF,
            data={
                "checkpoint_id": checkpoint_id,
                "node_name": node_name,
                "step_number": step_number,
            },
            created_at=created_at,
        )

    async def list_checkpoint_refs(
        self,
        *,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[SnapshotRecord]:
        statement = (
            select(SnapshotRecord)
            .where(
                SnapshotRecord.run_id == run_id,
                SnapshotRecord.snapshot_type == SnapshotType.CHECKPOINT_REF.value,
            )
            .order_by(SnapshotRecord.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def save_subagent_run(
        self,
        *,
        session_id: str,
        run_id: str,
        task_id: str,
        subagent_type: str,
        status: str,
        prompt: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        return await self.append_snapshot(
            session_id=session_id,
            run_id=run_id,
            label=f"subagent:{task_id}:{status}",
            snapshot_type=SnapshotType.SUBAGENT_RUN,
            data={
                "task_id": task_id,
                "subagent_type": subagent_type,
                "status": status,
                "prompt": prompt,
                "result": result or {},
                "error": error,
            },
            created_at=created_at,
        )

    async def save_review_report(
        self,
        *,
        session_id: str,
        run_id: str,
        review_type: str,
        status: str,
        report: dict[str, Any],
        created_at: datetime | None = None,
    ) -> SnapshotRecord:
        return await self.append_snapshot(
            session_id=session_id,
            run_id=run_id,
            label=f"review:{review_type}:{status}",
            snapshot_type=SnapshotType.REVIEW_REPORT,
            data={
                "review_type": review_type,
                "status": status,
                "report": report,
            },
            created_at=created_at,
        )

    async def list_graph_timeline(
        self,
        *,
        run_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> Sequence[SnapshotRecord]:
        statement = (
            select(SnapshotRecord)
            .where(
                SnapshotRecord.run_id == run_id,
                SnapshotRecord.snapshot_type.in_([
                    SnapshotType.ROUTE_DECISION.value,
                    SnapshotType.GRAPH_SNAPSHOT.value,
                    SnapshotType.CHECKPOINT_REF.value,
                    SnapshotType.SUBAGENT_RUN.value,
                    SnapshotType.REVIEW_REPORT.value,
                ]),
            )
            .order_by(SnapshotRecord.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return result.all()

    async def complete_run(
        self,
        *,
        run_id: str,
        status: RunStatus | str = RunStatus.COMPLETED,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        completed_at: datetime | None = None,
    ) -> RunRecord | None:
        now = completed_at or utc_now()
        async with self._session_factory() as session:
            record = await session.get(RunRecord, run_id)
            if record is None:
                return None

            record.status = _value(status)
            record.error = error
            record.completed_at = now
            record.metadata_ = _merge_metadata(record.metadata_, metadata)
            await self._touch_session(session, record.session_id, now)
            await session.commit()
            return record

    def _ensure_builtin_memory_files(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        memory_file = self.memory_root / "MEMORY.md"
        user_file = self.memory_root / "USER.md"
        if not memory_file.exists():
            memory_file.write_text("# MEMORY\n\n- Solo Agent project memory.\n", encoding="utf-8")
        if not user_file.exists():
            user_file.write_text("# USER\n\n- User preferences live here.\n", encoding="utf-8")

    async def _index_builtin_memory(self, builtin: dict[str, str]) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM messages_fts WHERE source = 'builtin' AND session_id = '__builtin__'")
            )
            for name, content in builtin.items():
                await session.execute(
                    text(
                        """
                        INSERT INTO messages_fts(message_id, session_id, role, source, content)
                        VALUES (:id, '__builtin__', :role, 'builtin', :content)
                        """
                    ),
                    {"id": f"builtin:{name}", "role": f"builtin:{name}", "content": content},
                )
            await session.commit()

    async def _index_summary(self, session_id: str, summary: SnapshotRecord) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM messages_fts WHERE message_id = :id"),
                {"id": f"summary:{summary.id}"},
            )
            await session.execute(
                text(
                    """
                    INSERT INTO messages_fts(message_id, session_id, role, source, content)
                    VALUES (:id, :session_id, 'summary', 'summary', :content)
                    """
                ),
                {
                    "id": f"summary:{summary.id}",
                    "session_id": session_id,
                    "content": _snapshot_summary(summary),
                },
            )
            await session.commit()

    @staticmethod
    async def _next_message_sequence(session: AsyncSession, session_id: str) -> int:
        statement = select(func.coalesce(func.max(MessageRecord.sequence), 0)).where(
            MessageRecord.session_id == session_id
        )
        current = await session.scalar(statement)
        return int(current or 0) + 1

    @staticmethod
    async def _touch_session(session: AsyncSession, session_id: str, at: datetime) -> None:
        record = await session.get(SessionRecord, session_id)
        if record is not None:
            record.updated_at = at

    @staticmethod
    async def _touch_session_for_run(session: AsyncSession, run_id: str, at: datetime) -> None:
        run = await session.get(RunRecord, run_id)
        if run is not None:
            await SQLiteMemoryRepository._touch_session(session, run.session_id, at)


async def init_sqlite_memory(
    database_path: str,
    *,
    echo: bool = False,
    memory_root: str | Path | None = None,
) -> SQLiteMemoryRepository:
    engine = create_memory_engine(database_path, echo=echo)
    await init_database(engine)
    return SQLiteMemoryRepository.from_engine(engine, memory_root=memory_root)
