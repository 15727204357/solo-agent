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
from .models import (
    MessageRecord,
    MessageRole,
    RunRecord,
    RunStatus,
    SessionRecord,
    SessionType,
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
    ) -> None:
        self._session_factory = session_factory
        self.memory_root = Path(memory_root or Path.cwd()).resolve()
        self._prefetch_cache: dict[tuple[str, str], dict[str, Any]] = {}

    @classmethod
    def from_engine(
        cls,
        engine: AsyncEngine,
        *,
        memory_root: str | Path | None = None,
    ) -> SQLiteMemoryRepository:
        return cls(create_session_factory(engine), memory_root=memory_root)

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
    ) -> dict[str, Any]:
        builtin = await self.load_builtin_memory()
        insights = _extract_preference_lines(payload)
        warnings: list[str] = []
        if insights:
            user_file = self.memory_root / "USER.md"
            existing = builtin.get("user", "")
            additions = "\n".join(f"- {line}" for line in insights if line not in existing)
            if additions:
                try:
                    user_file.write_text((existing.rstrip() + "\n" + additions + "\n").lstrip(), encoding="utf-8")
                    await self._index_builtin_memory(await self.load_builtin_memory())
                except OSError as exc:
                    warnings.append(str(exc))
        return {"insights": insights, "session_id": session_id, "warnings": warnings}

    async def load_builtin_memory(self) -> dict[str, str]:
        self._ensure_builtin_memory_files()
        return {
            "memory": (self.memory_root / "MEMORY.md").read_text(encoding="utf-8"),
            "user": (self.memory_root / "USER.md").read_text(encoding="utf-8"),
        }

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
