from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionType(StrEnum):
    """Stable session categories; future values can mirror SQLite v11 SessionType."""

    CHAT = "chat"
    PROJECT = "project"
    CHECKPOINT = "checkpoint"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class SnapshotType(StrEnum):
    CHECKPOINT = "checkpoint"
    CONTEXT = "context"
    SUMMARY = "summary"


class Base(DeclarativeBase):
    pass


class SessionRecord(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    session_type: Mapped[str] = mapped_column(String(64), default=SessionType.CHAT.value, index=True)
    workspace_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    runs: Mapped[list[RunRecord]] = relationship(back_populates="session", cascade="all, delete-orphan")
    messages: Mapped[list[MessageRecord]] = relationship(back_populates="session", cascade="all, delete-orphan")
    snapshots: Mapped[list[SnapshotRecord]] = relationship(back_populates="session", cascade="all, delete-orphan")


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default=RunStatus.RUNNING.value, index=True)
    provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[SessionRecord] = relationship(back_populates="runs")
    messages: Mapped[list[MessageRecord]] = relationship(back_populates="run")
    tool_calls: Mapped[list[ToolCallRecord]] = relationship(back_populates="run", cascade="all, delete-orphan")
    timing_points: Mapped[list[TimingPointRecord]] = relationship(back_populates="run", cascade="all, delete-orphan")
    snapshots: Mapped[list[SnapshotRecord]] = relationship(back_populates="run")


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    session: Mapped[SessionRecord] = relationship(back_populates="messages")
    run: Mapped[RunRecord | None] = relationship(back_populates="messages")
    tool_calls: Mapped[list[ToolCallRecord]] = relationship(back_populates="message")


class ToolCallRecord(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=ToolCallStatus.RUNNING.value, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[RunRecord] = relationship(back_populates="tool_calls")
    message: Mapped[MessageRecord | None] = relationship(back_populates="tool_calls")


class TimingPointRecord(Base):
    __tablename__ = "timing_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(255), index=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    offset_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    run: Mapped[RunRecord] = relationship(back_populates="timing_points")


class SnapshotRecord(Base):
    __tablename__ = "snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"), nullable=True, index=True)
    snapshot_type: Mapped[str] = mapped_column(String(64), default=SnapshotType.CHECKPOINT.value, index=True)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    session: Mapped[SessionRecord] = relationship(back_populates="snapshots")
    run: Mapped[RunRecord | None] = relationship(back_populates="snapshots")


Index("ix_messages_session_sequence", MessageRecord.session_id, MessageRecord.sequence)
Index("ix_tool_calls_run_started", ToolCallRecord.run_id, ToolCallRecord.started_at)
Index("ix_timing_points_run_created", TimingPointRecord.run_id, TimingPointRecord.created_at)
Index("ix_snapshots_session_created", SnapshotRecord.session_id, SnapshotRecord.created_at)
