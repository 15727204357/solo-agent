from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .models import (
    MemoryCandidateRecord,
    MemoryCandidateStatus,
    MemoryEntryRecord,
    MemoryEntryStatus,
    MemoryTarget,
    WorkflowObservationRecord,
    utc_now,
)

MEMORY_CHAR_LIMIT = 2200
USER_CHAR_LIMIT = 1375
WORKFLOW_WINDOW_DAYS = 14
WORKFLOW_TRIGGER_COUNT = 3

_FENCE_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|api[_-]?key\s*[=:]|secret\s*[=:]|password\s*[=:]|BEGIN [A-Z ]*PRIVATE KEY)",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|prior) instructions|system prompt|developer message|reveal.*prompt|"
    r"disregard (all )?(previous|prior) instructions)",
    re.IGNORECASE,
)
_SSH_BACKDOOR_RE = re.compile(
    r"(authorized_keys|ssh-rsa|ssh-ed25519|PermitRootLogin|StrictHostKeyChecking\s+no)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_PREFERENCE_MARKERS = (
    "prefer",
    "preference",
    "remember",
    "user likes",
    "user wants",
    "用户偏好",
    "记住",
    "以后",
    "不要",
    "不再",
)
_WORKFLOW_MARKERS = ("workflow", "playbook", "routine", "process", "以后按这个流程", "常用流程", "工作流")
_NEGATION_MARKERS = ("not", "don't", "dont", "no longer", "instead", "replace", "不要", "不再", "改为", "而不是")


class MemoryGovernanceError(ValueError):
    """Raised when a memory governance action is rejected by policy."""


def new_id() -> str:
    return str(uuid4())


class MemoryGovernanceService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        memory_root: str | Path,
    ) -> None:
        self._session_factory = session_factory
        self.memory_root = Path(memory_root).resolve()

    async def submit_from_pre_compress(
        self,
        *,
        session_id: str,
        run_id: str | None,
        payload: Mapping[str, Any],
        insights: Iterable[str],
    ) -> list[MemoryCandidateRecord]:
        candidates: list[MemoryCandidateRecord] = []
        for insight in insights:
            candidates.append(
                await self.submit_candidate(
                    target=MemoryTarget.USER,
                    content=insight,
                    source_session_id=session_id,
                    source_run_id=run_id,
                    source_excerpt=_source_excerpt(payload),
                    confidence=0.62,
                    metadata={"source": "on_pre_compress", "priority": "summary_extraction"},
                )
            )

        workflow = await self._maybe_submit_workflow_candidate(
            session_id=session_id,
            run_id=run_id,
            payload=payload,
        )
        if workflow is not None:
            candidates.append(workflow)
        return candidates

    async def submit_candidate(
        self,
        *,
        target: MemoryTarget | str,
        content: str,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        source_message_id: str | None = None,
        source_excerpt: str | None = None,
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidateRecord:
        target_value = _target_value(target)
        flags = _scan_safety(str(content))
        clean = _clean_candidate_content(content)
        now = utc_now()
        async with self._session_factory() as session:
            active_entries = await self._active_entries(session, target_value)
            duplicate = _exact_duplicate(clean, active_entries)
            conflict_ids = _conflict_ids(clean, active_entries)
            status = MemoryCandidateStatus.PENDING.value
            duplicate_of_id = None
            if flags:
                status = MemoryCandidateStatus.BLOCKED.value
            elif duplicate is not None:
                status = MemoryCandidateStatus.DUPLICATE.value
                duplicate_of_id = duplicate.id
            elif conflict_ids:
                flags.append("conflict")

            record = MemoryCandidateRecord(
                id=new_id(),
                target=target_value,
                content=clean,
                source_session_id=source_session_id,
                source_run_id=source_run_id,
                source_message_id=source_message_id,
                source_excerpt=source_excerpt,
                confidence=max(0.0, min(float(confidence), 1.0)),
                status=status,
                duplicate_of_id=duplicate_of_id,
                conflict_ids=conflict_ids,
                safety_flags=flags,
                metadata_=metadata or {},
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            await session.commit()
            return record

    async def list_candidates(
        self,
        *,
        status: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[MemoryCandidateRecord]:
        statement = select(MemoryCandidateRecord)
        if status:
            statement = statement.where(MemoryCandidateRecord.status == status)
        if target:
            statement = statement.where(MemoryCandidateRecord.target == target)
        statement = statement.order_by(MemoryCandidateRecord.created_at.desc()).limit(limit)
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return list(result.all())

    async def update_candidate(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        target: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryCandidateRecord | None:
        async with self._session_factory() as session:
            record = await session.get(MemoryCandidateRecord, candidate_id)
            if record is None:
                return None
            if record.status not in {MemoryCandidateStatus.PENDING.value, MemoryCandidateStatus.BLOCKED.value}:
                raise MemoryGovernanceError("Only pending or blocked candidates can be edited")
            if content is not None:
                raw_content = str(content)
                record.content = _clean_candidate_content(raw_content)
            if target is not None:
                record.target = _target_value(target)
            if confidence is not None:
                record.confidence = max(0.0, min(float(confidence), 1.0))
            if metadata:
                merged = dict(record.metadata_ or {})
                merged.update(metadata)
                record.metadata_ = merged

            active_entries = await self._active_entries(session, record.target)
            flags = _scan_safety(raw_content if content is not None else record.content)
            duplicate = _exact_duplicate(record.content, active_entries)
            record.conflict_ids = _conflict_ids(record.content, active_entries)
            if record.conflict_ids and "conflict" not in flags:
                flags.append("conflict")
            record.safety_flags = flags
            record.duplicate_of_id = duplicate.id if duplicate is not None else None
            if flags and any(flag != "conflict" for flag in flags):
                record.status = MemoryCandidateStatus.BLOCKED.value
            elif duplicate is not None:
                record.status = MemoryCandidateStatus.DUPLICATE.value
            else:
                record.status = MemoryCandidateStatus.PENDING.value
            record.updated_at = utc_now()
            await session.commit()
            return record

    async def approve_candidate(
        self,
        candidate_id: str,
        *,
        resolution: str = "add",
        content: str | None = None,
    ) -> tuple[MemoryCandidateRecord, MemoryEntryRecord]:
        resolution = resolution.lower().strip()
        if resolution not in {"add", "replace", "merge"}:
            raise MemoryGovernanceError("resolution must be add, replace, or merge")

        async with self._session_factory() as session:
            candidate = await session.get(MemoryCandidateRecord, candidate_id)
            if candidate is None:
                raise MemoryGovernanceError("Memory candidate not found")
            if candidate.status != MemoryCandidateStatus.PENDING.value:
                raise MemoryGovernanceError(f"Candidate is not pending: {candidate.status}")
            blocking_flags = [flag for flag in candidate.safety_flags or [] if flag != "conflict"]
            if blocking_flags:
                raise MemoryGovernanceError("Candidate is blocked by safety scan")
            if candidate.conflict_ids and resolution == "add":
                raise MemoryGovernanceError("conflict_requires_resolution")

            entry_content = _clean_candidate_content(content or candidate.content)
            self._check_capacity(candidate.target, entry_content, resolution)
            now = utc_now()
            supersedes_id = None
            if resolution in {"replace", "merge"}:
                supersedes_id = await self._supersede_conflicts(session, candidate.conflict_ids, now)

            entry = MemoryEntryRecord(
                id=new_id(),
                target=candidate.target,
                content=entry_content,
                source_candidate_id=candidate.id,
                confidence=candidate.confidence,
                supersedes_id=supersedes_id,
                status=MemoryEntryStatus.ACTIVE.value,
                metadata_={"resolution": resolution},
                created_at=now,
                updated_at=now,
            )
            candidate.content = entry_content
            candidate.status = MemoryCandidateStatus.APPROVED.value
            candidate.updated_at = now
            candidate.decided_at = now
            session.add(entry)
            await session.commit()
            entry_id = entry.id

        await self.materialize_published_memory()
        async with self._session_factory() as session:
            candidate = await session.get(MemoryCandidateRecord, candidate_id)
            entry = await session.get(MemoryEntryRecord, entry_id)
            if candidate is None or entry is None:
                raise MemoryGovernanceError("Approved memory could not be reloaded")
            return candidate, entry

    async def reject_candidate(self, candidate_id: str, *, reason: str | None = None) -> MemoryCandidateRecord | None:
        async with self._session_factory() as session:
            record = await session.get(MemoryCandidateRecord, candidate_id)
            if record is None:
                return None
            record.status = MemoryCandidateStatus.REJECTED.value
            record.decided_at = utc_now()
            record.updated_at = record.decided_at
            metadata = dict(record.metadata_ or {})
            if reason:
                metadata["rejection_reason"] = reason
            record.metadata_ = metadata
            await session.commit()
            return record

    async def list_entries(
        self,
        *,
        target: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[MemoryEntryRecord]:
        statement = select(MemoryEntryRecord)
        if target:
            statement = statement.where(MemoryEntryRecord.target == target)
        if not include_inactive:
            statement = statement.where(MemoryEntryRecord.status == MemoryEntryStatus.ACTIVE.value)
        statement = statement.order_by(MemoryEntryRecord.updated_at.desc()).limit(limit)
        async with self._session_factory() as session:
            result = await session.scalars(statement)
            return list(result.all())

    async def revoke_entry(self, entry_id: str, *, reason: str | None = None) -> MemoryEntryRecord | None:
        async with self._session_factory() as session:
            record = await session.get(MemoryEntryRecord, entry_id)
            if record is None:
                return None
            now = utc_now()
            record.status = MemoryEntryStatus.REVOKED.value
            record.revoked_at = now
            record.updated_at = now
            metadata = dict(record.metadata_ or {})
            if reason:
                metadata["revocation_reason"] = reason
            record.metadata_ = metadata
            await session.commit()

        await self.materialize_published_memory()
        async with self._session_factory() as session:
            return await session.get(MemoryEntryRecord, entry_id)

    async def ensure_seeded_builtin_entries(self) -> None:
        self._ensure_builtin_memory_files()
        async with self._session_factory() as session:
            count = await session.scalar(select(func.count(MemoryEntryRecord.id)))
            if int(count or 0) > 0:
                return
            now = utc_now()
            for target, path in (
                (MemoryTarget.MEMORY.value, self.memory_root / "MEMORY.md"),
                (MemoryTarget.USER.value, self.memory_root / "USER.md"),
            ):
                for entry in _parse_memory_file(path):
                    session.add(
                        MemoryEntryRecord(
                            id=new_id(),
                            target=target,
                            content=entry,
                            confidence=1.0,
                            status=MemoryEntryStatus.ACTIVE.value,
                            metadata_={"source": "legacy_memory_file"},
                            created_at=now,
                            updated_at=now,
                        )
                    )
            await session.commit()
        await self.materialize_published_memory()

    async def materialize_published_memory(self) -> None:
        self._ensure_builtin_memory_files()
        entries = await self.list_entries(include_inactive=False, limit=1000)
        grouped: dict[str, list[str]] = {MemoryTarget.MEMORY.value: [], MemoryTarget.USER.value: []}
        for entry in sorted(entries, key=lambda item: item.created_at):
            if entry.target in grouped:
                grouped[entry.target].append(entry.content)
            elif entry.target == MemoryTarget.SKILL.value:
                self._write_skill(entry)

        (self.memory_root / "MEMORY.md").write_text(_format_memory_file("MEMORY", grouped["memory"]), encoding="utf-8")
        (self.memory_root / "USER.md").write_text(_format_memory_file("USER", grouped["user"]), encoding="utf-8")

    async def _maybe_submit_workflow_candidate(
        self,
        *,
        session_id: str,
        run_id: str | None,
        payload: Mapping[str, Any],
    ) -> MemoryCandidateRecord | None:
        text = _flatten(payload)
        lowered = text.lower()
        if not any(marker.lower() in lowered for marker in _WORKFLOW_MARKERS):
            return None

        signature = _workflow_signature(text)
        explicit = "以后按这个流程" in text or "remember this workflow" in lowered or "use this workflow" in lowered
        now = utc_now()
        async with self._session_factory() as session:
            session.add(
                WorkflowObservationRecord(
                    id=new_id(),
                    signature=signature,
                    title=_workflow_title(text),
                    source_session_id=session_id,
                    source_run_id=run_id,
                    details={"excerpt": text[:500]},
                    created_at=now,
                )
            )
            window_start = now - timedelta(days=WORKFLOW_WINDOW_DAYS)
            count = await session.scalar(
                select(func.count(WorkflowObservationRecord.id)).where(
                    WorkflowObservationRecord.signature == signature,
                    WorkflowObservationRecord.created_at >= window_start,
                )
            )
            await session.commit()

        if not explicit and int(count or 0) < WORKFLOW_TRIGGER_COUNT:
            return None
        draft = _skill_draft(signature, text)
        return await self.submit_candidate(
            target=MemoryTarget.SKILL,
            content=draft,
            source_session_id=session_id,
            source_run_id=run_id,
            source_excerpt=text[:500],
            confidence=0.7 if explicit else 0.58,
            metadata={"source": "workflow_observation", "signature": signature, "observation_count": int(count or 0)},
        )

    async def _active_entries(self, session: AsyncSession, target: str) -> list[MemoryEntryRecord]:
        result = await session.scalars(
            select(MemoryEntryRecord).where(
                MemoryEntryRecord.target == target,
                MemoryEntryRecord.status == MemoryEntryStatus.ACTIVE.value,
            )
        )
        return list(result.all())

    async def _supersede_conflicts(self, session: AsyncSession, conflict_ids: list[str], at: Any) -> str | None:
        supersedes_id = None
        for entry_id in conflict_ids:
            entry = await session.get(MemoryEntryRecord, entry_id)
            if entry is None or entry.status != MemoryEntryStatus.ACTIVE.value:
                continue
            entry.status = MemoryEntryStatus.SUPERSEDED.value
            entry.updated_at = at
            supersedes_id = supersedes_id or entry.id
        return supersedes_id

    def _check_capacity(self, target: str, content: str, resolution: str) -> None:
        if target == MemoryTarget.SKILL.value:
            return
        limit = MEMORY_CHAR_LIMIT if target == MemoryTarget.MEMORY.value else USER_CHAR_LIMIT
        if len(content) > limit:
            raise MemoryGovernanceError(f"{target} memory entry exceeds {limit} characters")

    def _ensure_builtin_memory_files(self) -> None:
        self.memory_root.mkdir(parents=True, exist_ok=True)
        memory_file = self.memory_root / "MEMORY.md"
        user_file = self.memory_root / "USER.md"
        if not memory_file.exists():
            memory_file.write_text("# MEMORY\n", encoding="utf-8")
        if not user_file.exists():
            user_file.write_text("# USER\n", encoding="utf-8")

    def _write_skill(self, entry: MemoryEntryRecord) -> None:
        slug = str((entry.metadata_ or {}).get("slug") or _slugify(entry.content))
        skill_dir = self.memory_root / "skills" / "workflows" / slug
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(entry.content.rstrip() + "\n", encoding="utf-8")


def _target_value(target: MemoryTarget | str) -> str:
    value = target.value if isinstance(target, MemoryTarget) else str(target)
    if value not in {item.value for item in MemoryTarget}:
        raise MemoryGovernanceError(f"Unsupported memory target: {value}")
    return value


def _clean_candidate_content(content: str) -> str:
    return _FENCE_RE.sub("", str(content)).strip(" \t\r\n-")


def _scan_safety(content: str) -> list[str]:
    flags: list[str] = []
    if _FENCE_RE.search(content):
        flags.append("memory_fence_escape")
    if _INVISIBLE_RE.search(content):
        flags.append("invisible_unicode")
    if _SECRET_RE.search(content):
        flags.append("secret_like_content")
    if _INJECTION_RE.search(content):
        flags.append("prompt_injection")
    if _SSH_BACKDOOR_RE.search(content):
        flags.append("ssh_backdoor_pattern")
    if len(content) > 2200 or content.count("\n") > 60:
        flags.append("raw_dump_too_large")
    return flags


def _exact_duplicate(content: str, entries: Iterable[MemoryEntryRecord]) -> MemoryEntryRecord | None:
    normalized = _normalize(content)
    for entry in entries:
        if _normalize(entry.content) == normalized:
            return entry
    return None


def _conflict_ids(content: str, entries: Iterable[MemoryEntryRecord]) -> list[str]:
    lowered = content.lower()
    wants_replace = any(marker in lowered for marker in _NEGATION_MARKERS)
    conflicts: list[str] = []
    for entry in entries:
        similarity = _similarity(content, entry.content)
        if similarity >= 0.62 or (wants_replace and similarity >= 0.28):
            conflicts.append(entry.id)
    return conflicts


def _similarity(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(value) if len(token) > 1]


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def _flatten(value: Any) -> str:
    if isinstance(value, Mapping):
        return "\n".join(_flatten(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return "\n".join(_flatten(item) for item in value)
    return str(value)


def _source_excerpt(payload: Mapping[str, Any]) -> str:
    return _flatten(payload)[:500]


def _parse_memory_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        entries.append(clean.strip("- ").strip())
    return entries


def _format_memory_file(title: str, entries: list[str]) -> str:
    body = "\n".join(f"- {entry}" for entry in entries)
    return f"# {title}\n\n{body}\n" if body else f"# {title}\n"


def _workflow_signature(text: str) -> str:
    tokens = _tokens(text)
    meaningful = [token for token in tokens if token not in {"workflow", "process", "routine", "please"}]
    return "-".join(meaningful[:12]) or "workflow"


def _workflow_title(text: str) -> str:
    first = next((line.strip(" -") for line in text.splitlines() if line.strip()), "Workflow")
    return first[:120]


def _skill_draft(signature: str, text: str) -> str:
    title = _workflow_title(text)
    return (
        f"# {title}\n\n"
        "## Trigger\n"
        f"Use this skill when the task matches this workflow signature: `{signature}`.\n\n"
        "## Steps\n"
        "1. Confirm the current task matches the remembered workflow.\n"
        "2. Reuse the established sequence from the source conversation.\n"
        "3. Adapt paths, commands, and verification to the current repository.\n\n"
        "## Verification\n"
        "Run the smallest relevant checks for the files or behavior touched by the workflow.\n\n"
        "## Not Applicable\n"
        "Do not use this skill when the user asks for a different process or the repository conventions conflict.\n\n"
        "## Source Notes\n"
        f"{text[:900].strip()}\n"
    )


def _slugify(content: str) -> str:
    tokens = _tokens(content)
    slug = "-".join(tokens[:6]).strip("-")
    return slug[:80] or "workflow-skill"
