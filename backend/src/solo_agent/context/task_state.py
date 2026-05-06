from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from uuid import uuid4

TaskStatus = Literal["pending", "in_progress", "completed", "blocked", "deleted"]
TASK_STATUSES: tuple[TaskStatus, ...] = ("pending", "in_progress", "completed", "blocked", "deleted")

_TASK_JSON_RE = re.compile(r"<task-list-json>\s*(?P<body>.*?)\s*</task-list-json>", re.DOTALL | re.IGNORECASE)
_CHECKBOX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\[([ xX/>\!-])\]\s*(?P<subject>.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(?P<subject>.+?)\s*$")
_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])?\s*(?:\[(?P<bracket>[^\]]+)\]|(?P<prefix>[\w \u4e00-\u9fff]+)\s*[:：])\s*(?P<subject>.+?)\s*$",
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(r"^(?P<subject>.+?)\s*(?:\((?P<paren>[^)]+)\)|（(?P<cparen>[^）]+)）)\s*$", re.IGNORECASE)

_CHECKBOX_STATUS: dict[str, TaskStatus] = {
    " ": "pending",
    "x": "completed",
    "X": "completed",
    "/": "in_progress",
    ">": "in_progress",
    "!": "blocked",
    "-": "deleted",
}
_STATUS_ALIASES: dict[str, TaskStatus] = {
    "todo": "pending",
    "pending": "pending",
    "open": "pending",
    "待办": "pending",
    "未开始": "pending",
    "doing": "in_progress",
    "in progress": "in_progress",
    "in_progress": "in_progress",
    "active": "in_progress",
    "进行中": "in_progress",
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "finished": "completed",
    "完成": "completed",
    "已完成": "completed",
    "blocked": "blocked",
    "blocker": "blocked",
    "阻塞": "blocked",
    "卡住": "blocked",
    "deleted": "deleted",
    "removed": "deleted",
    "cancelled": "deleted",
    "canceled": "deleted",
    "删除": "deleted",
    "取消": "deleted",
}


@dataclass(slots=True)
class TaskListItem:
    """对齐 oh-my-openagent/Claude Code 风格的任务条目。"""

    subject: str
    description: str = ""
    status: TaskStatus = "pending"
    active_form: str = ""
    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    owner: str = "solo-agent"
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"T-{uuid4()}")

    @property
    def title(self) -> str:
        """兼容旧测试和旧调用方。"""

        return self.subject

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "task-system"))

    def __post_init__(self) -> None:
        self.subject = _clean_subject(self.subject)
        self.description = self.description.strip()
        self.active_form = self.active_form.strip() or _default_active_form(self.subject, self.status)
        self.status = _normalize_status(self.status) or "pending"
        self.blocked_by = [str(item) for item in self.blocked_by if str(item).strip()]
        self.blocks = [str(item) for item in self.blocks if str(item).strip()]
        if not self.id.startswith("T-"):
            self.id = f"T-{self.id}"

    def update(
        self,
        *,
        subject: str | None = None,
        description: str | None = None,
        status: str | None = None,
        active_form: str | None = None,
        blocked_by: list[str] | None = None,
        blocks: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if subject is not None:
            self.subject = _clean_subject(subject)
        if description is not None:
            self.description = description.strip()
        if status is not None:
            normalized = _normalize_status(status)
            if normalized is None:
                raise ValueError(f"Unsupported task status: {status}")
            self.status = normalized
        if active_form is not None:
            self.active_form = active_form.strip()
        if blocked_by is not None:
            self.blocked_by = [str(item) for item in blocked_by if str(item).strip()]
        if blocks is not None:
            self.blocks = [str(item) for item in blocks if str(item).strip()]
        if metadata:
            self.metadata.update(metadata)
        if not self.active_form:
            self.active_form = _default_active_form(self.subject, self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status,
            "activeForm": self.active_form,
            "active_form": self.active_form,
            "blockedBy": list(self.blocked_by),
            "blocked_by": list(self.blocked_by),
            "blocks": list(self.blocks),
            "owner": self.owner,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskListItem:
        return cls(
            id=str(raw.get("id") or f"T-{uuid4()}"),
            subject=str(raw.get("subject") or raw.get("title") or raw.get("task") or ""),
            description=str(raw.get("description") or ""),
            status=_normalize_status(str(raw.get("status") or "pending")) or "pending",
            active_form=str(raw.get("activeForm") or raw.get("active_form") or ""),
            blocked_by=_as_string_list(raw.get("blockedBy") or raw.get("blocked_by")),
            blocks=_as_string_list(raw.get("blocks")),
            owner=str(raw.get("owner") or "solo-agent"),
            metadata=dict(raw.get("metadata") or {}),
        )

    def format_line(self, index: int) -> str:
        blockers = f" blocked_by={self.blocked_by}" if self.blocked_by else ""
        return f"{index}. [{self.status}] {self.subject}{blockers}"


@dataclass(slots=True)
class TaskListState:
    items: list[TaskListItem] = field(default_factory=list)
    continue_from: str = ""
    thread_id: str = ""

    _ACTIVE_STATUSES: ClassVar[tuple[TaskStatus, ...]] = ("in_progress", "blocked", "pending")

    @classmethod
    def from_text(cls, text: str, *, thread_id: str = "") -> TaskListState:
        structured = cls.from_structured_text(text, thread_id=thread_id)
        if structured.items:
            return structured

        items: list[TaskListItem] = []
        seen: set[str] = set()
        for line in text.splitlines():
            item = _parse_line(line)
            if item is None:
                continue
            key = item.subject.casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
        state = cls(items=items, thread_id=thread_id)
        state.ensure_single_active()
        return state

    @classmethod
    def from_structured_text(cls, text: str, *, thread_id: str = "") -> TaskListState:
        match = _TASK_JSON_RE.search(text)
        if match is None:
            return cls(thread_id=thread_id)
        try:
            raw = json.loads(match.group("body"))
        except json.JSONDecodeError:
            return cls(thread_id=thread_id)
        return cls.from_payload(raw, thread_id=thread_id)

    @classmethod
    def from_payload(cls, payload: Any, *, thread_id: str = "") -> TaskListState:
        if isinstance(payload, dict):
            raw_items = payload.get("tasks") or payload.get("items") or []
            continue_from = str(payload.get("continueFrom") or payload.get("continue_from") or "")
            thread_id = str(payload.get("threadID") or payload.get("thread_id") or thread_id)
        elif isinstance(payload, list):
            raw_items = payload
            continue_from = ""
        else:
            return cls(thread_id=thread_id)

        items = [TaskListItem.from_dict(item) for item in raw_items if isinstance(item, dict)]
        state = cls(items=items, continue_from=continue_from, thread_id=thread_id)
        state.ensure_single_active()
        return state

    @classmethod
    def from_plan(cls, plan: object, *, thread_id: str = "") -> TaskListState:
        if isinstance(plan, str):
            return cls.from_text(plan, thread_id=thread_id)
        return cls.from_payload(plan, thread_id=thread_id)

    def ensure_single_active(self) -> None:
        active = [item for item in self.items if item.status == "in_progress"]
        if not active:
            for item in self.items:
                if item.status == "pending":
                    item.update(status="in_progress")
                    break
        elif len(active) > 1:
            for item in active[1:]:
                item.update(status="pending")
        self.continue_from = self.continue_from or self._infer_continue_from()

    def create(
        self,
        subject: str,
        *,
        description: str = "",
        status: TaskStatus = "pending",
        active_form: str = "",
        blocked_by: list[str] | None = None,
        blocks: list[str] | None = None,
        owner: str = "solo-agent",
        metadata: dict[str, Any] | None = None,
    ) -> TaskListItem:
        item = TaskListItem(
            subject=subject,
            description=description,
            status=status,
            active_form=active_form,
            blocked_by=blocked_by or [],
            blocks=blocks or [],
            owner=owner,
            metadata=metadata or {"source": "task_create"},
        )
        self.items.append(item)
        self.ensure_single_active()
        return item

    def get(self, task_id: str) -> TaskListItem | None:
        return next((item for item in self.items if item.id == task_id), None)

    def update(self, task_id: str, **updates: Any) -> TaskListItem:
        item = self.get(task_id)
        if item is None:
            raise KeyError(f"Task not found: {task_id}")
        item.update(**updates)
        self.ensure_single_active()
        return item

    def active_items(self) -> list[TaskListItem]:
        return [item for item in self.items if item.status != "deleted"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threadID": self.thread_id,
            "thread_id": self.thread_id,
            "continueFrom": self.continue_from or self._infer_continue_from(),
            "continue_from": self.continue_from or self._infer_continue_from(),
            "tasks": [item.to_dict() for item in self.items],
            "items": [item.to_dict() for item in self.items],
        }

    def format_block(self) -> str:
        lines = [
            "<task-state>",
            "[System note: Structured TaskList state, NOT new user input.]",
            f"Thread: {self.thread_id or '(current session)'}",
            f"Continue from: {self.continue_from or self._infer_continue_from() or '(no active task)'}",
            "",
            "Tasks:",
        ]
        if not self.items:
            lines.append("- (no tasks detected)")
        else:
            lines.extend(item.format_line(index) for index, item in enumerate(self.items, start=1))
        lines.append("</task-state>")
        return "\n".join(lines)

    def format_json_block(self) -> str:
        return "<task-list-json>\n" + json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n</task-list-json>"

    def _infer_continue_from(self) -> str:
        for status in self._ACTIVE_STATUSES:
            for item in self.items:
                if item.status == status:
                    return item.subject
        return ""


def task_planner_instruction() -> str:
    return (
        "Also include a machine-readable <task-list-json> block with this schema: "
        "{tasks:[{subject, description, status, activeForm, blockedBy, blocks, owner, metadata}], continueFrom}. "
        "Use imperative subjects, present-continuous activeForm, and exactly one in_progress task."
    )


def _parse_line(line: str) -> TaskListItem | None:
    stripped = line.strip()
    if not stripped:
        return None
    checkbox = _CHECKBOX_RE.match(stripped)
    if checkbox:
        status = _CHECKBOX_STATUS[checkbox.group(1)]
        return TaskListItem(subject=checkbox.group("subject"), status=status, metadata={"source": "checkbox"})
    prefix = _PREFIX_RE.match(stripped)
    if prefix:
        status = _normalize_status(prefix.group("bracket") or prefix.group("prefix"))
        if status is not None:
            return TaskListItem(subject=prefix.group("subject"), status=status, metadata={"source": "status-prefix"})
    suffix = _SUFFIX_RE.match(stripped)
    if suffix:
        status = _normalize_status(suffix.group("paren") or suffix.group("cparen"))
        if status is not None:
            return TaskListItem(subject=suffix.group("subject"), status=status, metadata={"source": "status-suffix"})
    bullet = _BULLET_RE.match(stripped)
    if bullet:
        return TaskListItem(subject=bullet.group("subject"), status="pending", metadata={"source": "bullet"})
    return None


def _normalize_status(value: str | None) -> TaskStatus | None:
    if value is None:
        return None
    key = " ".join(str(value).strip().lower().replace("_", " ").split())
    normalized = _STATUS_ALIASES.get(key)
    if normalized in TASK_STATUSES:
        return normalized
    if key.replace(" ", "_") in TASK_STATUSES:
        return key.replace(" ", "_")  # type: ignore[return-value]
    return None


def _clean_subject(subject: str) -> str:
    return re.sub(r"\s+", " ", subject).strip(" \t-:：")


def _default_active_form(subject: str, status: str) -> str:
    if status == "completed":
        return f"Completed: {subject}"
    if status == "blocked":
        return f"Unblocking: {subject}"
    if status == "deleted":
        return f"Deleted: {subject}"
    return f"Working on: {subject}"


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]
