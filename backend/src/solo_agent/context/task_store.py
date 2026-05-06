from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .task_state import TaskListState


class WorkspaceTaskStore:
    """工作区内的轻量任务状态存储，便于 Web 端和工具层共享。"""

    def __init__(self, workspace_root: str | Path, *, directory: str = ".solo-agent/tasks") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.directory = (self.workspace_root / directory).resolve()
        if not self.directory.is_relative_to(self.workspace_root):
            raise PermissionError("Task store directory must stay inside workspace")

    def load(self, thread_id: str) -> TaskListState:
        path = self._path(thread_id)
        if not path.exists():
            return TaskListState(thread_id=thread_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return TaskListState(thread_id=thread_id)
        return TaskListState.from_payload(payload, thread_id=thread_id)

    def save(self, state: TaskListState) -> TaskListState:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(state.thread_id or "default")
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return state

    def create_task(self, thread_id: str, **kwargs: Any) -> dict[str, Any]:
        state = self.load(thread_id)
        item = state.create(**kwargs)
        self.save(state)
        return {"task": item.to_dict(), "state": state.to_dict()}

    def get_task(self, thread_id: str, task_id: str) -> dict[str, Any]:
        state = self.load(thread_id)
        item = state.get(task_id)
        if item is None:
            raise KeyError(f"Task not found: {task_id}")
        return {"task": item.to_dict()}

    def list_tasks(self, thread_id: str, include_deleted: bool = False) -> dict[str, Any]:
        state = self.load(thread_id)
        tasks = state.items if include_deleted else state.active_items()
        return {"tasks": [item.to_dict() for item in tasks], "state": state.to_dict()}

    def update_task(self, thread_id: str, task_id: str, **updates: Any) -> dict[str, Any]:
        state = self.load(thread_id)
        item = state.update(task_id, **updates)
        self.save(state)
        return {"task": item.to_dict(), "state": state.to_dict()}

    def replace_state(self, thread_id: str, state: TaskListState) -> dict[str, Any]:
        state.thread_id = state.thread_id or thread_id
        self.save(state)
        return {"state": state.to_dict()}

    def _path(self, thread_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in thread_id)[:120]
        return self.directory / f"{safe or 'default'}.json"
