"""Unified workspace backend contract for production coding workflows."""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from solo_agent.workflow.sandbox.command_workspace import (
    CommandWorkspace,
    build_workspace_manifest,
    diff_manifests,
    prepare_command_workspace,
)

WorkspaceBackendKind = Literal["local", "copy", "docker"]


@dataclass(frozen=True)
class WorkspaceBackendMetadata:
    kind: WorkspaceBackendKind
    workspace_root: str
    command_workspace_root: str
    sandbox_id: str = ""
    isolated: bool = False
    available: bool = True
    reason: str = ""
    network_policy: str = "deny"
    resource_limits: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "workspace_root": self.workspace_root,
            "command_workspace_root": self.command_workspace_root,
            "sandbox_id": self.sandbox_id,
            "isolated": self.isolated,
            "available": self.available,
            "reason": self.reason,
            "network_policy": self.network_policy,
            "resource_limits": dict(self.resource_limits or {}),
        }


class WorkspaceBackend(ABC):
    """Backend abstraction inspired by production coding-agent workspace layers."""

    kind: WorkspaceBackendKind

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        session_id: str = "session",
        run_id: str = "run",
        network_policy: str = "deny",
        resource_limits: dict[str, Any] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id
        self.run_id = run_id
        self.network_policy = network_policy
        self.resource_limits = dict(resource_limits or {})
        self.command_workspace: CommandWorkspace | None = None

    @abstractmethod
    def prepare(self) -> CommandWorkspace:
        """Prepare the backend and return a command workspace."""

    def metadata(self) -> dict[str, Any]:
        workspace = self.command_workspace
        command_root = workspace.command_workspace_root if workspace else self.workspace_root
        return WorkspaceBackendMetadata(
            kind=self.kind,
            workspace_root=str(self.workspace_root),
            command_workspace_root=str(command_root),
            sandbox_id=workspace.sandbox_id if workspace else "",
            isolated=bool(workspace and workspace.created),
            network_policy=self.network_policy,
            resource_limits=self.resource_limits,
        ).to_dict()

    def manifest(self) -> dict[str, Any]:
        workspace = self._require_workspace()
        return build_workspace_manifest(workspace.command_workspace_root)

    def diff(self) -> dict[str, Any]:
        workspace = self._require_workspace()
        before = {}
        if workspace.baseline_manifest_path and workspace.baseline_manifest_path.exists():
            import json

            before = json.loads(workspace.baseline_manifest_path.read_text(encoding="utf-8")).get("files", {})
        after = build_workspace_manifest(workspace.command_workspace_root).get("files", {})
        return diff_manifests(before, after)

    def cleanup(self) -> dict[str, Any]:
        workspace = self._require_workspace()
        return workspace.cleanup()

    def resume(self) -> dict[str, Any]:
        workspace = self._require_workspace()
        return {**workspace.metadata(), "resume": "available"}

    def _require_workspace(self) -> CommandWorkspace:
        if self.command_workspace is None:
            raise RuntimeError("Workspace backend has not been prepared")
        return self.command_workspace


class LocalWorkspaceBackend(WorkspaceBackend):
    kind: WorkspaceBackendKind = "local"

    def prepare(self) -> CommandWorkspace:
        self.command_workspace = prepare_command_workspace(
            self.workspace_root,
            session_id=self.session_id,
            run_id=self.run_id,
            sandbox_mode="local",
            network_policy=self.network_policy,
        )
        return self.command_workspace


class CopyWorkspaceBackend(WorkspaceBackend):
    kind: WorkspaceBackendKind = "copy"

    def prepare(self) -> CommandWorkspace:
        self.command_workspace = prepare_command_workspace(
            self.workspace_root,
            session_id=self.session_id,
            run_id=self.run_id,
            sandbox_mode="copy",
            network_policy=self.network_policy,
        )
        return self.command_workspace


class DockerWorkspaceBackend(WorkspaceBackend):
    kind: WorkspaceBackendKind = "docker"

    def prepare(self) -> CommandWorkspace:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker workspace backend requested but docker is not available on PATH")
        completed = subprocess.run(["docker", "version"], capture_output=True, text=True, timeout=10, check=False)
        if completed.returncode != 0:
            raise RuntimeError("Docker workspace backend requested but docker is not reachable")
        self.command_workspace = prepare_command_workspace(
            self.workspace_root,
            session_id=self.session_id,
            run_id=self.run_id,
            sandbox_mode="copy",
            network_policy=self.network_policy,
        )
        return self.command_workspace


def create_workspace_backend(
    kind: str,
    workspace_root: str | Path,
    *,
    session_id: str = "session",
    run_id: str = "run",
    network_policy: str = "deny",
    resource_limits: dict[str, Any] | None = None,
) -> WorkspaceBackend:
    normalized = str(kind or "copy").strip().lower()
    backend_cls: type[WorkspaceBackend]
    if normalized == "local":
        backend_cls = LocalWorkspaceBackend
    elif normalized == "docker":
        backend_cls = DockerWorkspaceBackend
    else:
        backend_cls = CopyWorkspaceBackend
    return backend_cls(
        workspace_root,
        session_id=session_id,
        run_id=run_id,
        network_policy=network_policy,
        resource_limits=resource_limits,
    )

