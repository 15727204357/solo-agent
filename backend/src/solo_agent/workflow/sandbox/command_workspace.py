"""Per-run command workspace helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SANDBOX_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".uv-cache",
    ".solo-agent",
    "dist",
    "build",
    "target",
}

MANIFEST_NAME = "baseline_manifest.json"


@dataclass(frozen=True)
class CommandWorkspace:
    mode: str
    workspace_root: Path
    command_workspace_root: Path
    created: bool = False
    sandbox_id: str = ""
    sandbox_root: Path | None = None
    cache_root: Path | None = None
    baseline_manifest_path: Path | None = None
    baseline_commit: str | None = None
    dirty_overlay: tuple[str, ...] = field(default_factory=tuple)
    network_policy: str = "deny"
    env_policy: str = "minimal"

    def metadata(self) -> dict[str, object]:
        return {
            "sandbox_id": self.sandbox_id,
            "mode": self.mode,
            "backend": "copy" if self.mode == "isolated" else self.mode,
            "workspace_root": str(self.workspace_root),
            "command_workspace_root": str(self.command_workspace_root),
            "sandbox_root": str(self.sandbox_root or self.command_workspace_root.parent),
            "cache_root": str(self.cache_root or (self.workspace_root / ".solo-agent" / "cache")),
            "baseline_manifest_path": str(self.baseline_manifest_path or ""),
            "baseline_commit": self.baseline_commit,
            "dirty_overlay": list(self.dirty_overlay),
            "network_policy": self.network_policy,
            "env_policy": self.env_policy,
            "created": self.created,
        }

    def create_checkpoint(self, label: str) -> dict[str, object]:
        checkpoint_dir = self._checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{_safe_segment(label)}.json"
        manifest = build_workspace_manifest(self.command_workspace_root)
        baseline = _read_json(self.baseline_manifest_path)
        checkpoint = {
            "label": label,
            "sandbox": self.metadata(),
            "manifest": manifest,
            "diff_summary": diff_manifests(baseline.get("files", {}), manifest.get("files", {})),
        }
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "label": label,
            "checkpoint_path": str(checkpoint_path),
            "changed_files": checkpoint["diff_summary"]["changed_files"],
            "new_files": checkpoint["diff_summary"]["new_files"],
            "deleted_files": checkpoint["diff_summary"]["deleted_files"],
        }

    def rollback_to_checkpoint(self, label: str) -> dict[str, object]:
        checkpoint_path = self._checkpoint_dir() / f"{_safe_segment(label)}.json"
        checkpoint = _read_json(checkpoint_path)
        files = checkpoint.get("manifest", {}).get("files", {})
        if not isinstance(files, dict):
            raise ValueError(f"Invalid sandbox checkpoint: {label}")
        current = build_workspace_manifest(self.command_workspace_root).get("files", {})
        for rel in set(current) - set(files):
            target = (self.command_workspace_root / rel).resolve()
            if _is_relative_to(target, self.command_workspace_root) and target.is_file():
                target.unlink()
        for rel, item in files.items():
            if not isinstance(item, dict):
                continue
            source = self.command_workspace_root / rel
            if item.get("type") == "file":
                source.parent.mkdir(parents=True, exist_ok=True)
                content = item.get("content")
                if isinstance(content, str):
                    source.write_text(content, encoding="utf-8")
        return {"label": label, "rollback": "completed", "checkpoint_path": str(checkpoint_path)}

    def cleanup(self) -> dict[str, object]:
        if not self.created:
            return {**self.metadata(), "cleanup": "not_needed"}
        sandboxes_root = (self.workspace_root / ".solo-agent" / "sandboxes").resolve()
        target = (self.sandbox_root or self.command_workspace_root.parent).resolve()
        try:
            target.relative_to(sandboxes_root)
        except ValueError:
            return {**self.metadata(), "cleanup": "skipped_outside_sandbox_root"}
        if self.mode == "worktree":
            completed = _run_git(self.workspace_root, ["worktree", "remove", "--force", str(self.command_workspace_root)])
            if completed.returncode == 0:
                shutil.rmtree(target, ignore_errors=True)
                return {**self.metadata(), "cleanup": "completed"}
        shutil.rmtree(target, ignore_errors=True)
        return {**self.metadata(), "cleanup": "completed"}

    def _checkpoint_dir(self) -> Path:
        return (self.sandbox_root or self.command_workspace_root.parent) / "checkpoints"


class SandboxManager:
    """Create lightweight production-explainable command workspaces."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        session_id: str,
        run_id: str,
        sandbox_mode: str = "auto",
        network_policy: str = "deny",
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id
        self.run_id = run_id
        self.sandbox_mode = _normalize_mode(sandbox_mode)
        self.network_policy = network_policy if network_policy in {"deny", "allow_local", "allow_all"} else "deny"
        self.sandbox_id = f"{_safe_segment(session_id)}-{_safe_segment(run_id)}"
        self.sandbox_root = (
            self.workspace_root
            / ".solo-agent"
            / "sandboxes"
            / _safe_segment(session_id)
            / _safe_segment(run_id)
        )
        self.command_root = self.sandbox_root / "workspace"
        self.cache_root = self.workspace_root / ".solo-agent" / "cache"

    def prepare(self) -> CommandWorkspace:
        mode = self._select_mode()
        if mode == "local":
            return CommandWorkspace(
                mode="local",
                workspace_root=self.workspace_root,
                command_workspace_root=self.workspace_root,
                sandbox_id=self.sandbox_id,
                sandbox_root=self.workspace_root,
                cache_root=self.cache_root,
                network_policy=self.network_policy,
            )
        if self.sandbox_root.exists():
            shutil.rmtree(self.sandbox_root)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        baseline_commit = _git_head(self.workspace_root) if _is_git_repo(self.workspace_root) else None
        dirty_overlay = tuple(_git_dirty_paths(self.workspace_root)) if _is_git_repo(self.workspace_root) else tuple()
        if mode == "worktree" and not self._prepare_worktree():
            mode = "copy"
        if mode in {"copy", "isolated"}:
            self._prepare_copy()
        _overlay_workspace(self.workspace_root, self.command_root)
        manifest = build_workspace_manifest(self.command_root)
        manifest.update(
            {
                "sandbox_id": self.sandbox_id,
                "backend": "copy" if mode == "isolated" else mode,
                "workspace_root": str(self.workspace_root),
                "command_workspace_root": str(self.command_root),
                "baseline_commit": baseline_commit,
                "dirty_overlay": list(dirty_overlay),
            }
        )
        manifest_path = self.sandbox_root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        workspace = CommandWorkspace(
            mode=mode,
            workspace_root=self.workspace_root,
            command_workspace_root=self.command_root,
            created=True,
            sandbox_id=self.sandbox_id,
            sandbox_root=self.sandbox_root,
            cache_root=self.cache_root,
            baseline_manifest_path=manifest_path,
            baseline_commit=baseline_commit,
            dirty_overlay=dirty_overlay,
            network_policy=self.network_policy,
        )
        workspace.create_checkpoint("created")
        return workspace

    def _select_mode(self) -> str:
        if self.sandbox_mode == "auto":
            return "worktree" if _is_git_repo(self.workspace_root) else "copy"
        if self.sandbox_mode == "worktree" and not _is_git_repo(self.workspace_root):
            return "copy"
        return self.sandbox_mode

    def _prepare_worktree(self) -> bool:
        completed = _run_git(self.workspace_root, ["worktree", "add", "--detach", str(self.command_root), "HEAD"])
        if completed.returncode != 0:
            shutil.rmtree(self.command_root, ignore_errors=True)
            return False
        return True

    def _prepare_copy(self) -> None:
        shutil.copytree(
            self.workspace_root,
            self.command_root,
            ignore=_ignore_sandbox_entries,
            dirs_exist_ok=False,
            symlinks=True,
        )


def prepare_command_workspace(
    workspace_root: str | Path,
    *,
    session_id: str,
    run_id: str,
    sandbox_mode: str = "auto",
    network_policy: str = "deny",
) -> CommandWorkspace:
    return SandboxManager(
        workspace_root,
        session_id=session_id,
        run_id=run_id,
        sandbox_mode=sandbox_mode,
        network_policy=network_policy,
    ).prepare()


def build_workspace_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for item in root.rglob("*"):
        if item.is_dir() or _is_excluded(item, root):
            continue
        rel = item.relative_to(root).as_posix()
        stat = item.stat()
        payload: dict[str, Any] = {
            "type": "file",
            "sha256": _sha256_file(item),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 256_000:
            payload["content"] = item.read_text(encoding="utf-8", errors="replace")
        files[rel] = payload
    return {"files": files}


def diff_manifests(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    changed = sorted(
        rel for rel in before_keys & after_keys
        if isinstance(before.get(rel), dict)
        and isinstance(after.get(rel), dict)
        and before[rel].get("sha256") != after[rel].get("sha256")
    )
    return {
        "changed_files": changed,
        "new_files": sorted(after_keys - before_keys),
        "deleted_files": sorted(before_keys - after_keys),
    }


def _overlay_workspace(source_root: Path, target_root: Path) -> None:
    for item in source_root.rglob("*"):
        if item.is_dir() or _is_excluded(item, source_root):
            continue
        rel = item.relative_to(source_root)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target, follow_symlinks=False)


def _ignore_sandbox_entries(directory: str, names: list[str]) -> set[str]:
    base = Path(directory)
    ignored: set[str] = set()
    for name in names:
        candidate = base / name
        if name in SANDBOX_EXCLUDES or candidate.is_symlink():
            ignored.add(name)
    return ignored


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in SANDBOX_EXCLUDES for part in parts) or path.is_symlink()


def _normalize_mode(value: str) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in {"auto", "local", "copy", "isolated", "worktree", "docker"}:
        return "copy" if mode == "docker" else mode
    return "auto"


def _is_git_repo(root: Path) -> bool:
    return _run_git(root, ["rev-parse", "--is-inside-work-tree"]).returncode == 0


def _git_head(root: Path) -> str | None:
    completed = _run_git(root, ["rev-parse", "HEAD"])
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty_paths(root: Path) -> list[str]:
    completed = _run_git(root, ["status", "--porcelain"])
    if completed.returncode != 0:
        return []
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        if len(line) > 3:
            paths.append(line[3:].strip().replace("\\", "/"))
    return paths


def _run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(["git", *args], returncode=1, stdout="", stderr="")


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_segment(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value).strip())
    return safe[:80] or "run"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
