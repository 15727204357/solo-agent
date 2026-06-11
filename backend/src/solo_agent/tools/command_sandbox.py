"""Workspace-bounded command sandbox for programming workflows."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solo_agent.workflow.sandbox.command_workspace import MANIFEST_NAME, build_workspace_manifest, diff_manifests


class CommandPolicyError(ValueError):
    """Raised when a command violates the local command policy."""


@dataclass(frozen=True)
class CommandResult:
    command: str
    returncode: int | None
    output: str
    truncated: bool = False
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": self.command,
            "returncode": self.returncode,
            "output": self.output,
            "truncated": self.truncated,
        }
        if self.timed_out:
            payload["timed_out"] = True
        return payload


class LocalCommandSandbox:
    """Run allowlisted programming commands without invoking a shell."""

    _blocked_executables = {
        "bash",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
    }
    _dangerous_git = {
        ("git", "reset"),
        ("git", "clean"),
        ("git", "push"),
        ("git", "commit"),
        ("git", "checkout"),
        ("git", "switch"),
        ("git", "merge"),
        ("git", "rebase"),
        ("git", "tag"),
    }
    _dangerous_tokens = {
        "rm",
        "rmdir",
        "del",
        "erase",
        "remove-item",
        "set-content",
        "out-file",
        "invoke-webrequest",
        "curl",
        "wget",
    }
    _network_or_install_commands = {
        "curl",
        "wget",
        "pip",
        "pip3",
        "npx",
    }
    _install_like_tokens = {
        "install",
        "sync",
        "lock",
        "add",
        "remove",
        "publish",
        "deploy",
        "download",
    }
    _secret_pattern = re.compile(
        r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN)",
        re.I,
    )
    _shell_meta_pattern = re.compile(r"[;&|<>`]")

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        sandbox_mode: str = "local",
        sandbox_id: str = "",
        cache_root: str | Path | None = None,
        network_policy: str = "deny",
        command_timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
        max_changed_files: int = 200,
        max_workspace_bytes: int = 512_000_000,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.sandbox_mode = sandbox_mode
        self.sandbox_id = sandbox_id or self._discover_sandbox_id()
        self.cache_root = Path(cache_root or self._default_cache_root()).resolve()
        self.network_policy = network_policy if network_policy in {"deny", "allow_local", "allow_all"} else "deny"
        self.command_timeout_seconds = command_timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.max_changed_files = max_changed_files
        self.max_workspace_bytes = max_workspace_bytes

    def run(
        self,
        *,
        command: str,
        args: Sequence[str] | None = None,
        cwd: str = ".",
        timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
        purpose: str = "",
    ) -> dict[str, Any]:
        del purpose
        argv = [str(command), *[str(arg) for arg in (args or [])]]
        workdir = self._resolve_cwd(cwd)
        self._validate_argv(argv)
        self._validate_resource_budget()
        timeout = max(
            1,
            min(
                int(timeout_seconds or self.command_timeout_seconds),
                int(self.command_timeout_seconds or 300),
                300,
            ),
        )
        output_budget = min(int(max_output_bytes or self.max_output_bytes), int(self.max_output_bytes or 32_000))
        env, env_metadata = self._sanitized_env()
        try:
            completed = subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
            output = _truncate_text(f"{completed.stdout}{completed.stderr}", output_budget)
            resources = self._resource_usage(returncode=completed.returncode, timed_out=False)
            self._enforce_changed_file_cap(resources)
            result = CommandResult(
                command=_display_command(argv),
                returncode=completed.returncode,
                output=output["text"],
                truncated=output["truncated"],
            ).to_dict()
            result["metadata"] = {
                "sandbox": self._sandbox_metadata(
                    workdir=workdir,
                    returncode=completed.returncode,
                    truncated=output["truncated"],
                    timed_out=False,
                    resource_usage=resources,
                    env_metadata=env_metadata,
                )
            }
            return result
        except subprocess.TimeoutExpired as exc:
            output = _truncate_text(f"{exc.stdout or ''}{exc.stderr or ''}", output_budget)
            resources = self._resource_usage(returncode=None, timed_out=True)
            result = CommandResult(
                command=_display_command(argv),
                returncode=None,
                output=output["text"],
                truncated=output["truncated"],
                timed_out=True,
            ).to_dict()
            result["metadata"] = {
                "sandbox": self._sandbox_metadata(
                    workdir=workdir,
                    returncode=None,
                    truncated=output["truncated"],
                    timed_out=True,
                    resource_usage=resources,
                    env_metadata=env_metadata,
                )
            }
            return result

    def _resolve_cwd(self, cwd: str) -> Path:
        candidate = Path(cwd or ".")
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise CommandPolicyError(f"cwd escapes workspace root: {cwd}") from exc
        if not resolved.exists() or not resolved.is_dir():
            raise CommandPolicyError(f"cwd is not an existing directory: {cwd}")
        return resolved

    def _validate_argv(self, argv: Sequence[str]) -> None:
        if not argv or not argv[0].strip():
            raise CommandPolicyError("command must not be empty")
        executable = Path(argv[0]).name.casefold()
        normalized = [part.casefold() for part in argv]

        if executable in self._blocked_executables:
            raise CommandPolicyError("shell executables are not allowed; pass structured argv instead")
        if any(self._shell_meta_pattern.search(part) for part in argv):
            raise CommandPolicyError("shell metacharacters are not allowed in run_command arguments")
        if any(self._secret_pattern.search(part) for part in argv):
            raise CommandPolicyError("commands that reference secrets or tokens are not allowed")
        if any(part in self._dangerous_tokens for part in normalized):
            raise CommandPolicyError("destructive or network commands are not allowed")
        if self.network_policy == "deny" and self._looks_network_or_install_like(normalized):
            raise CommandPolicyError("network access and dependency installation are disabled by sandbox policy")
        if "ruff" in normalized and "format" in normalized and "--check" not in normalized:
            raise CommandPolicyError("formatting commands that rewrite files are not allowed")
        for prefix in self._dangerous_git:
            if tuple(normalized[: len(prefix)]) == prefix:
                raise CommandPolicyError(f"{' '.join(prefix)} is not allowed")
        if not self._is_allowlisted(normalized):
            raise CommandPolicyError(f"command is not allowlisted: {_display_command(argv)}")

    def _is_allowlisted(self, argv: Sequence[str]) -> bool:
        command = Path(argv[0]).name.casefold()
        if command in {"pytest", "eslint", "tsc"}:
            return True
        if command == "ruff":
            return len(argv) >= 2 and (argv[1] == "check" or (argv[1] == "format" and "--check" in argv))
        if command in {"python", "python.exe", "python3"}:
            return len(argv) >= 3 and argv[1] == "-m" and argv[2] in {"pytest", "ruff", "mypy"}
        if command == "uv":
            return (
                len(argv) >= 3
                and argv[1] in {"run", "tool"}
                and not any(part in {"pip", "install", "sync", "lock", "add", "remove"} for part in argv[2:])
            )
        if command in {"npm", "pnpm", "yarn"}:
            if len(argv) >= 2 and argv[1] == "test":
                return True
            return len(argv) >= 3 and argv[1] == "run" and argv[2] in {"build", "test", "lint", "typecheck", "check"}
        if command == "cargo":
            return len(argv) >= 2 and argv[1] in {"test", "check", "clippy"}
        if command == "go":
            return len(argv) >= 2 and argv[1] == "test"
        if command == "git":
            return len(argv) >= 2 and argv[1] in {"status", "diff", "show", "log"}
        return False

    def _looks_network_or_install_like(self, argv: Sequence[str]) -> bool:
        command = Path(argv[0]).name.casefold()
        if command in self._network_or_install_commands:
            return True
        if command in {"uv", "npm", "pnpm", "yarn", "go", "cargo"}:
            return any(part in self._install_like_tokens for part in argv[1:])
        if command in {"python", "python.exe", "python3"}:
            return any(part in {"pip", "ensurepip"} for part in argv[1:])
        return False

    def _sandbox_metadata(
        self,
        *,
        workdir: Path,
        returncode: int | None,
        truncated: bool,
        timed_out: bool,
        resource_usage: dict[str, Any],
        env_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "mode": self.sandbox_mode,
            "backend": self.sandbox_mode,
            "workspace_root": str(self.workspace_root),
            "cwd": str(workdir),
            "returncode": returncode,
            "truncated": truncated,
            "timed_out": timed_out,
            "network_policy": self.network_policy,
            "network_enforcement": "policy",
            "env_policy": "minimal",
            "env": env_metadata,
            "cache_paths": self._cache_paths(),
            "resource_usage": resource_usage,
            "changed_file_count": int(resource_usage.get("changed_file_count", 0)),
            "baseline_commit": self._baseline_manifest().get("baseline_commit"),
        }

    def _sanitized_env(self) -> tuple[dict[str, str], dict[str, Any]]:
        allowed_keys = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CI",
        }
        env: dict[str, str] = {}
        redacted: list[str] = []
        for key, value in os.environ.items():
            if self._secret_pattern.search(key):
                redacted.append(key)
                continue
            if key.upper() in allowed_keys:
                env[key] = value
        for key, value in self._cache_paths().items():
            env[key] = value
        env.setdefault("CI", "1")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return env, {"included_keys": sorted(env), "redacted_keys": sorted(redacted)}

    def _cache_paths(self) -> dict[str, str]:
        return {
            "UV_CACHE_DIR": str(self.cache_root / "uv"),
            "PIP_CACHE_DIR": str(self.cache_root / "pip"),
            "npm_config_cache": str(self.cache_root / "npm"),
            "CARGO_HOME": str(self.cache_root / "cargo"),
            "GOMODCACHE": str(self.cache_root / "go-mod"),
        }

    def _resource_usage(self, *, returncode: int | None, timed_out: bool) -> dict[str, Any]:
        manifest = build_workspace_manifest(self.workspace_root)
        baseline = self._baseline_manifest()
        diff = diff_manifests(baseline.get("files", {}), manifest.get("files", {}))
        workspace_bytes = sum(int(item.get("size", 0)) for item in manifest.get("files", {}).values() if isinstance(item, dict))
        return {
            "returncode": returncode,
            "timed_out": timed_out,
            "workspace_bytes": workspace_bytes,
            "changed_file_count": len(diff["changed_files"]) + len(diff["new_files"]) + len(diff["deleted_files"]),
            "changed_files": diff["changed_files"],
            "new_files": diff["new_files"],
            "deleted_files": diff["deleted_files"],
            "limit_hit": timed_out or workspace_bytes > self.max_workspace_bytes,
        }

    def _baseline_manifest(self) -> dict[str, Any]:
        manifest_path = self.workspace_root.parent / MANIFEST_NAME
        if not manifest_path.exists():
            return {"files": build_workspace_manifest(self.workspace_root).get("files", {})}
        import json

        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _enforce_changed_file_cap(self, resources: dict[str, Any]) -> None:
        if int(resources.get("changed_file_count", 0)) > self.max_changed_files:
            raise CommandPolicyError("sandbox changed file limit exceeded")

    def _validate_resource_budget(self) -> None:
        workspace_bytes = sum(
            item.stat().st_size
            for item in self.workspace_root.rglob("*")
            if item.is_file() and ".solo-agent" not in item.parts
        )
        if workspace_bytes > self.max_workspace_bytes:
            raise CommandPolicyError("sandbox workspace size limit exceeded")

    def _default_cache_root(self) -> Path:
        if self.workspace_root.name == "workspace" and self.workspace_root.parent.name:
            if len(self.workspace_root.parents) >= 3:
                return self.workspace_root.parents[2] / "cache"
            return self.workspace_root / ".solo-agent" / "cache"
        return self.workspace_root / ".solo-agent" / "cache"

    def _discover_sandbox_id(self) -> str:
        if self.workspace_root.name == "workspace" and len(self.workspace_root.parents) >= 2:
            return self.workspace_root.parent.name
        return "local"


def _truncate_text(text: str, max_bytes: int) -> dict[str, Any]:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return {"text": text, "truncated": False}
    suffix = "\n...[truncated; rerun with a narrower target or larger max_output_bytes]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = raw[:head_budget].decode("utf-8", errors="ignore")
    tail = raw[-tail_budget:].decode("utf-8", errors="ignore") if tail_budget else ""
    return {"text": f"{head}{suffix}\n{tail}", "truncated": True}


def _display_command(command: Sequence[str]) -> str:
    return " ".join(command)
