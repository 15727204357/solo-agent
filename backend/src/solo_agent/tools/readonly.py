"""Workspace-bounded tools for context, skills, guarded edits, and quality checks."""

from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import re
import subprocess
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solo_agent.context import WorkspaceTaskStore

DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".uv-cache",
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "node_modules",
    "dist",
    "build",
}

_FENCE_TAG_RE = re.compile(r"</?\s*(?:skill-context|memory-context)\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class WorkspaceTools:
    """Factory object for workspace tools.

    The first writable tools are hash-anchored on purpose: the agent must prove
    it is editing the version of the file it just inspected.
    """

    workspace_root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())

    def list_files(
        self,
        path: str = ".",
        *,
        recursive: bool = True,
        max_entries: int = 500,
        include_dirs: bool = False,
    ) -> dict[str, Any]:
        root = self._resolve_inside_workspace(path)
        entries: list[dict[str, Any]] = []

        iterator: Iterable[Path]
        if root.is_file():
            iterator = [root]
        elif recursive:
            iterator = root.rglob("*")
        else:
            iterator = root.iterdir()

        for item in iterator:
            if self._is_excluded(item):
                continue
            if item.is_dir() and not include_dirs:
                continue
            entries.append(
                {
                    "path": self._relative(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                }
            )
            if len(entries) >= max_entries:
                break

        return {
            "root": str(self.workspace_root),
            "path": self._relative(root),
            "entries": entries,
            "truncated": len(entries) >= max_entries,
        }

    def read_file(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        max_bytes: int = 64_000,
    ) -> dict[str, Any]:
        file_path = self._resolve_readable_file(path)
        size = file_path.stat().st_size
        if size > max_bytes:
            with file_path.open("rb") as handle:
                raw = handle.read(max_bytes)
            truncated = True
        else:
            raw = file_path.read_bytes()
            truncated = False

        return {
            "path": self._relative(file_path),
            "content": raw.decode(encoding, errors="replace"),
            "sha256": _sha256_file(file_path),
            "size": size,
            "truncated": truncated,
        }

    def search_text(
        self,
        query: str,
        path: str = ".",
        *,
        glob: str = "*",
        max_matches: int = 100,
        max_file_bytes: int = 512_000,
    ) -> dict[str, Any]:
        if not query:
            raise ValueError("query must not be empty")

        search_root = self._resolve_inside_workspace(path)
        candidates = [search_root] if search_root.is_file() else search_root.rglob("*")
        matches: list[dict[str, Any]] = []

        for file_path in candidates:
            if not self._is_text_candidate(file_path, glob=glob, max_file_bytes=max_file_bytes):
                continue

            for line_number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if query in line:
                    matches.append(
                        {
                            "path": self._relative(file_path),
                            "line": line_number,
                            "text": line,
                        }
                    )
                    if len(matches) >= max_matches:
                        return {
                            "query": query,
                            "matches": matches,
                            "truncated": True,
                        }

        return {"query": query, "matches": matches, "truncated": False}

    def get_file_hash(self, path: str) -> dict[str, Any]:
        file_path = self._resolve_readable_file(path)
        stat = file_path.stat()
        return {
            "path": self._relative(file_path),
            "sha256": _sha256_file(file_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def workspace_snapshot(
        self,
        path: str = ".",
        *,
        max_entries: int = 500,
    ) -> dict[str, Any]:
        root = self._resolve_inside_workspace(path)
        files: list[Path] = []
        dirs = 0
        for item in root.rglob("*") if root.is_dir() else [root]:
            if self._is_excluded(item):
                continue
            if item.is_dir():
                dirs += 1
                continue
            files.append(item)
            if len(files) >= max_entries:
                break

        extensions = Counter(file.suffix.lower() or "<none>" for file in files)
        key_names = {
            "pyproject.toml",
            "README.md",
            "requirements.txt",
            "uv.lock",
            "package.json",
            "SKILL.md",
        }
        key_files = [self._relative(file) for file in files if file.name in key_names]
        return {
            "root": str(self.workspace_root),
            "path": self._relative(root),
            "file_count": len(files),
            "directory_count": dirs,
            "extensions": dict(extensions.most_common(20)),
            "key_files": sorted(key_files)[:50],
            "truncated": len(files) >= max_entries,
        }

    def inspect_python_symbols(self, path: str) -> dict[str, Any]:
        file_path = self._resolve_readable_file(path)
        if file_path.suffix != ".py":
            raise ValueError("inspect_python_symbols only supports .py files")
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        imports: list[dict[str, Any]] = []
        symbols: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    {"module": alias.name, "name": alias.asname or alias.name, "line": node.lineno}
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imports.extend(
                    {
                        "module": "." * node.level + (node.module or ""),
                        "name": alias.asname or alias.name,
                        "line": node.lineno,
                    }
                    for alias in node.names
                )
            elif isinstance(node, ast.ClassDef):
                symbols.append(_symbol_dict("class", node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(_symbol_dict("function", node))

        return {
            "path": self._relative(file_path),
            "imports": sorted(imports, key=lambda item: item["line"]),
            "symbols": sorted(symbols, key=lambda item: item["line_start"]),
        }

    def prepare_edit(
        self,
        path: str,
        *,
        anchor: str | None = None,
        old_text: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        context_lines: int = 5,
        max_bytes: int = 80_000,
    ) -> dict[str, Any]:
        file_path = self._resolve_readable_file(path)
        content = _read_limited_text(file_path, max_bytes=max_bytes)
        lines = content.splitlines()
        anchor = anchor or old_text
        start, end = _resolve_window(lines, anchor=anchor, line_start=line_start, line_end=line_end)
        start = max(1, start - context_lines)
        end = min(len(lines), end + context_lines)
        snippet = "\n".join(lines[start - 1 : end])
        return {
            "path": self._relative(file_path),
            "sha256": _sha256_file(file_path),
            "expected_hash": _sha256_file(file_path),
            "line_start": start,
            "line_end": end,
            "line_count": len(lines),
            "snippet": snippet,
            "truncated": file_path.stat().st_size > max_bytes,
        }

    def preview_patch(
        self,
        path: str,
        *,
        expected_hash: str,
        new_text: str,
        old_text: str | None = None,
        anchor: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        file_path = self._resolve_writable_file(path)
        original = file_path.read_text(encoding="utf-8", errors="replace")
        updated = self._build_updated_text(
            file_path,
            expected_hash=expected_hash,
            original=original,
            new_text=new_text,
            old_text=old_text,
            anchor=anchor,
            line_start=line_start,
            line_end=line_end,
        )
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=self._relative(file_path),
                tofile=self._relative(file_path),
                lineterm="",
            )
        )
        return {
            "path": self._relative(file_path),
            "changed": original != updated,
            "sha256": expected_hash,
            "new_sha256": _sha256_text(updated),
            "diff": diff,
            "truncated": False,
        }

    def apply_text_edit(
        self,
        path: str,
        *,
        expected_hash: str,
        new_text: str,
        old_text: str | None = None,
        anchor: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        file_path = self._resolve_writable_file(path)
        original = file_path.read_text(encoding="utf-8", errors="replace")
        updated = self._build_updated_text(
            file_path,
            expected_hash=expected_hash,
            original=original,
            new_text=new_text,
            old_text=old_text,
            anchor=anchor,
            line_start=line_start,
            line_end=line_end,
        )
        if original == updated:
            return {
                "path": self._relative(file_path),
                "changed": False,
                "sha256": expected_hash,
                "new_sha256": expected_hash,
            }
        file_path.write_text(updated, encoding="utf-8")
        return {
            "path": self._relative(file_path),
            "changed": True,
            "sha256": expected_hash,
            "new_sha256": _sha256_file(file_path),
        }

    def run_pytest(
        self,
        *,
        target: str | None = None,
        args: Sequence[str] | None = None,
        timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
    ) -> dict[str, Any]:
        command_args = list(args or ["-q"])
        if target:
            command_args.append(self._relative(self._resolve_command_target(target)))
        return self._run_allowed_command(
            ["uv", "run", "--extra", "dev", "python", "-m", "pytest", *command_args],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def run_ruff_check(
        self,
        *,
        target: str | None = None,
        args: Sequence[str] | None = None,
        timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
    ) -> dict[str, Any]:
        command_args = list(args or ["."])
        if target:
            command_args = [self._relative(self._resolve_command_target(target))]
        return self._run_allowed_command(
            ["uv", "run", "--extra", "dev", "ruff", "check", *command_args],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def run_ruff_format_check(
        self,
        *,
        target: str | None = None,
        args: Sequence[str] | None = None,
        timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
    ) -> dict[str, Any]:
        command_args = list(args or ["."])
        if target:
            command_args = [self._relative(self._resolve_command_target(target))]
        return self._run_allowed_command(
            ["uv", "run", "--extra", "dev", "ruff", "format", "--check", *command_args],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def git_status(
        self,
        *,
        timeout_seconds: int = 30,
        max_output_bytes: int = 12_000,
    ) -> dict[str, Any]:
        return self._run_git_command(
            ["git", "status", "--short", "--branch"],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def git_diff(
        self,
        *,
        path: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 24_000,
    ) -> dict[str, Any]:
        command = ["git", "diff", "--"]
        if path:
            command.append(self._relative(self._resolve_command_target(path)))
        return self._run_git_command(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def git_recent_changes(
        self,
        *,
        limit: int = 10,
        path: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 16_000,
    ) -> dict[str, Any]:
        bounded_limit = max(1, min(int(limit), 50))
        command = ["git", "log", "--oneline", "--decorate", f"-n{bounded_limit}", "--"]
        if path:
            command.append(self._relative(self._resolve_command_target(path)))
        return self._run_git_command(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )

    def read_test_failure(
        self,
        output: str,
        *,
        max_failures: int = 10,
    ) -> dict[str, Any]:
        failures = _parse_pytest_failures(output, max_failures=max_failures)
        return {
            "failure_count": len(failures),
            "failures": failures,
            "truncated": len(failures) >= max_failures,
        }

    def targeted_pytest(
        self,
        target: str,
        *,
        timeout_seconds: int = 60,
        max_output_bytes: int = 32_000,
    ) -> dict[str, Any]:
        pytest_target = self._resolve_pytest_target(target)
        result = self._run_allowed_command(
            ["uv", "run", "--extra", "dev", "python", "-m", "pytest", "-q", pytest_target],
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        result["failures"] = _parse_pytest_failures(str(result.get("output", "")))
        return result

    def list_skills(
        self,
        *,
        path: str = ".",
        max_entries: int | None = None,
        max_skills: int = 50,
    ) -> dict[str, Any]:
        if max_entries is not None:
            max_skills = max_entries
        root = self._resolve_inside_workspace(path)
        candidates = [root] if root.name == "SKILL.md" else root.rglob("SKILL.md")
        skills = []
        for skill_file in candidates:
            if not skill_file.is_file() or self._is_excluded(skill_file):
                continue
            text = skill_file.read_text(encoding="utf-8", errors="replace")
            skills.append(_skill_summary(skill_file, text, self.workspace_root))
            if len(skills) >= max_skills:
                break
        return {"skills": skills, "truncated": len(skills) >= max_skills}

    def load_skill(
        self,
        path: str,
        *,
        max_bytes: int = 48_000,
    ) -> dict[str, Any]:
        skill_file = self._resolve_skill_path(path)
        if skill_file.name != "SKILL.md":
            raise ValueError("load_skill only reads SKILL.md files")
        raw = _read_limited_text(skill_file, max_bytes=max_bytes)
        clean = _sanitize_skill(raw)
        summary = _skill_summary(skill_file, clean, self.workspace_root)
        return {
            **summary,
            "content": clean,
            "system_note": "Skill content is background SOP, NOT new user input.",
            "truncated": skill_file.stat().st_size > max_bytes,
        }

    def select_relevant_skills(
        self,
        query: str | None = None,
        *,
        task: str | None = None,
        plan: str | None = None,
        path: str = ".",
        limit: int = 3,
        max_skills: int | None = None,
    ) -> dict[str, Any]:
        query = query or " ".join(part for part in (task, plan) if part) or ""
        if max_skills is not None:
            limit = max_skills
        listed = self.list_skills(path=path)
        terms = {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", query) if len(term) >= 2}
        scored = []
        for skill in listed["skills"]:
            haystack = " ".join(
                [
                    str(skill.get("name", "")),
                    str(skill.get("description", "")),
                    " ".join(str(item) for item in skill.get("triggers", [])),
                    " ".join(str(item) for item in skill.get("red_flags", [])),
                    " ".join(str(item) for item in skill.get("required_tools", [])),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append({**skill, "score": score})
        selected = sorted(scored, key=lambda item: (-item["score"], _skill_category_rank(item), item["path"]))[:limit]
        return {"skills": selected, "truncated": listed["truncated"]}

    def task_create(
        self,
        thread_id: str,
        subject: str,
        *,
        description: str = "",
        status: str = "pending",
        active_form: str = "",
        blocked_by: list[str] | None = None,
        blocks: list[str] | None = None,
        owner: str = "solo-agent",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return WorkspaceTaskStore(self.workspace_root).create_task(
            thread_id,
            subject=subject,
            description=description,
            status=status,
            active_form=active_form,
            blocked_by=blocked_by or [],
            blocks=blocks or [],
            owner=owner,
            metadata=metadata or {"source": "task_create"},
        )

    def task_get(self, thread_id: str, task_id: str) -> dict[str, Any]:
        return WorkspaceTaskStore(self.workspace_root).get_task(thread_id, task_id)

    def task_list(self, thread_id: str, *, include_deleted: bool = False) -> dict[str, Any]:
        return WorkspaceTaskStore(self.workspace_root).list_tasks(thread_id, include_deleted=include_deleted)

    def task_update(
        self,
        thread_id: str,
        task_id: str,
        *,
        subject: str | None = None,
        description: str | None = None,
        status: str | None = None,
        active_form: str | None = None,
        blocked_by: list[str] | None = None,
        blocks: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return WorkspaceTaskStore(self.workspace_root).update_task(
            thread_id,
            task_id,
            subject=subject,
            description=description,
            status=status,
            active_form=active_form,
            blocked_by=blocked_by,
            blocks=blocks,
            metadata=metadata,
        )

    def _build_updated_text(
        self,
        file_path: Path,
        *,
        expected_hash: str,
        original: str,
        new_text: str,
        old_text: str | None,
        anchor: str | None,
        line_start: int | None,
        line_end: int | None,
    ) -> str:
        current_hash = _sha256_file(file_path)
        if current_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {self._relative(file_path)}: expected {expected_hash}, got {current_hash}")
        if old_text is not None:
            count = original.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must match exactly once, found {count}")
            return original.replace(old_text, new_text, 1)
        if anchor is not None:
            count = original.count(anchor)
            if count != 1:
                raise ValueError(f"anchor must match exactly once, found {count}")
            return original.replace(anchor, new_text, 1)
        if line_start is not None and line_end is not None:
            lines = original.splitlines(keepends=True)
            if line_start < 1 or line_end < line_start or line_end > len(lines):
                raise ValueError("line range is out of bounds")
            replacement = new_text if new_text.endswith(("\n", "\r")) else f"{new_text}\n"
            return "".join([*lines[: line_start - 1], replacement, *lines[line_end:]])
        raise ValueError("edit requires old_text, anchor, or line_start/line_end")

    def _run_allowed_command(
        self,
        command: list[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> dict[str, Any]:
        timeout = max(1, min(int(timeout_seconds), 300))
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = _truncate_text(f"{completed.stdout}{completed.stderr}", max_output_bytes)
            return {
                "command": _display_command(command),
                "returncode": completed.returncode,
                "output": output["text"],
                "truncated": output["truncated"],
            }
        except subprocess.TimeoutExpired as exc:
            output = _truncate_text(f"{exc.stdout or ''}{exc.stderr or ''}", max_output_bytes)
            return {
                "command": _display_command(command),
                "returncode": None,
                "output": output["text"],
                "timed_out": True,
                "truncated": output["truncated"],
            }

    def _run_git_command(
        self,
        command: list[str],
        *,
        timeout_seconds: int,
        max_output_bytes: int,
    ) -> dict[str, Any]:
        result = self._run_allowed_command(
            command,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        output = str(result.get("output", ""))
        if result.get("returncode") == 128 and "not a git repository" in output.lower():
            result["error"] = {
                "code": "not_git_repository",
                "message": "Workspace is not inside a git repository.",
            }
        return result

    def _resolve_command_target(self, target: str) -> Path:
        resolved = self._resolve_inside_workspace(target)
        if self._is_excluded(resolved):
            raise PermissionError(f"Command target is excluded: {target}")
        return resolved

    def _resolve_pytest_target(self, target: str) -> str:
        if not target.strip():
            raise ValueError("targeted_pytest requires a test path or node id")

        path_text, *selectors = target.split("::")
        test_path = self._resolve_command_target(path_text)
        if not test_path.exists():
            raise ValueError(f"Test target does not exist: {path_text}")
        if selectors and not test_path.is_file():
            raise ValueError("Pytest node ids must target a test file")
        if not _is_test_path(test_path, self.workspace_root):
            raise ValueError("targeted_pytest only accepts workspace test paths")
        for selector in selectors:
            if not _SAFE_PYTEST_SELECTOR_RE.fullmatch(selector):
                raise ValueError(f"Unsafe pytest selector: {selector}")
        return "::".join([self._relative(test_path), *selectors])

    def _resolve_skill_path(self, path_or_name: str) -> Path:
        try:
            direct = self._resolve_inside_workspace(path_or_name)
        except PermissionError:
            raise

        if direct.is_dir():
            direct = direct / "SKILL.md"
        if direct.is_file() and direct.name == "SKILL.md":
            if self._is_excluded(direct):
                raise PermissionError(f"Skill path is excluded: {path_or_name}")
            return direct

        for skill in self.list_skills(max_skills=200)["skills"]:
            if path_or_name in {str(skill["name"]), str(skill["path"]), Path(str(skill["path"])).parent.name}:
                return self._resolve_inside_workspace(str(skill["path"]))
        raise ValueError(f"Skill not found: {path_or_name}")

    def _resolve_inside_workspace(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()

        if not _is_relative_to(resolved, self.workspace_root):
            raise PermissionError(f"Path escapes workspace root: {path}")
        return resolved

    def _resolve_readable_file(self, path: str) -> Path:
        file_path = self._resolve_inside_workspace(path)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        if self._is_excluded(file_path):
            raise PermissionError(f"Reading excluded or secret file is not allowed: {path}")
        return file_path

    def _resolve_writable_file(self, path: str) -> Path:
        file_path = self._resolve_inside_workspace(path)
        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")
        if self._is_excluded(file_path):
            raise PermissionError(f"Writing excluded or secret file is not allowed: {path}")
        return file_path

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix() or "."

    def _is_excluded(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            if not _is_relative_to(resolved, self.workspace_root):
                return True
            relative_parts = resolved.relative_to(self.workspace_root).parts
        except ValueError:
            return True
        return any(
            part in DEFAULT_EXCLUDES or part.startswith(".env.") or part.endswith(".pem")
            for part in relative_parts
        )

    def _is_text_candidate(self, file_path: Path, *, glob: str, max_file_bytes: int) -> bool:
        return (
            file_path.is_file()
            and not self._is_excluded(file_path)
            and fnmatch.fnmatch(file_path.name, glob)
            and file_path.stat().st_size <= max_file_bytes
            and not _looks_binary(file_path)
        )


def list_files(workspace_root: str | Path, path: str = ".", **kwargs: Any) -> dict[str, Any]:
    return WorkspaceTools(workspace_root).list_files(path, **kwargs)


def read_file(workspace_root: str | Path, path: str, **kwargs: Any) -> dict[str, Any]:
    return WorkspaceTools(workspace_root).read_file(path, **kwargs)


def search_text(
    workspace_root: str | Path,
    query: str,
    path: str = ".",
    **kwargs: Any,
) -> dict[str, Any]:
    return WorkspaceTools(workspace_root).search_text(query, path, **kwargs)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        chunk = handle.read(1024)
    return b"\0" in chunk


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _symbol_dict(kind: str, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": node.name,
        "line_start": node.lineno,
        "line_end": getattr(node, "end_lineno", node.lineno),
        "docstring": ast.get_docstring(node),
    }


def _read_limited_text(path: Path, *, max_bytes: int) -> str:
    raw = path.read_bytes()
    return raw[:max_bytes].decode("utf-8", errors="replace")


def _resolve_window(
    lines: list[str],
    *,
    anchor: str | None,
    line_start: int | None,
    line_end: int | None,
) -> tuple[int, int]:
    if line_start is not None or line_end is not None:
        start = line_start or 1
        end = line_end or start
        if start < 1 or end < start or end > len(lines):
            raise ValueError("line range is out of bounds")
        return start, end
    if anchor:
        matches = [index for index, line in enumerate(lines, start=1) if anchor in line]
        if len(matches) != 1:
            raise ValueError(f"anchor must match exactly one line, found {len(matches)}")
        return matches[0], matches[0]
    return 1, min(len(lines), 40)


def _truncate_text(text: str, max_bytes: int) -> dict[str, Any]:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return {"text": text, "truncated": False}
    return {
        "text": raw[:max_bytes].decode("utf-8", errors="replace") + "\n...[truncated]",
        "truncated": True,
    }


def _display_command(command: Sequence[str]) -> str:
    return " ".join(command)


_FAILED_SUMMARY_RE = re.compile(r"^FAILED\s+(?P<nodeid>\S+?)(?:\s+-\s+(?P<reason>.*))?$")
_FAILURE_HEADER_RE = re.compile(r"^_{2,}\s+(?P<test>.+?)\s+_{2,}$")
_LOCATION_RE = re.compile(r"(?P<path>[\w./\\-]+\.py):\d+")
_SAFE_PYTEST_SELECTOR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[A-Za-z0-9_.:,/+= -]+\])?")


def _parse_pytest_failures(output: str, *, max_failures: int = 10) -> list[dict[str, str | None]]:
    lines = output.splitlines()
    snippets_by_test = _assertion_snippets_by_test(lines)
    failures: list[dict[str, str | None]] = []

    for line in lines:
        match = _FAILED_SUMMARY_RE.match(line.strip())
        if not match:
            continue
        nodeid = match.group("nodeid")
        path, test_name = _split_pytest_nodeid(nodeid)
        failures.append(
            {
                "path": path,
                "test": test_name,
                "assertion": snippets_by_test.get(test_name),
                "summary": match.group("reason"),
            }
        )
        if len(failures) >= max_failures:
            return failures

    if failures:
        return failures

    current_test: str | None = None
    current_path: str | None = None
    current_assertion: str | None = None
    for line in lines:
        header = _FAILURE_HEADER_RE.match(line.strip())
        if header:
            if current_test:
                failures.append({"path": current_path, "test": current_test, "assertion": current_assertion, "summary": None})
                if len(failures) >= max_failures:
                    return failures
            current_test = header.group("test")
            current_path = None
            current_assertion = None
            continue
        if current_test and current_path is None:
            location = _LOCATION_RE.search(line)
            if location:
                current_path = location.group("path").replace("\\", "/")
        if current_test and current_assertion is None:
            snippet = _extract_assertion_line(line)
            if snippet:
                current_assertion = snippet

    if current_test and len(failures) < max_failures:
        failures.append({"path": current_path, "test": current_test, "assertion": current_assertion, "summary": None})
    return failures


def _assertion_snippets_by_test(lines: list[str]) -> dict[str, str]:
    snippets: dict[str, str] = {}
    current_test: str | None = None
    for line in lines:
        header = _FAILURE_HEADER_RE.match(line.strip())
        if header:
            current_test = header.group("test")
            continue
        if current_test and current_test not in snippets:
            snippet = _extract_assertion_line(line)
            if snippet:
                snippets[current_test] = snippet
    return snippets


def _extract_assertion_line(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("E       "):
        stripped = stripped.removeprefix("E       ").strip()
    if stripped.startswith(">"):
        stripped = stripped.lstrip("> ").strip()
    if stripped.startswith("assert "):
        return stripped
    return None


def _split_pytest_nodeid(nodeid: str) -> tuple[str, str]:
    path, *selectors = nodeid.split("::")
    return path.replace("\\", "/"), "::".join(selectors)


def _is_test_path(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if "tests" in parts:
        return True
    return path.is_file() and path.suffix == ".py" and (path.name.startswith("test_") or path.name.endswith("_test.py"))


def _sanitize_skill(text: str) -> str:
    return _FENCE_TAG_RE.sub("", text)


def _skill_summary(skill_file: Path, text: str, root: Path) -> dict[str, Any]:
    frontmatter = _parse_frontmatter(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    title = str(
        frontmatter.get("name")
        or next((line.lstrip("# ") for line in lines if line.startswith("#")), skill_file.parent.name)
    )
    description = str(
        frontmatter.get("description")
        or next((line for line in lines if not line.startswith("#") and line != "---"), "")
    )
    return {
        "name": title,
        "description": description[:240],
        "category": frontmatter.get("category", "general"),
        "triggers": frontmatter.get("triggers", []),
        "red_flags": frontmatter.get("red_flags", []),
        "required_tools": frontmatter.get("required_tools", []),
        "path": skill_file.relative_to(root).as_posix(),
    }


def _skill_category_rank(skill: Mapping[str, Any]) -> int:
    category = str(skill.get("category", "")).lower()
    if category == "behavior":
        return 0
    if category == "tools":
        return 1
    if category == "workflow":
        return 2
    return 3


def _parse_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict[str, Any] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            break
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        value = raw_value.strip()
        data[key.strip()] = _parse_frontmatter_value(value)
    return data


def _parse_frontmatter_value(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return value.strip("'\"")
