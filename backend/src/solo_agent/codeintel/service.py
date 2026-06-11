"""Workspace-bounded Python code intelligence service."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .index_store import INDEX_VERSION, CodeIndexStore
from .python_indexer import PythonIndexer, module_name
from .retrieval import semantic_search
from .test_relevance import relevant_tests

DEFAULT_CODEINTEL_EXCLUDES = {
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
    ".solo-agent/sandboxes",
    "node_modules",
    "dist",
    "build",
}


class CodeIntelligenceService:
    """Persistent, incremental, LSP-like Python code index."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        max_files: int = 2_000,
        max_file_bytes: int = 512_000,
        index_ttl_seconds: int = 30,
        db_path: Path | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.index_ttl_seconds = index_ttl_seconds
        self.store = CodeIndexStore(self.workspace_root, db_path=db_path)
        self.indexer = PythonIndexer(self.workspace_root)
        self._last_refresh = 0.0

    def status(self, path: str = ".", *, refresh: bool = False) -> dict[str, Any]:
        changed = self.refresh(force=refresh)
        return {
            "index_version": INDEX_VERSION,
            "backend": "python_lsp_like",
            "languages": ["python"],
            "stale": self.is_stale(),
            "changed_files_indexed": changed["indexed"],
            "deleted_files_removed": changed["deleted"],
            "path": self._relative(self._resolve(path)),
            **self.store.metadata(),
        }

    def refresh(self, *, force: bool = False) -> dict[str, Any]:
        if not force and not self.is_stale():
            return {"indexed": 0, "skipped": 0, "deleted": 0, "errors": 0}
        files = self._python_files()
        current_paths = {self._relative(path) for path in files}
        deleted = self.store.indexed_paths() - current_paths
        if deleted:
            self.store.delete_paths(deleted)
        indexed = 0
        skipped = 0
        errors = 0
        for file_path in files:
            rel = self._relative(file_path)
            stat = file_path.stat()
            existing = self.store.file_record(rel)
            if (
                not force
                and existing is not None
                and existing.size == stat.st_size
                and existing.mtime == stat.st_mtime
                and existing.sha256 == _sha256_file(file_path)
            ):
                skipped += 1
                continue
            indexed_file = self.indexer.index_file(file_path)
            if indexed_file.file.parse_error:
                errors += 1
            self.store.replace_file(indexed_file)
            indexed += 1
        self._last_refresh = time.time()
        return {"indexed": indexed, "skipped": skipped, "deleted": len(deleted), "errors": errors}

    def is_stale(self) -> bool:
        if time.time() - self._last_refresh > self.index_ttl_seconds:
            return True
        for file_path in self._python_files():
            rel = self._relative(file_path)
            stat = file_path.stat()
            existing = self.store.file_record(rel)
            if existing is None or existing.size != stat.st_size or existing.mtime != stat.st_mtime:
                return True
        return False

    def code_map(self, path: str = ".", *, max_files: int = 500) -> dict[str, Any]:
        changed = self.refresh()
        root = self._resolve(path)
        rel_root = self._relative(root)
        rows = self.store.code_map_rows(path_prefix="" if rel_root == "." else rel_root, limit=max_files)
        modules = [
            {"path": item["path"], "module": item["module"], "imports": [], "symbols": [], "calls": []}
            | ({"error": item["parse_error"]} if item.get("parse_error") else {})
            for item in rows["modules"]
        ]
        test_files = rows["tests"]
        if rel_root == ".":
            test_files = sorted({*test_files, *[path for path in self._test_paths_from_files(rows["files"])]})
        entrypoints = [
            str(item["path"])
            for item in rows["files"]
            if Path(str(item["path"])).name in {"__main__.py", "main.py", "app.py"}
        ]
        return {
            "root": rel_root,
            "file_count": len(rows["files"]),
            "python_file_count": sum(1 for item in rows["files"] if str(item["path"]).endswith(".py")),
            "modules": modules[:max_files],
            "symbols": rows["symbols"][: max_files * 8],
            "import_edges": rows["imports"][: max_files * 12],
            "call_edges": rows["calls"][: max_files * 12],
            "test_files": sorted(test_files)[:100],
            "entrypoints": sorted(entrypoints)[:50],
            "truncated": len(rows["files"]) >= max_files,
            "index_version": INDEX_VERSION,
            "backend": "python_lsp_like",
            "languages": ["python"],
            "stale": self.is_stale(),
            "symbol_count": len(rows["symbols"]),
            "call_edge_count": len(rows["calls"]),
            "parse_errors": rows["parse_errors"],
            "refresh": changed,
        }

    def find_references(self, symbol: str, path: str = ".", *, max_matches: int = 100) -> dict[str, Any]:
        self.refresh()
        if not symbol.strip():
            raise ValueError("symbol must not be empty")
        root = self._relative(self._resolve(path))
        matches = self.store.references(symbol.strip(), limit=max_matches * 2)
        if root != ".":
            prefix = root.rstrip("/")
            matches = [
                item
                for item in matches
                if item["path"] == prefix or str(item["path"]).startswith(f"{prefix}/")
            ]
        return {
            "symbol": symbol,
            "matches": matches[:max_matches],
            "truncated": len(matches) > max_matches,
            "index_version": INDEX_VERSION,
        }

    def analyze_impact(
        self,
        *,
        paths: list[str] | tuple[str, ...] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
        include_tests: bool = True,
    ) -> dict[str, Any]:
        self.refresh()
        normalized_paths = [self._relative(self._resolve(path)) for path in (paths or []) if str(path).strip()]
        normalized_symbols = [str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()]
        code_map = self.code_map(".", max_files=self.max_files)
        import_edges = code_map.get("import_edges", [])
        modules_by_path = {
            str(module.get("path")): str(module.get("module"))
            for module in code_map.get("modules", [])
            if isinstance(module, dict)
        }
        changed_modules = {
            modules_by_path.get(path, "") or module_name(self.workspace_root / path, self.workspace_root)
            for path in normalized_paths
        }
        changed_modules = {module for module in changed_modules if module}
        affected: set[str] = set(normalized_paths)
        for edge in import_edges:
            target = str(edge.get("target") or "")
            if any(
                target == module or target.startswith(f"{module}.") or module.startswith(f"{target}.")
                for module in changed_modules
            ):
                affected.add(str(edge.get("path") or ""))
        references: list[dict[str, Any]] = []
        for symbol in normalized_symbols[:10]:
            found = self.find_references(symbol, max_matches=100)
            symbol_matches = [item for item in found.get("matches", []) if isinstance(item, dict)]
            references.extend(symbol_matches)
            affected.update(str(item.get("path") or "") for item in symbol_matches)
        test_result = (
            relevant_tests(self.store, paths=sorted(affected), symbols=normalized_symbols, max_tests=20)
            if include_tests
            else {"related_tests": [], "verify_commands": []}
        )
        verify_commands = list(test_result.get("verify_commands") or [])
        if not verify_commands and include_tests:
            verify_commands = ["pytest -q"]
        return {
            "input_paths": normalized_paths,
            "input_symbols": normalized_symbols,
            "affected_files": sorted(path for path in affected if path),
            "affected_modules": sorted(changed_modules),
            "references": references[:100],
            "related_tests": list(test_result.get("related_tests") or []),
            "test_relevance": list(test_result.get("tests") or []),
            "verify_commands": verify_commands,
            "code_map": {
                "python_file_count": code_map.get("python_file_count", 0),
                "entrypoints": code_map.get("entrypoints", []),
                "test_file_count": len(code_map.get("test_files", [])),
                "index_version": INDEX_VERSION,
            },
            "index_version": INDEX_VERSION,
        }

    def semantic_code_search(self, query: str, path: str = ".", *, max_matches: int = 20) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        self.refresh()
        result = semantic_search(self.store, query, max_matches=max_matches * 3)
        rel_root = self._relative(self._resolve(path))
        matches = result["matches"]
        if rel_root != ".":
            prefix = rel_root.rstrip("/")
            matches = [
                item
                for item in matches
                if item["path"] == prefix or str(item["path"]).startswith(f"{prefix}/")
            ]
        return {
            **result,
            "matches": matches[:max_matches],
            "truncated": len(matches) > max_matches,
            "index_version": INDEX_VERSION,
        }

    def symbol_search(self, query: str, *, kind: str | None = None, max_results: int = 50) -> dict[str, Any]:
        self.refresh()
        return {
            "query": query,
            "kind": kind,
            "symbols": self.store.symbols(query, kind=kind, limit=max_results),
            "index_version": INDEX_VERSION,
        }

    def symbol_definition(
        self,
        *,
        symbol: str | None = None,
        qualified_name: str | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        self.refresh()
        query = qualified_name or symbol or ""
        if not query.strip():
            raise ValueError("symbol or qualified_name must be provided")
        matches = self.store.symbols(query, limit=20)
        if qualified_name:
            matches = [item for item in matches if item.get("qualified_name") == qualified_name]
        if symbol:
            matches = [
                item
                for item in matches
                if item.get("name") == symbol or str(item.get("qualified_name", "")).endswith(f".{symbol}")
            ]
        if path:
            rel = self._relative(self._resolve(path))
            matches = [item for item in matches if item.get("path") == rel]
        return {"query": query, "definitions": matches, "index_version": INDEX_VERSION}

    def call_graph(
        self,
        *,
        symbol: str | None = None,
        path: str | None = None,
        direction: str = "both",
        depth: int = 1,
    ) -> dict[str, Any]:
        self.refresh()
        if direction not in {"incoming", "outgoing", "both"}:
            raise ValueError("direction must be incoming, outgoing, or both")
        if not symbol and not path:
            raise ValueError("symbol or path must be provided")
        edges = self.store.calls_for(symbol or "", direction=direction, limit=max(100, depth * 200)) if symbol else []
        if path:
            rel = self._relative(self._resolve(path))
            path_edges = self.store.code_map_rows(path_prefix=rel, limit=1_000).get("calls", [])
            edges = [*edges, *path_edges] if edges else path_edges
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for edge in edges:
            key = (edge.get("path"), edge.get("caller_qualified"), edge.get("callee"), edge.get("line"))
            if key not in seen:
                deduped.append(edge)
                seen.add(key)
        return {
            "symbol": symbol,
            "path": path,
            "direction": direction,
            "depth": depth,
            "edges": deduped[:500],
            "index_version": INDEX_VERSION,
        }

    def test_relevance(
        self,
        *,
        paths: list[str] | tuple[str, ...] | None = None,
        symbols: list[str] | tuple[str, ...] | None = None,
        max_tests: int = 20,
    ) -> dict[str, Any]:
        self.refresh()
        normalized_paths = [self._relative(self._resolve(path)) for path in (paths or []) if str(path).strip()]
        normalized_symbols = [str(symbol).strip() for symbol in (symbols or []) if str(symbol).strip()]
        return relevant_tests(self.store, paths=normalized_paths, symbols=normalized_symbols, max_tests=max_tests) | {
            "index_version": INDEX_VERSION
        }

    def _python_files(self) -> list[Path]:
        files: list[Path] = []
        for file_path in self.workspace_root.rglob("*.py"):
            if self._is_excluded(file_path) or not file_path.is_file():
                continue
            try:
                if file_path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            files.append(file_path)
            if len(files) >= self.max_files:
                break
        return files

    def _is_excluded(self, path: Path) -> bool:
        rel = self._relative(path)
        parts = rel.split("/")
        if any(part in DEFAULT_CODEINTEL_EXCLUDES for part in parts):
            return True
        return any(
            rel == excluded or rel.startswith(f"{excluded}/")
            for excluded in DEFAULT_CODEINTEL_EXCLUDES
            if "/" in excluded
        )

    def _resolve(self, path: str | Path) -> Path:
        candidate = (self.workspace_root / str(path)).resolve()
        if not _is_relative_to(candidate, self.workspace_root):
            raise PermissionError(f"Path escapes workspace: {path}")
        return candidate

    def _relative(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.workspace_root)
        value = rel.as_posix()
        return value or "."

    def _test_paths_from_files(self, files: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        for item in files:
            path = str(item.get("path") or "")
            normalized = f"/{path}"
            if Path(path).name.startswith("test_") or "/tests/" in normalized:
                paths.append(path)
        return paths


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
