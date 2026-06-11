"""SQLite storage for the local code intelligence index."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .models import FileRecord, ImportEdge, IndexedFile, TestRecord

INDEX_VERSION = "2"


class CodeIndexStore:
    def __init__(self, workspace_root: Path, db_path: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.db_path = db_path or (self.workspace_root / ".solo-agent" / "codeintel" / "index.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    module TEXT NOT NULL,
                    parse_error TEXT NOT NULL DEFAULT '',
                    indexed_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    module TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    parent TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    docstring_summary TEXT NOT NULL,
                    decorators_json TEXT NOT NULL,
                    visibility TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
                CREATE INDEX IF NOT EXISTS idx_symbols_qualified ON symbols(qualified_name);
                CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);
                CREATE TABLE IF NOT EXISTS imports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    module TEXT NOT NULL,
                    target TEXT NOT NULL,
                    name TEXT NOT NULL,
                    line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_imports_path ON imports(path);
                CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target);
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    module TEXT NOT NULL,
                    caller TEXT NOT NULL,
                    caller_qualified TEXT NOT NULL,
                    callee TEXT NOT NULL,
                    resolved_target TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_qualified);
                CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee);
                CREATE INDEX IF NOT EXISTS idx_calls_resolved ON calls(resolved_target);
                CREATE TABLE IF NOT EXISTS references_idx (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    line INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_refs_symbol ON references_idx(symbol);
                CREATE INDEX IF NOT EXISTS idx_refs_path ON references_idx(path);
                CREATE TABLE IF NOT EXISTS tests (
                    path TEXT PRIMARY KEY,
                    test_symbols_json TEXT NOT NULL,
                    fixtures_json TEXT NOT NULL,
                    markers_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS search_docs (
                    doc_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    content TEXT NOT NULL
                );
                """
            )
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS search_docs_fts
                    USING fts5(doc_id UNINDEXED, path, kind, name, qualified_name, content)
                    """
                )
                conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('fts5_enabled', 'true')")
            except sqlite3.OperationalError:
                conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('fts5_enabled', 'false')")
            conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('index_version', ?)", (INDEX_VERSION,))
            conn.commit()

    def file_record(self, path: str) -> FileRecord | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()
        return _file_from_row(row) if row else None

    def indexed_paths(self) -> set[str]:
        with self.connect() as conn:
            return {str(row["path"]) for row in conn.execute("SELECT path FROM files")}

    def replace_file(self, indexed: IndexedFile) -> None:
        path = indexed.file.path
        now = time.time()
        with self.connect() as conn:
            self._delete_path(conn, path)
            conn.execute(
                """
                INSERT INTO files(path, sha256, mtime, size, module, parse_error, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    indexed.file.path,
                    indexed.file.sha256,
                    indexed.file.mtime,
                    indexed.file.size,
                    indexed.file.module,
                    indexed.file.parse_error,
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO symbols(
                    id, name, qualified_name, kind, path, module, line_start, line_end,
                    parent, signature, docstring_summary, decorators_json, visibility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.id,
                        item.name,
                        item.qualified_name,
                        item.kind,
                        item.path,
                        item.module,
                        item.line_start,
                        item.line_end,
                        item.parent,
                        item.signature,
                        item.docstring_summary,
                        json.dumps(item.decorators),
                        item.visibility,
                    )
                    for item in indexed.symbols
                ],
            )
            conn.executemany(
                "INSERT INTO imports(path, module, target, name, line) VALUES (?, ?, ?, ?, ?)",
                [(item.path, item.module, item.target, item.name, item.line) for item in indexed.imports],
            )
            conn.executemany(
                """
                INSERT INTO calls(path, module, caller, caller_qualified, callee, resolved_target, confidence, line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.path,
                        item.module,
                        item.caller,
                        item.caller_qualified,
                        item.callee,
                        item.resolved_target,
                        item.confidence,
                        item.line,
                    )
                    for item in indexed.calls
                ],
            )
            conn.executemany(
                "INSERT INTO references_idx(path, symbol, line, kind, text) VALUES (?, ?, ?, ?, ?)",
                [(item.path, item.symbol, item.line, item.kind, item.text) for item in indexed.references],
            )
            conn.executemany(
                "INSERT INTO tests(path, test_symbols_json, fixtures_json, markers_json) VALUES (?, ?, ?, ?)",
                [
                    (
                        item.path,
                        json.dumps(item.test_symbols),
                        json.dumps(item.fixtures),
                        json.dumps(item.markers),
                    )
                    for item in indexed.tests
                ],
            )
            conn.executemany(
                "INSERT INTO search_docs(doc_id, path, kind, name, qualified_name, content) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["doc_id"],
                        item["path"],
                        item["kind"],
                        item["name"],
                        item["qualified_name"],
                        item["content"],
                    )
                    for item in indexed.search_documents
                ],
            )
            if self.fts_enabled(conn):
                conn.executemany(
                    "INSERT INTO search_docs_fts(doc_id, path, kind, name, qualified_name, content) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            item["doc_id"],
                            item["path"],
                            item["kind"],
                            item["name"],
                            item["qualified_name"],
                            item["content"],
                        )
                        for item in indexed.search_documents
                    ],
                )
            conn.commit()

    def delete_paths(self, paths: set[str]) -> None:
        with self.connect() as conn:
            for path in paths:
                self._delete_path(conn, path)
            conn.commit()

    def _delete_path(self, conn: sqlite3.Connection, path: str) -> None:
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.execute("DELETE FROM symbols WHERE path = ?", (path,))
        conn.execute("DELETE FROM imports WHERE path = ?", (path,))
        conn.execute("DELETE FROM calls WHERE path = ?", (path,))
        conn.execute("DELETE FROM references_idx WHERE path = ?", (path,))
        conn.execute("DELETE FROM tests WHERE path = ?", (path,))
        conn.execute("DELETE FROM search_docs WHERE path = ?", (path,))
        if self.fts_enabled(conn):
            conn.execute("DELETE FROM search_docs_fts WHERE path = ?", (path,))

    def fts_enabled(self, conn: sqlite3.Connection | None = None) -> bool:
        close = False
        if conn is None:
            conn = self.connect()
            close = True
        try:
            row = conn.execute("SELECT value FROM metadata WHERE key = 'fts5_enabled'").fetchone()
            return bool(row and row["value"] == "true")
        finally:
            if close:
                conn.close()

    def metadata(self) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute("SELECT key, value FROM metadata").fetchall()
            counts = {
                "file_count": conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "python_file_count": conn.execute("SELECT COUNT(*) FROM files WHERE path LIKE '%.py'").fetchone()[0],
                "symbol_count": conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
                "call_edge_count": conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
                "import_edge_count": conn.execute("SELECT COUNT(*) FROM imports").fetchone()[0],
                "test_file_count": conn.execute("SELECT COUNT(*) FROM tests").fetchone()[0],
                "parse_error_count": conn.execute("SELECT COUNT(*) FROM files WHERE parse_error != ''").fetchone()[0],
            }
        data = {str(row["key"]): str(row["value"]) for row in rows}
        return data | counts | {"db_path": str(self.db_path)}

    def code_map_rows(self, path_prefix: str = "", limit: int = 500) -> dict[str, list[dict[str, Any]]]:
        where, params = _path_filter(path_prefix)
        with self.connect() as conn:
            files = [
                _dict(row)
                for row in conn.execute(f"SELECT * FROM files {where} ORDER BY path LIMIT ?", (*params, limit))
            ]
            modules = files
            symbols = [
                _symbol_dict(row)
                for row in conn.execute(
                    f"SELECT * FROM symbols {where} ORDER BY path, line_start LIMIT ?",
                    (*params, limit * 8),
                )
            ]
            imports = [
                _dict(row)
                for row in conn.execute(
                    f"SELECT path, module, target, name, line FROM imports {where} ORDER BY path, line LIMIT ?",
                    (*params, limit * 12),
                )
            ]
            calls = [
                _dict(row)
                for row in conn.execute(
                    f"""
                    SELECT path, module, caller, caller_qualified, callee, resolved_target, confidence, line
                    FROM calls {where}
                    ORDER BY path, line LIMIT ?
                    """,
                    (*params, limit * 12),
                )
            ]
            if where:
                parse_error_rows = conn.execute(
                    f"SELECT path, parse_error FROM files {where} AND parse_error != '' ORDER BY path",
                    params,
                )
            else:
                parse_error_rows = conn.execute("SELECT path, parse_error FROM files WHERE parse_error != '' ORDER BY path")
            parse_errors = [{"path": row["path"], "error": row["parse_error"]} for row in parse_error_rows]
            tests = [
                str(row["path"])
                for row in conn.execute(f"SELECT path FROM tests {where} ORDER BY path LIMIT 100", (*params,))
            ]
        return {
            "files": files,
            "modules": modules,
            "symbols": symbols,
            "imports": imports,
            "calls": calls,
            "parse_errors": parse_errors,
            "tests": tests,
        }

    def symbols(self, query: str = "", kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if query:
            clauses.append("(name LIKE ? OR qualified_name LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM symbols {where} ORDER BY path, line_start LIMIT ?", (*params, limit)).fetchall()
        return [_symbol_dict(row) for row in rows]

    def references(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT path, symbol, line, kind, text FROM references_idx WHERE symbol = ? ORDER BY path, line LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [_dict(row) for row in rows]

    def imports(self) -> list[ImportEdge]:
        with self.connect() as conn:
            rows = conn.execute("SELECT path, module, target, name, line FROM imports").fetchall()
        return [ImportEdge(**_dict(row)) for row in rows]

    def calls_for(self, symbol: str, direction: str, limit: int = 500) -> list[dict[str, Any]]:
        like = f"%{symbol}%"
        if direction == "incoming":
            sql = "SELECT * FROM calls WHERE callee LIKE ? OR resolved_target LIKE ? ORDER BY path, line LIMIT ?"
        elif direction == "outgoing":
            sql = "SELECT * FROM calls WHERE caller LIKE ? OR caller_qualified LIKE ? ORDER BY path, line LIMIT ?"
        else:
            sql = """
                SELECT * FROM calls
                WHERE caller LIKE ? OR caller_qualified LIKE ? OR callee LIKE ? OR resolved_target LIKE ?
                ORDER BY path, line LIMIT ?
            """
            with self.connect() as conn:
                return [_dict(row) for row in conn.execute(sql, (like, like, like, like, limit)).fetchall()]
        with self.connect() as conn:
            return [_dict(row) for row in conn.execute(sql, (like, like, limit)).fetchall()]

    def tests(self) -> list[TestRecord]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM tests ORDER BY path").fetchall()
        return [
            TestRecord(
                path=str(row["path"]),
                test_symbols=json.loads(row["test_symbols_json"]),
                fixtures=json.loads(row["fixtures_json"]),
                markers=json.loads(row["markers_json"]),
            )
            for row in rows
        ]

    def search_docs_like(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = [term for term in query.replace('"', " ").split() if term]
        if not terms:
            return []
        with self.connect() as conn:
            if self.fts_enabled(conn):
                fts_query = " OR ".join(term.replace("'", "''") for term in terms)
                try:
                    rows = conn.execute(
                        """
                        SELECT doc_id, path, kind, name, qualified_name,
                               snippet(search_docs_fts, 5, '', '', '...', 18) AS snippet,
                               bm25(search_docs_fts) AS rank
                        FROM search_docs_fts
                        WHERE search_docs_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, limit),
                    ).fetchall()
                    return [_dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            like = f"%{query}%"
            rows = conn.execute(
                """
                SELECT doc_id, path, kind, name, qualified_name, substr(content, 1, 280) AS snippet, 1.0 AS rank
                FROM search_docs
                WHERE content LIKE ? OR path LIKE ? OR qualified_name LIKE ?
                ORDER BY path
                LIMIT ?
                """,
                (like, like, like, limit),
            ).fetchall()
        return [_dict(row) for row in rows]


def _file_from_row(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        path=str(row["path"]),
        sha256=str(row["sha256"]),
        mtime=float(row["mtime"]),
        size=int(row["size"]),
        module=str(row["module"]),
        parse_error=str(row["parse_error"] or ""),
    )


def _dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _symbol_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = _dict(row)
    data["decorators"] = json.loads(str(data.pop("decorators_json") or "[]"))
    data["line"] = data["line_start"]
    return data


def _path_filter(path_prefix: str) -> tuple[str, tuple[str, ...]]:
    if not path_prefix or path_prefix == ".":
        return "", ()
    prefix = path_prefix.rstrip("/")
    return "WHERE (path = ? OR path LIKE ?)", (prefix, f"{prefix}/%")
