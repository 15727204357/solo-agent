"""Python AST indexing for the local code intelligence service."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from .models import CallEdge, FileRecord, ImportEdge, IndexedFile, ReferenceRecord, SymbolRecord, TestRecord

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


class PythonIndexer:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()

    def index_file(self, file_path: Path) -> IndexedFile:
        file_path = file_path.resolve()
        rel_path = _relative(file_path, self.workspace_root)
        source = file_path.read_text(encoding="utf-8", errors="replace")
        stat = file_path.stat()
        module = module_name(file_path, self.workspace_root)
        file_record = FileRecord(
            path=rel_path,
            sha256=hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest(),
            mtime=stat.st_mtime,
            size=stat.st_size,
            module=module,
        )
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return IndexedFile(
                file=FileRecord(**{**file_record.__dict__, "parse_error": str(exc)}),
                search_documents=[_file_search_doc(rel_path, module, source, parse_error=str(exc))],
            )

        visitor = _PythonVisitor(rel_path, module, source)
        visitor.visit(tree)
        visitor.resolve_calls()
        documents = [_file_search_doc(rel_path, module, source)]
        for symbol in visitor.symbols:
            documents.append(
                {
                    "doc_id": f"symbol:{symbol.id}",
                    "path": symbol.path,
                    "kind": symbol.kind,
                    "name": symbol.name,
                    "qualified_name": symbol.qualified_name,
                    "content": " ".join(
                        part
                        for part in [
                            symbol.path,
                            symbol.module,
                            symbol.kind,
                            symbol.name,
                            symbol.qualified_name,
                            symbol.signature,
                            symbol.docstring_summary,
                            " ".join(symbol.decorators),
                        ]
                        if part
                    ),
                }
            )
        return IndexedFile(
            file=file_record,
            symbols=visitor.symbols,
            imports=visitor.imports,
            calls=visitor.calls,
            references=visitor.references,
            tests=visitor.tests(),
            search_documents=documents,
        )


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, path: str, module: str, source: str) -> None:
        self.path = path
        self.module = module
        self.source = source
        self.lines = source.splitlines()
        self.scope: list[str] = []
        self.symbols: list[SymbolRecord] = []
        self.imports: list[ImportEdge] = []
        self.calls: list[CallEdge] = []
        self.references: list[ReferenceRecord] = []
        self.import_aliases: dict[str, str] = {}
        self.local_symbols: dict[str, str] = {}
        self.test_symbols: list[str] = []
        self.fixtures: list[str] = []
        self.markers: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.import_aliases[name] = alias.name
            self.imports.append(ImportEdge(self.path, self.module, alias.name, node.lineno, name=name))
            self.references.append(_reference(self.path, name, node.lineno, "import", self.lines))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target_module = "." * node.level + (node.module or "")
        for alias in node.names:
            name = alias.asname or alias.name
            target = f"{target_module}.{alias.name}" if target_module and alias.name != "*" else target_module
            self.import_aliases[name] = target
            self.imports.append(ImportEdge(self.path, self.module, target_module, node.lineno, name=name))
            self.references.append(_reference(self.path, name, node.lineno, "import", self.lines))
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_symbol("class", node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_symbol("function", node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_symbol("function", node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope:
            for target in node.targets:
                for name in _target_names(target):
                    self._add_assignment_symbol(name, node.lineno, getattr(node, "end_lineno", node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self.scope:
            for name in _target_names(node.target):
                self._add_assignment_symbol(name, node.lineno, getattr(node, "end_lineno", node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = call_name(node.func)
        if callee:
            caller = ".".join(self.scope)
            self.calls.append(
                CallEdge(
                    path=self.path,
                    module=self.module,
                    caller=caller,
                    caller_qualified=f"{self.module}.{caller}" if caller else self.module,
                    callee=callee,
                    line=getattr(node, "lineno", 0),
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.references.append(_reference(self.path, node.id, getattr(node, "lineno", 0), "reference", self.lines))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.references.append(_reference(self.path, node.attr, getattr(node, "lineno", 0), "reference", self.lines))
        self.generic_visit(node)

    def _visit_symbol(self, kind: str, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = ".".join(self.scope)
        local_qualified = ".".join([*self.scope, node.name])
        qualified_name = f"{self.module}.{local_qualified}"
        decorators = [_unparse(item) for item in getattr(node, "decorator_list", [])]
        actual_kind = _symbol_kind(kind, parent, decorators)
        signature = _signature(node)
        symbol = SymbolRecord(
            id=f"{self.path}:{node.lineno}:{node.name}",
            name=node.name,
            qualified_name=qualified_name,
            kind=actual_kind,
            path=self.path,
            module=self.module,
            line_start=node.lineno,
            line_end=getattr(node, "end_lineno", node.lineno),
            parent=parent,
            signature=signature,
            docstring_summary=_docstring_summary(ast.get_docstring(node)),
            decorators=decorators,
            visibility="private" if node.name.startswith("_") and not node.name.startswith("__") else "public",
        )
        self.symbols.append(symbol)
        self.local_symbols[node.name] = qualified_name
        self.references.append(_reference(self.path, node.name, node.lineno, "definition", self.lines))
        if node.name.startswith("test_") or self.path.replace("\\", "/").split("/")[-1].startswith("test_"):
            self.test_symbols.append(node.name)
        if any(
            _decorator_name(item).endswith("pytest.fixture") or _decorator_name(item) == "fixture"
            for item in node.decorator_list
        ):
            self.fixtures.append(node.name)
        self.markers.extend(
            marker.removeprefix("pytest.mark.")
            for marker in (_decorator_name(item) for item in getattr(node, "decorator_list", []))
            if marker.startswith("pytest.mark.")
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _add_assignment_symbol(self, name: str, line_start: int, line_end: int) -> None:
        qualified_name = f"{self.module}.{name}"
        self.symbols.append(
            SymbolRecord(
                id=f"{self.path}:{line_start}:{name}",
                name=name,
                qualified_name=qualified_name,
                kind="constant" if name.isupper() else "variable",
                path=self.path,
                module=self.module,
                line_start=line_start,
                line_end=line_end,
                visibility="private" if name.startswith("_") else "public",
            )
        )
        self.local_symbols[name] = qualified_name
        self.references.append(_reference(self.path, name, line_start, "definition", self.lines))

    def resolve_calls(self) -> None:
        resolved: list[CallEdge] = []
        for call in self.calls:
            head = call.callee.split(".")[0]
            target = self.local_symbols.get(call.callee) or self.local_symbols.get(head) or self.import_aliases.get(head, "")
            confidence = 0.85 if target in self.local_symbols.values() else 0.55 if target else 0.25
            resolved.append(CallEdge(**{**call.__dict__, "resolved_target": target, "confidence": confidence}))
        self.calls = resolved

    def tests(self) -> list[TestRecord]:
        normalized = self.path.replace("\\", "/")
        is_test = normalized.split("/")[-1].startswith("test_") or "/tests/" in f"/{normalized}"
        if not is_test:
            return []
        return [TestRecord(self.path, sorted(set(self.test_symbols)), sorted(set(self.fixtures)), sorted(set(self.markers)))]


def module_name(file_path: Path, workspace_root: Path) -> str:
    rel = file_path.resolve().relative_to(workspace_root.resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _signature(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(_unparse(base) for base in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({_unparse(node.args)}){returns}"


def _symbol_kind(kind: str, parent: str, decorators: list[str]) -> str:
    if kind == "function" and parent:
        return "method"
    if any(item.endswith("pytest.fixture") or item == "fixture" for item in decorators):
        return "fixture"
    return kind


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return call_name(node.func)
    return call_name(node)


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for item in node.elts:
            names.extend(_target_names(item))
        return names
    return []


def _reference(path: str, symbol: str, line: int, kind: str, lines: list[str]) -> ReferenceRecord:
    text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
    return ReferenceRecord(path=path, symbol=symbol, line=line, kind=kind, text=text)


def _file_search_doc(path: str, module: str, source: str, *, parse_error: str = "") -> dict[str, str]:
    return {
        "doc_id": f"file:{path}",
        "path": path,
        "kind": "file",
        "name": Path(path).name,
        "qualified_name": module,
        "content": f"{path}\n{module}\n{parse_error}\n{source[:120_000]}",
    }


def _docstring_summary(docstring: str | None) -> str:
    if not docstring:
        return ""
    return " ".join(docstring.strip().split())[:500]


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
