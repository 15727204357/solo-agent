"""Data models for the local code intelligence index."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    mtime: float
    size: int
    module: str
    parse_error: str = ""


@dataclass(frozen=True)
class SymbolRecord:
    id: str
    name: str
    qualified_name: str
    kind: str
    path: str
    module: str
    line_start: int
    line_end: int
    parent: str = ""
    signature: str = ""
    docstring_summary: str = ""
    decorators: list[str] = field(default_factory=list)
    visibility: str = "public"


@dataclass(frozen=True)
class ImportEdge:
    path: str
    module: str
    target: str
    line: int
    name: str = ""


@dataclass(frozen=True)
class CallEdge:
    path: str
    module: str
    caller: str
    callee: str
    line: int
    caller_qualified: str = ""
    resolved_target: str = ""
    confidence: float = 0.25


@dataclass(frozen=True)
class ReferenceRecord:
    path: str
    symbol: str
    line: int
    kind: str
    text: str


@dataclass(frozen=True)
class TestRecord:
    path: str
    test_symbols: list[str] = field(default_factory=list)
    fixtures: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IndexedFile:
    file: FileRecord
    symbols: list[SymbolRecord] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    references: list[ReferenceRecord] = field(default_factory=list)
    tests: list[TestRecord] = field(default_factory=list)
    search_documents: list[dict[str, str]] = field(default_factory=list)


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
