from __future__ import annotations

from pathlib import Path

from solo_agent.codeintel import CodeIntelligenceService
from solo_agent.tools import create_default_registry


def test_python_codeintel_indexes_symbols_imports_calls_and_tests(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    tests = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "service.py").write_text(
        '"""Greeting service."""\n\n'
        "from pathlib import Path\n\n"
        "SERVICE_NAME = 'greeting'\n\n"
        "class Service:\n"
        "    def greet(self, name: str) -> str:\n"
        "        return format_greeting(name)\n\n"
        "def format_greeting(name: str) -> str:\n"
        "    Path('.')\n"
        "    return f'hello {name}'\n",
        encoding="utf-8",
    )
    (tests / "test_service.py").write_text(
        "import pytest\n"
        "from pkg.service import Service\n\n"
        "@pytest.fixture\n"
        "def service():\n"
        "    return Service()\n\n"
        "@pytest.mark.unit\n"
        "def test_greet(service):\n"
        "    assert service.greet('Ada') == 'hello Ada'\n",
        encoding="utf-8",
    )

    service = CodeIntelligenceService(tmp_path, index_ttl_seconds=3600)
    mapped = service.code_map(max_files=20)

    assert mapped["backend"] == "python_lsp_like"
    assert mapped["index_version"] == "2"
    assert any(symbol["qualified_name"] == "pkg.service.Service.greet" for symbol in mapped["symbols"])
    assert any(symbol["kind"] == "constant" and symbol["name"] == "SERVICE_NAME" for symbol in mapped["symbols"])
    assert any(edge["target"] == "pathlib" for edge in mapped["import_edges"])
    assert any(
        edge["callee"] == "format_greeting" and edge["resolved_target"] == "pkg.service.format_greeting"
        for edge in mapped["call_edges"]
    )
    assert "tests/test_service.py" in mapped["test_files"]

    relevance = service.test_relevance(paths=["pkg/service.py"], symbols=["Service"])
    assert relevance["related_tests"] == ["tests/test_service.py"]
    assert relevance["tests"][0]["reasons"]
    assert relevance["verify_commands"][0].startswith("pytest -q tests/test_service.py")


def test_codeintel_incremental_refresh_and_syntax_errors(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    target = package / "service.py"
    target.write_text("def old_name():\n    return 1\n", encoding="utf-8")
    broken = package / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")

    service = CodeIntelligenceService(tmp_path, index_ttl_seconds=3600)
    first = service.status(refresh=True)
    assert first["parse_error_count"] == 1
    assert service.symbol_search("old_name")["symbols"]

    target.write_text("def new_name():\n    return 2\n", encoding="utf-8")
    second = service.status(refresh=True)
    assert second["changed_files_indexed"] >= 1
    assert service.symbol_search("old_name")["symbols"] == []
    assert service.symbol_search("new_name")["symbols"][0]["qualified_name"] == "pkg.service.new_name"

    broken.unlink()
    third = service.status(refresh=True)
    assert third["deleted_files_removed"] >= 1
    assert third["parse_error_count"] == 0


def test_codeintel_registry_exposes_new_readonly_tools(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def helper():\n"
        "    return 1\n\n"
        "def main():\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    registry = create_default_registry(tmp_path)

    status = registry.call("code_index_status", {"refresh": True})
    symbols = registry.call("symbol_search", {"query": "helper"})
    definition = registry.call("symbol_definition", {"symbol": "helper"})
    graph = registry.call("call_graph", {"symbol": "helper", "direction": "both"})
    semantic = registry.call("semantic_code_search", {"query": "main helper"})

    assert status["ok"] is True
    assert status["result"]["backend"] == "python_lsp_like"
    assert symbols["result"]["symbols"][0]["qualified_name"] == "app.helper"
    assert definition["result"]["definitions"][0]["path"] == "app.py"
    assert any(edge["callee"] == "helper" for edge in graph["result"]["edges"])
    assert semantic["result"]["matches"][0]["reason"]["symbol_score"] >= 0
