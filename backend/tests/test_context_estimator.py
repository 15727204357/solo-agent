from __future__ import annotations

import sys
from types import ModuleType

from solo_agent.context import ContextTokenEstimator


def test_plain_text_estimate_uses_utf8_bytes_ceiling() -> None:
    estimator = ContextTokenEstimator()

    estimate = estimator.estimate_text("你好a")

    assert estimate.tokens == 2
    assert estimate.utf8_bytes == 7
    assert estimate.kind == "text"
    assert not estimate.cache_hit


def test_cache_key_uses_hash_language_and_kind() -> None:
    estimator = ContextTokenEstimator()

    first = estimator.estimate_text("print('hi')", kind="code", language="python")
    second = estimator.estimate_text("print('hi')", kind="code", language="python")
    different_kind = estimator.estimate_text("print('hi')", kind="text", language="python")

    assert not first.cache_hit
    assert second.cache_hit
    assert not different_kind.cache_hit
    assert different_kind.kind == "text"


def test_code_estimate_falls_back_without_tree_sitter(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tree_sitter", None)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    estimator = ContextTokenEstimator()

    estimate = estimator.estimate_text("x = 1", kind="code", language="python")

    assert estimate.tokens == 2
    assert estimate.kind == "code"
    assert estimate.language == "python"
    assert not estimate.used_tree_sitter


def test_code_estimate_uses_tree_sitter_when_available(monkeypatch) -> None:
    tree_sitter = ModuleType("tree_sitter")
    language_pack = ModuleType("tree_sitter_language_pack")

    class FakeNode:
        def __init__(self, start_byte: int, end_byte: int, children: tuple[FakeNode, ...] = ()) -> None:
            self.start_byte = start_byte
            self.end_byte = end_byte
            self.named_children = children

    class FakeParser:
        def parse(self, source: bytes):
            assert source == b"x = 1"
            root = FakeNode(0, 5, (FakeNode(0, 1), FakeNode(4, 5)))
            return type("FakeTree", (), {"root_node": root})()

    def get_parser(language: str) -> FakeParser:
        assert language == "python"
        return FakeParser()

    language_pack.get_parser = get_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter", tree_sitter)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", language_pack)
    estimator = ContextTokenEstimator()

    estimate = estimator.estimate_text("x = 1", kind="code", language="py")

    assert estimate.tokens == 2
    assert estimate.language == "py"
    assert estimate.used_tree_sitter


def test_mixed_text_detects_fenced_code_blocks(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "tree_sitter", None)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", None)
    estimator = ContextTokenEstimator()

    estimate = estimator.estimate_text("说明\n```python\nx = 1\n```\n结束", kind="mixed")

    assert estimate.kind == "mixed"
    assert estimate.tokens == sum(part.tokens for part in estimate.parts)
    assert [part.kind for part in estimate.parts] == ["text", "code", "text"]
    assert estimate.parts[1].language == "python"
