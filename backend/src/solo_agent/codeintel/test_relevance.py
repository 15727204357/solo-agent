"""Heuristic test relevance scoring over the indexed Python graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .index_store import CodeIndexStore


def relevant_tests(
    store: CodeIndexStore,
    *,
    paths: list[str],
    symbols: list[str],
    max_tests: int = 20,
) -> dict[str, Any]:
    tests = store.tests()
    if not tests:
        return {"input_paths": paths, "input_symbols": symbols, "tests": [], "related_tests": [], "verify_commands": []}

    imports = store.imports()
    changed_stems = {Path(path).stem.removeprefix("test_") for path in paths if path}
    changed_modules = {
        str(row.get("module") or "")
        for row in store.code_map_rows(limit=10_000).get("modules", [])
        if str(row.get("path") or "") in set(paths)
    }
    symbol_refs: dict[str, set[str]] = {}
    for symbol in symbols:
        symbol_refs[symbol] = {str(ref.get("path") or "") for ref in store.references(symbol, limit=500)}

    scored: list[dict[str, Any]] = []
    for test in tests:
        score = 0.0
        reasons: list[str] = []
        test_path = test.path
        test_stem = Path(test_path).stem.removeprefix("test_")
        for stem in changed_stems:
            if stem and (stem in test_stem or stem in test_path):
                score += 4.0
                reasons.append(f"path_stem:{stem}")
        imported_targets = [
            edge.target
            for edge in imports
            if edge.path == test_path
        ]
        for module in changed_modules:
            if module and any(
                target == module or target.startswith(f"{module}.") or module.startswith(f"{target}.")
                for target in imported_targets
            ):
                score += 5.0
                reasons.append(f"imports:{module}")
        for symbol, ref_paths in symbol_refs.items():
            if test_path in ref_paths:
                score += 4.0
                reasons.append(f"symbol_ref:{symbol}")
        if test.fixtures:
            score += min(len(test.fixtures), 3) * 0.5
            reasons.append("fixtures")
        if test.markers:
            score += min(len(test.markers), 3) * 0.25
            reasons.append("markers")
        if not paths and not symbols:
            score += 1.0
            reasons.append("fallback")
        if score > 0:
            scored.append(
                {
                    "path": test_path,
                    "score": round(score, 3),
                    "reasons": sorted(set(reasons)),
                    "test_symbols": test.test_symbols[:20],
                    "fixtures": test.fixtures[:20],
                    "markers": test.markers[:20],
                    "command": _pytest_command(test_path, test.test_symbols),
                }
            )
    if not scored:
        scored = [
            {
                "path": test.path,
                "score": 0.1,
                "reasons": ["fallback_all_tests"],
                "test_symbols": test.test_symbols[:20],
                "fixtures": test.fixtures[:20],
                "markers": test.markers[:20],
                "command": _pytest_command(test.path, test.test_symbols),
            }
            for test in tests[:max_tests]
        ]
    scored.sort(key=lambda item: (-float(item["score"]), str(item["path"])))
    selected = scored[:max_tests]
    return {
        "input_paths": paths,
        "input_symbols": symbols,
        "tests": selected,
        "related_tests": [str(item["path"]) for item in selected],
        "verify_commands": [str(item["command"]) for item in selected[:5]],
    }


def _pytest_command(path: str, test_symbols: list[str]) -> str:
    if len(test_symbols) == 1:
        return f"pytest -q {path}::{test_symbols[0]}"
    return f"pytest -q {path}"
