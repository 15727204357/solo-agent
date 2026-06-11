"""Explainable local retrieval over indexed code documents."""

from __future__ import annotations

import re
from typing import Any

from .index_store import CodeIndexStore

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[A-Za-z0-9]+")


def semantic_search(store: CodeIndexStore, query: str, *, max_matches: int = 20) -> dict[str, Any]:
    query_terms = {term.casefold() for term in _TOKEN_RE.findall(query)}
    raw = store.search_docs_like(query, max(max_matches * 4, 20))
    scored: list[dict[str, Any]] = []
    for item in raw:
        path = str(item.get("path") or "")
        name = str(item.get("name") or "")
        qualified = str(item.get("qualified_name") or "")
        haystack = f"{path} {name} {qualified} {item.get('snippet') or ''}".casefold()
        matched = sorted(term for term in query_terms if term in haystack)
        path_score = sum(3 for term in query_terms if term in path.casefold())
        symbol_score = sum(4 for term in query_terms if term in f"{name} {qualified}".casefold())
        text_score = len(matched)
        fts_rank = float(item.get("rank") or 0.0)
        score = path_score + symbol_score + text_score + max(0.0, 2.0 - min(abs(fts_rank), 2.0))
        if score <= 0 and not matched:
            continue
        scored.append(
            {
                "path": path,
                "score": round(score, 3),
                "kind": item.get("kind") or "file",
                "name": name,
                "qualified_name": qualified,
                "matched_terms": matched,
                "snippet": str(item.get("snippet") or "").replace("\n", " ")[:500],
                "reason": {
                    "path_score": path_score,
                    "symbol_score": symbol_score,
                    "text_score": text_score,
                    "fts_rank": fts_rank,
                },
            }
        )
    scored.sort(key=lambda value: (-float(value["score"]), str(value["path"]), str(value.get("qualified_name") or "")))
    return {
        "query": query,
        "matches": scored[:max_matches],
        "truncated": len(scored) > max_matches,
        "engine": "sqlite_fts5_bm25",
    }
