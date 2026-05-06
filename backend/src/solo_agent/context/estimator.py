from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib import import_module
from math import ceil
from typing import Any, Literal

EstimateKind = Literal["text", "code", "mixed"]

_FENCED_CODE_RE = re.compile(
    r"(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)\r?\n(?P<body>.*?)(?:\r?\n(?P=fence)[ \t]*(?=\r?\n|$)|$)",
    re.DOTALL,
)


@dataclass(frozen=True)
class TokenEstimate:
    tokens: int
    utf8_bytes: int
    kind: EstimateKind
    language: str | None = None
    used_tree_sitter: bool = False
    cache_hit: bool = False
    parts: tuple[TokenEstimate, ...] = field(default_factory=tuple)


class ContextTokenEstimator:
    """上下文 token 粗估器。

    这里避免绑定具体模型 tokenizer：普通文本使用 bytes/4 的保守近似；
    代码在可用时借助 Tree-sitter 语法叶子节点做结构化估算。
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str | None, EstimateKind], TokenEstimate] = {}

    def estimate_text(self, text: str, *, kind: EstimateKind = "text", language: str | None = None) -> TokenEstimate:
        cache_key = self._cache_key(text, kind=kind, language=language)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return replace(cached, cache_hit=True)

        if kind == "mixed":
            estimate = self._estimate_mixed(text)
        elif kind == "code":
            estimate = self._estimate_code(text, language=language)
        else:
            estimate = self._estimate_plain_text(text, kind="text", language=language)

        self._cache[cache_key] = estimate
        return estimate

    def clear_cache(self) -> None:
        self._cache.clear()

    def estimate_state(self, state: Any) -> int:
        """按当前 AgentState 快照估算整段上下文预算。"""

        import json

        payload = state.snapshot() if hasattr(state, "snapshot") else getattr(state, "__dict__", state)
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return self.estimate_text(text, kind="mixed").tokens

    def _estimate_mixed(self, text: str) -> TokenEstimate:
        parts: list[TokenEstimate] = []
        cursor = 0

        for match in _FENCED_CODE_RE.finditer(text):
            if match.start() > cursor:
                parts.append(self.estimate_text(text[cursor : match.start()], kind="text"))

            language = _normalize_language(match.group("info"))
            parts.append(self.estimate_text(match.group("body"), kind="code", language=language))
            cursor = match.end()

        if cursor < len(text):
            parts.append(self.estimate_text(text[cursor:], kind="text"))

        return TokenEstimate(
            tokens=sum(part.tokens for part in parts),
            utf8_bytes=len(text.encode("utf-8")),
            kind="mixed",
            parts=tuple(parts),
        )

    def _estimate_plain_text(self, text: str, *, kind: EstimateKind, language: str | None = None) -> TokenEstimate:
        utf8_bytes = len(text.encode("utf-8"))
        return TokenEstimate(
            tokens=ceil(utf8_bytes / 4) if utf8_bytes else 0,
            utf8_bytes=utf8_bytes,
            kind=kind,
            language=language,
        )

    def _estimate_code(self, code: str, *, language: str | None) -> TokenEstimate:
        parser = self._load_parser(language)
        if parser is None:
            return self._estimate_plain_text(code, kind="code", language=language)

        try:
            tree = parser.parse(code.encode("utf-8"))
            syntax_tokens = _count_leaf_nodes(tree.root_node)
        except Exception:
            return self._estimate_plain_text(code, kind="code", language=language)

        utf8_bytes = len(code.encode("utf-8"))
        return TokenEstimate(
            tokens=syntax_tokens,
            utf8_bytes=utf8_bytes,
            kind="code",
            language=language,
            used_tree_sitter=True,
        )

    def _load_parser(self, language: str | None) -> Any | None:
        if not language:
            return None

        try:
            import_module("tree_sitter")
            language_pack = import_module("tree_sitter_language_pack")
        except Exception:
            return None

        normalized = _normalize_language(language)
        if normalized is None:
            return None

        try:
            get_parser = getattr(language_pack, "get_parser", None)
            if get_parser is not None:
                return get_parser(normalized)

            get_language = getattr(language_pack, "get_language", None)
            if get_language is None:
                return None

            parser_type = import_module("tree_sitter").Parser
            parser = parser_type()
            parser.language = get_language(normalized)
        except Exception:
            return None

        return parser

    @staticmethod
    def _cache_key(text: str, *, kind: EstimateKind, language: str | None) -> tuple[str, str | None, EstimateKind]:
        digest = sha256(text.encode("utf-8")).hexdigest()
        return digest, _normalize_language(language), kind


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None

    value = language.strip().split(maxsplit=1)[0].lower()
    if not value:
        return None

    aliases = {
        "js": "javascript",
        "py": "python",
        "rb": "ruby",
        "rs": "rust",
        "sh": "bash",
        "shell": "bash",
        "ts": "typescript",
    }
    return aliases.get(value, value)


def _count_leaf_nodes(root: Any) -> int:
    leaves = 0
    stack = [root]

    while stack:
        node = stack.pop()
        children = tuple(getattr(node, "named_children", None) or getattr(node, "children", ()) or ())

        if children:
            stack.extend(children)
            continue

        # Tree-sitter 叶子节点近似代码 token；空源码至少不能制造 token。
        if getattr(node, "start_byte", 0) != getattr(node, "end_byte", 0):
            leaves += 1

    return leaves
