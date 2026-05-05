"""Tool registry and read-only tools for Solo Agent."""

from .readonly import WorkspaceTools, list_files, read_file, search_text
from .registry import ToolRegistry, ToolSpec, create_default_registry

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "WorkspaceTools",
    "create_default_registry",
    "list_files",
    "read_file",
    "search_text",
]
