"""Template and static path helpers."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from fastapi.templating import Jinja2Templates

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _has_frontend_assets(path: Path) -> bool:
    return (path / "templates").is_dir() and (path / "static").is_dir()


def _discover_frontend_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "frontend"
        if _has_frontend_assets(candidate):
            return candidate

    installed_candidate = Path(sys.prefix) / "frontend"
    if _has_frontend_assets(installed_candidate):
        return installed_candidate

    return PACKAGE_ROOT


FRONTEND_ROOT = _discover_frontend_root()


def frontend_root() -> Path:
    return FRONTEND_ROOT


def template_directory() -> Path:
    return frontend_root() / "templates"


def static_directory() -> Path:
    return frontend_root() / "static"


@lru_cache(maxsize=1)
def templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(template_directory()))
