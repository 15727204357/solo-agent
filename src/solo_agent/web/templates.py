"""Template and static path helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi.templating import Jinja2Templates

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def template_directory() -> Path:
    return PACKAGE_ROOT / "templates"


def static_directory() -> Path:
    return PACKAGE_ROOT / "static"


@lru_cache(maxsize=1)
def templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(template_directory()))

