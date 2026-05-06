"""FastAPI application factory for the Solo Agent Web MVP."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from solo_agent.settings import get_settings
from solo_agent.web.routes import router
from solo_agent.web.templates import static_directory


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")

    app.mount("/static", StaticFiles(directory=static_directory()), name="static")
    app.include_router(router)

    return app


app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "solo_agent.web.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()

