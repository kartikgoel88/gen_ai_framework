"""FastAPI app factory and base setup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from ..config import get_settings


def create_app(
    title: str = "Gen AI Framework",
    description: str = "Modular Gen AI API with LLM, RAG and domain clients",
    version: str = "0.1.0",
    **kwargs,
) -> FastAPI:
    """Create a FastAPI application with framework defaults."""
    settings = get_settings()

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        **kwargs,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def root():
        """Redirect root to API docs."""
        return RedirectResponse(url="/docs", status_code=302)

    @app.get("/info")
    def info():
        """Service info (for health checks or discovery)."""
        return {"service": title, "version": version, "docs": "/docs"}

    return app
