"""Celery app factory. Requires celery[redis] and Redis when CELERY_BROKER_URL is set."""

from typing import Optional

_celery_app: Optional[object] = None


def get_celery_app(broker_url: Optional[str] = None, result_backend: Optional[str] = None):
    """Create or return Celery app. Returns None if broker_url is empty or celery not installed."""
    global _celery_app
    if not broker_url:
        return None
    if _celery_app is not None:
        return _celery_app
    try:
        from celery import Celery
        _celery_app = Celery(
            "gen_ai_framework",
            broker=broker_url,
            backend=result_backend or broker_url,
            include=["src.framework.queue.tasks"],
        )
        _celery_app.conf.task_serializer = "json"
        _celery_app.conf.result_serializer = "json"
        _celery_app.conf.accept_content = ["json"]
        _celery_app.conf.task_track_started = True
        return _celery_app
    except ImportError:
        return None


def is_queue_available(broker_url: Optional[str] = None) -> bool:
    """Return True if Celery is configured and available."""
    app = get_celery_app(broker_url=broker_url) if broker_url else None
    return app is not None
