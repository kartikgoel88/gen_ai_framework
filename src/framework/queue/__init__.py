"""Task queue (Celery) for batch RAG, batch bills, long agent runs."""

from .app import get_celery_app, is_queue_available

__all__ = ["get_celery_app", "is_queue_available"]
