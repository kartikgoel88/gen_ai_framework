"""Documents client: examples and utilities using the document framework (OCR, extraction)."""

__all__ = ["PASSPORT_ENTITIES", "extract_passport_entities", "main", "parse_passport_images"]


def __getattr__(name: str):
    """Lazy import so running python -m src.clients.documents.parse_passport doesn't conflict."""
    if name in __all__:
        from .parse_passport import (
            PASSPORT_ENTITIES,
            extract_passport_entities,
            main,
            parse_passport_images,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
