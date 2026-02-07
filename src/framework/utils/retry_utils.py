"""Shared retry and backoff helpers for error recovery."""


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception appears to be a rate limit (e.g. HTTP 429)."""
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg


def compute_backoff_delay(
    attempt: int,
    initial_delay: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 60.0,
    rate_limit_min: float | None = None,
) -> float:
    """
    Compute delay in seconds for exponential backoff.

    Args:
        attempt: Current attempt index (0-based).
        initial_delay: Base delay for attempt 0.
        factor: Multiplier per attempt (delay = initial_delay * factor ** attempt).
        max_delay: Cap on delay.
        rate_limit_min: If set, returned delay is at least this (e.g. 60 for rate limits).

    Returns:
        Delay in seconds.
    """
    delay = min(initial_delay * (factor ** attempt), max_delay)
    if rate_limit_min is not None and delay < rate_limit_min:
        return rate_limit_min
    return delay
