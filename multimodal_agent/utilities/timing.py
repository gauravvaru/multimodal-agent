"""Timing utilities."""

import time


def start_timer() -> float:
    """Return a monotonic timestamp for latency measurement."""
    return time.perf_counter()


def elapsed_ms(start: float) -> float:
    """Return elapsed milliseconds since start_timer."""
    return (time.perf_counter() - start) * 1000.0
