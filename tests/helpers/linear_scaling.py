"""Shared assertion that a string function runs in linear time."""

import time
from collections.abc import Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")


def assert_runs_linearly(fn: Callable[[str], ResultT], build: Callable[[int], str], n: int, what: str) -> ResultT:
    """Fail if min CPU time at n hits 8x that at n // 4 (2 ms floor) or 2 s; linear ~4x, quadratic ~16x."""
    name = getattr(fn, "__name__", "function")
    small, large = build(n // 4), build(n)
    min_small = min_large = float("inf")
    ratio = float("inf")
    for _ in range(3):
        start = time.process_time()
        fn(small)
        min_small = min(min_small, time.process_time() - start)
        start = time.process_time()
        result = fn(large)
        min_large = min(min_large, time.process_time() - start)
        ratio = min_large / max(min_small, 2e-3)
        if ratio < 8.0:
            break
    assert ratio < 8.0, (
        f"{name} scales super-linearly on {what}: "
        f"min small {min_small:.4f}s, min large {min_large:.4f}s, ratio {ratio:.1f}"
    )
    assert min_large < 2.0, f"{name} is too slow on {what}: min large {min_large:.4f}s"
    return result
