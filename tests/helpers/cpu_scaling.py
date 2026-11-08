"""CPU-time scaling checks for operations on growing text inputs."""

import time
from collections.abc import Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")


def assert_scales_linearly(
    operation: Callable[[str], ResultT], build: Callable[[int], str], n: int, what: str
) -> ResultT:
    """Require <8x CPU time for 4x input (small floored at 2 ms) and <2 s; linear is ~4x, quadratic ~16x."""
    small, large = build(n // 4), build(n)
    min_small = min_large = float("inf")
    ratio = float("inf")
    for _ in range(3):
        start = time.process_time()
        operation(small)
        min_small = min(min_small, time.process_time() - start)
        start = time.process_time()
        result = operation(large)
        min_large = min(min_large, time.process_time() - start)
        ratio = min_large / max(min_small, 2e-3)
        if ratio < 8.0:
            break
    assert ratio < 8.0, (
        f"{what} scales super-linearly: min small {min_small:.4f}s, min large {min_large:.4f}s, ratio {ratio:.1f}"
    )
    assert min_large < 2.0, f"{what} is too slow: min large {min_large:.4f}s"
    return result
