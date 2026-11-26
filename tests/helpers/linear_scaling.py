"""Shared assertion that a string function runs in linear time."""

import sys
import time
from collections.abc import Callable
from typing import TypeVar

ResultT = TypeVar("ResultT")

GAP = 8


def assert_runs_linearly(fn: Callable[[str], ResultT], build: Callable[[int], str], n: int, what: str) -> ResultT:
    """Fail if min run time at n hits GAP**1.5x that at n // GAP (2 ms floor) or its CPU time hits 2 s."""
    name = getattr(fn, "__name__", "function")
    threshold = GAP**1.5
    small, large = build(n // GAP), build(n)
    min_small = min_large = min_cpu_large = float("inf")
    ratio = float("inf")
    # process_time ticks at 15.6 ms on Windows, too coarse for the ratio.
    clock = time.perf_counter if sys.platform == "win32" else time.process_time
    for _ in range(3):
        start = clock()
        fn(small)
        min_small = min(min_small, clock() - start)
        start, cpu_start = clock(), time.process_time()
        result = fn(large)
        min_large = min(min_large, clock() - start)
        min_cpu_large = min(min_cpu_large, time.process_time() - cpu_start)
        ratio = min_large / max(min_small, 2e-3)
        if ratio < threshold:
            break
    assert ratio < threshold, (
        f"{name} scales super-linearly on {what}: "
        f"min small {min_small:.4f}s, min large {min_large:.4f}s, ratio {ratio:.1f}"
    )
    assert min_cpu_large < 2.0, f"{name} is too slow on {what}: min large {min_cpu_large:.4f}s"
    return result
