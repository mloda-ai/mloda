"""Seeded scheduling-jitter helper for SYNC drop-timing regression tests.

A real THREADING/MULTIPROCESSING run can finish a ready step several planner passes
after it first becomes runnable. Plain SYNC always finishes it on the very next
pass, which hides step-ordering bugs that a `JoinStep`'s "wait on every ancestor of
its consumer" rule would otherwise mask (see mloda-ai/mloda#1430).

`run_under_scheduling_jitter` drives a zero-arg `run_fn` under several seeds, each
seed randomly refusing otherwise-ready steps for up to `max_defers` planner passes
through a not-yet-existing seam on `ExecutionOrchestrator`.

Intended production seam (Green phase, not implemented here):

`ExecutionOrchestrator._can_run_step` gains one extra check right before its final
`return True`::

    if self._defer_ready_step(step, made_progress):
        return False

where `made_progress` is `_run_planner_pass`'s own local flag, threaded through as a
new parameter of `_can_run_step` (it already tracks whether some step was granted or
finished earlier in the current pass). `_defer_ready_step` is a plain instance
method defaulting to `return False` (never defer), so untouched runs are unaffected.
Tests override it per class via
`monkeypatch.setattr(ExecutionOrchestrator, "_defer_ready_step", <fn>)`.

Because the hook is only consulted for a step that would otherwise be allowed to
run, and because it is only permitted to defer once `made_progress` is already
True, at least one ready step per pass is always let through without the hook
itself needing to know how many other ready steps exist this pass: this is the
seam-side half of the `_raise_if_stalled` guarantee
(`ExecutionOrchestrator._raise_if_stalled`, run.py:213) the issue calls out.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Sequence, TypeVar

import pytest

from mloda.core.runtime.run import ExecutionOrchestrator

T = TypeVar("T")


def run_under_scheduling_jitter(
    run_fn: Callable[[], T],
    seeds: Sequence[int],
    monkeypatch: pytest.MonkeyPatch,
    max_defers: int = 3,
) -> list[tuple[int, T]]:
    """Run `run_fn` once per seed with seeded, bounded, ready-step deferral injected
    into `ExecutionOrchestrator`; return `[(seed, result), ...]` in seed order.

    Each seed gets its own `random.Random(seed)` and its own per-step defer-count
    map, so seeds are independent and a run is reproducible. A step is refused at
    most `max_defers` times; after that it is always allowed through, so a run
    cannot stall solely because jitter is exhausted. `run_fn` must be a zero-arg
    callable that performs one full mloda run (e.g. a lambda wrapping
    `mloda.run_all(...)`) and returns its result.
    """
    results: list[tuple[int, T]] = []
    for seed in seeds:
        rng = random.Random(seed)  # nosec B311
        defer_counts: dict[int, int] = defaultdict(int)

        def _defer_ready_step(
            self: ExecutionOrchestrator,
            step: Any,
            made_progress: bool,
            _rng: random.Random = rng,
            _defer_counts: dict[int, int] = defer_counts,
        ) -> bool:
            # Never defer the first ready step of a pass: this is what keeps
            # `_raise_if_stalled` from ever firing on a jittered run.
            if not made_progress:
                return False
            key = id(step)
            if _defer_counts[key] >= max_defers:
                return False
            defer = _rng.random() < 0.5
            if defer:
                _defer_counts[key] += 1
            return defer

        monkeypatch.setattr(ExecutionOrchestrator, "_defer_ready_step", _defer_ready_step)
        results.append((seed, run_fn()))
    return results
