"""Pin the gc.freeze contract that keeps the suite's per-test gc.collect() calls off the hot path.

26 test modules run gc.collect() in an autouse fixture, because a throwaway FeatureGroup subclass
defined in a test body lingers in FeatureGroup.__subclasses__() until a GC pass reclaims its reference
cycle. Each collect rescans the whole imported heap (pandas, polars, duckdb, pyarrow, sklearn, every
plugin and test module), so one collect costs ~280ms and gc dominates the suite's wall time. Freezing
the imported graph into the permanent generation once collection has finished makes those collects
free without weakening the isolation they buy.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from tests import conftest as tests_conftest


_THROWAWAY_NAME = "GcFreezeThrowawayFeatureGroup"

_FREEZE_MISSING = (
    "gc.get_freeze_count() is 0: the imported object graph is not frozen. tests/conftest.py must call "
    "gc.freeze() once pytest collection has finished, so the per-test gc.collect() calls stop rescanning "
    "the whole imported heap. Without it a single gc.collect() costs ~280ms and the suite spends most of "
    "its wall time in gc."
)

_UNFREEZE_MISSING = (
    "tests/conftest.py defines no pytest_sessionfinish hook. gc.freeze() must be undone with gc.unfreeze() "
    "before the interpreter finalizes, or the multiprocessing tests abort with 'Fatal Python error: "
    "gilstate_tss_set: failed to set current tstate (TSS)'."
)


@pytest.fixture(autouse=True)
def _no_feature_group_registry_pollution() -> Any:
    """Guarantee this module never leaks its throwaway FeatureGroup subclass (see the filter test's twin fixture)."""
    yield
    gc.collect()
    gc.collect()
    leaked = [c for c in get_all_subclasses(FeatureGroup) if c.__module__ == __name__]
    assert not leaked, f"Leaked FeatureGroup subclasses from {__name__}: {[c.__name__ for c in leaked]}"


def _define_throwaway_feature_group() -> type[FeatureGroup]:
    """Define a FeatureGroup subclass inside a function body, exactly as the suite's throwaway probes do."""

    class GcFreezeThrowawayFeatureGroup(FeatureGroup):
        pass

    return GcFreezeThrowawayFeatureGroup


def _throwaway_is_registered() -> bool:
    """True while a subclass named _THROWAWAY_NAME is still reachable from FeatureGroup."""
    return any(cls.__name__ == _THROWAWAY_NAME for cls in get_all_subclasses(FeatureGroup))


def test_import_graph_is_frozen_while_tests_run() -> None:
    """The imported graph sits in the permanent generation for the whole run, so gc.collect() never rescans it."""
    assert gc.get_freeze_count() > 0, _FREEZE_MISSING


def test_gc_collect_still_reaps_throwaway_feature_groups_under_a_frozen_graph() -> None:
    """Freezing must not extend the permanent generation to classes created later: one gc.collect() still reaps."""
    assert gc.get_freeze_count() > 0, _FREEZE_MISSING

    # Only booleans stay in this frame across the collect: a retained class reference, or an assert-rewrite
    # temporary holding one, would pin the very cycle whose reclamation is under test.
    control = _define_throwaway_feature_group()
    registered_while_referenced = control in get_all_subclasses(FeatureGroup)
    del control
    gc.collect()
    still_registered = _throwaway_is_registered()

    assert registered_while_referenced, (
        "Positive control failed: a FeatureGroup subclass defined in a function body is not reachable from "
        "get_all_subclasses(FeatureGroup) even while strongly referenced, so this test can no longer detect "
        "a leak. Fix the probe, not the freeze hook."
    )
    assert not still_registered, (
        f"{_THROWAWAY_NAME} survived gc.collect() and stays in get_all_subclasses(FeatureGroup). Every autouse "
        "isolation fixture in the suite depends on gc.collect() reaping throwaway subclasses; freezing the "
        "import graph must leave classes created after the freeze collectable."
    )


def test_conftest_unfreezes_the_graph_at_session_finish() -> None:
    """gc.freeze() is only safe when paired with gc.unfreeze() before interpreter finalization."""
    assert callable(getattr(tests_conftest, "pytest_sessionfinish", None)), _UNFREEZE_MISSING
