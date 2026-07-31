"""R2: ``is_match_abort`` must not fail open on a hostile exception (#845 follow-up).

A ``__getattr__`` that raises would blow the marker read up inside the seam's own ``except`` block, and a
permissively truthy one would fake framework provenance, so the marker is read from ``exc.__dict__``
(every ``BaseException`` has one, including a ``__slots__ = ()`` subclass). Both directions are pinned at
the helper level AND through the seam, where a raise out of the ``except`` block is uncontained.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, TypeVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import escalate_match_abort, is_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass


HOSTILE_MESSAGE = "boom_845r_hostile_attribute_access"
PERMISSIVE_MESSAGE = "boom_845r_permissive_attribute_access"
SLOTTED_MESSAGE = "boom_845r_slotted_framework_raise"
HOSTILE_FEATURE = "hostile_marker_feat_845r"
PERMISSIVE_FEATURE = "permissive_marker_feat_845r"
HOSTILE_CLASS_NAME = "HostileMarkerFG845r"
PERMISSIVE_CLASS_NAME = "PermissiveMarkerFG845r"
MATCHER_ERROR_STAGE = "matcher_error"

T = TypeVar("T")


class MarkerProbeFw845r(ComputeFramework):
    """Dummy compute framework for the marker-robustness tests."""


def _is_dunder(name: str) -> bool:
    """Dunder lookups stay untouched so the interpreter's own machinery keeps working on these doubles."""
    return name.startswith("__") and name.endswith("__")


class HostileGetattrError845r(Exception):
    """Exception whose attribute access raises, modelling a proxying or broken exception class."""

    def __getattr__(self, name: str) -> Any:
        if _is_dunder(name):
            raise AttributeError(name)
        raise RuntimeError(f"hostile attribute access for '{name}'")


class PermissiveGetattrError845r(Exception):
    """Exception whose attribute access is truthy for everything, modelling a mock-like exception class."""

    def __getattr__(self, name: str) -> Any:
        if _is_dunder(name):
            raise AttributeError(name)
        return True


class SlottedFrameworkError845r(Exception):
    """Framework-owned raise declaring no instance layout of its own; BaseException still supplies __dict__."""

    __slots__ = ()


def _capture(call: Callable[[], T]) -> tuple[Optional[T], Optional[str]]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _make_raising_fg(class_name: str, feature_name: str, exc_factory: Callable[[], Exception]) -> type[FeatureGroup]:
    """Build a candidate whose matcher raises the given exception for its own feature name."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class MarkerProbeFG845r(FeatureGroup):
        """Raises an exception whose attribute protocol is what the marker read must survive."""

        @classmethod
        def feature_names_supported(cls) -> set[str]:
            return {feature_name}

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {MarkerProbeFw845r}

        @classmethod
        def match_feature_group_criteria(
            cls,
            name: FeatureName | str,
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if str(name) != feature_name:
                return False
            raise exc_factory()

        def input_features(self, options: Options, feature_name_arg: FeatureName) -> Optional[set[Feature]]:
            return None

    MarkerProbeFG845r.__name__ = class_name
    MarkerProbeFG845r.__qualname__ = class_name
    return MarkerProbeFG845r


@dataclass(frozen=True)
class _ContainmentSnapshot:
    """Plain-data readout of one seam evaluation. Holds no class and no exception object."""

    escaped: Optional[str]
    failure_kind: Optional[str]
    eliminated_names: tuple[str, ...]
    stage: Optional[str]
    reason: Optional[str]


def _evaluate_raising_matcher(
    class_name: str, feature_name: str, exc_factory: Callable[[], Exception]
) -> _ContainmentSnapshot:
    """Evaluate the double's own feature name and read the containment out as plain data."""
    broken_fg = _make_raising_fg(class_name, feature_name, exc_factory)
    try:
        feature = Feature(feature_name)
        plugins: FeatureGroupEnvironmentMapping = {broken_fg: {MarkerProbeFw845r}}
        result, escaped = _capture(partial(IdentifyFeatureGroupClass.evaluate, feature, plugins, None))
        if result is None:
            return _ContainmentSnapshot(escaped, None, (), None, None)

        elimination = result.eliminations.get(broken_fg)
        snapshot = _ContainmentSnapshot(
            escaped=None,
            failure_kind=result.failure_kind,
            eliminated_names=tuple(sorted(fg.get_class_name() for fg in result.eliminations)),
            stage=None if elimination is None else str(elimination.stage),
            reason=None if elimination is None else str(elimination.reason),
        )
        del elimination
        del result
        del plugins
        return snapshot
    finally:
        del broken_fg
        gc.collect()


class TestIsMatchAbortReadsProvenanceOnly:
    """The marker read answers from what escalate_match_abort wrote, never from an attribute hook."""

    def test_hostile_getattr_is_not_a_match_abort(self) -> None:
        """A raising __getattr__ must read as unmarked, not blow up the read itself."""
        assert is_match_abort(HostileGetattrError845r(HOSTILE_MESSAGE)) is False

    def test_permissive_getattr_is_not_a_match_abort(self) -> None:
        """A truthy-for-everything __getattr__ must not be mistaken for framework provenance."""
        assert is_match_abort(PermissiveGetattrError845r(PERMISSIVE_MESSAGE)) is False

    def test_marking_still_works_on_a_slotted_exception(self) -> None:
        """Control: a genuinely marked raise is still framework-owned, even with __slots__ = ()."""
        exc = SlottedFrameworkError845r(SLOTTED_MESSAGE)

        marked = escalate_match_abort(exc)

        assert marked is exc
        assert is_match_abort(marked) is True
        assert is_match_abort(SlottedFrameworkError845r(SLOTTED_MESSAGE)) is False


class TestHostileExceptionStaysContainedAtTheSeam:
    """A raising __getattr__ must not turn one broken candidate into a poisoned resolution."""

    def test_hostile_matcher_raise_does_not_escape_evaluate(self) -> None:
        """The marker read runs inside the seam's except block, so it must never raise there."""
        snapshot = _evaluate_raising_matcher(
            HOSTILE_CLASS_NAME, HOSTILE_FEATURE, lambda: HostileGetattrError845r(HOSTILE_MESSAGE)
        )

        assert snapshot.escaped is None, (
            f"reading the marker must not let a new exception escape the seam, got: {snapshot.escaped}"
        )
        assert snapshot.failure_kind == "none"
        assert snapshot.eliminated_names == (HOSTILE_CLASS_NAME,)
        assert snapshot.stage == MATCHER_ERROR_STAGE


class TestPermissiveExceptionIsNotEscalated:
    """A permissive attribute hook must not buy a plugin raise framework provenance."""

    def test_permissive_matcher_raise_stays_contained(self) -> None:
        """Without a real escalate_match_abort call the raise is contained as a matcher_error near-miss."""
        snapshot = _evaluate_raising_matcher(
            PERMISSIVE_CLASS_NAME, PERMISSIVE_FEATURE, lambda: PermissiveGetattrError845r(PERMISSIVE_MESSAGE)
        )

        assert snapshot.escaped is None, (
            f"an unmarked raise must stay contained whatever its __getattr__ answers, got: {snapshot.escaped}"
        )
        assert snapshot.failure_kind == "none"
        assert snapshot.eliminated_names == (PERMISSIVE_CLASS_NAME,)
        assert snapshot.stage == MATCHER_ERROR_STAGE
        assert snapshot.reason is not None
        assert "PermissiveGetattrError845r" in snapshot.reason
