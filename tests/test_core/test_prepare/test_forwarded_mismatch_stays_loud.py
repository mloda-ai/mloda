"""R1: the forwarded-name-mismatch guard must stay loud at the ENGINE seam (#845 follow-up).

The damaging case is a SECOND group claiming the same name: contained, the mismatch candidate is dropped,
the rival wins and the forwarded value is silently ignored. Every existing test for this guard calls the
matcher directly, so these pins go through the resolution seam. Doubles are dropped per test.
"""

import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypeVar

from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.property_spec import property_spec
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping

from tests.test_core.test_prepare.identify_seam import evaluate_or_raise


PROBE_TYPE_KEY = "forward_probe_type_845r"
LOUD_FEATURE = "value__median_loudprobe845r"
MISMATCH_CLASS_NAME = "ForwardMismatchLoudFG845r"
RIVAL_CLASS_NAME = "RivalNameOwnerFG845r"
RAISE_TYPE_NAME = "ValueError"

T = TypeVar("T")


class LoudProbeFw845r(ComputeFramework):
    """Dummy compute framework for the forwarded-mismatch seam tests."""


def _capture(call: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run call, returning (value, None) or (None, 'Type: message'). No traceback is retained."""
    try:
        return call(), None
    except Exception as exc:  # noqa: BLE001  (an escape, or its absence, is the fact under test)
        return None, f"{type(exc).__name__}: {exc}"


def _forwarded_mismatch_options() -> Options:
    """Options whose forwarded probe type ('sum') contradicts the name-parsed one ('median')."""
    options = Options(group={PROBE_TYPE_KEY: "sum"})
    options.inherited_group_keys = frozenset({PROBE_TYPE_KEY})
    return options


def _make_mismatch_fg() -> type[FeatureGroup]:
    """Chain-parsed candidate whose matcher raises the framework forwarded/name mismatch ValueError."""
    # Class objects are cyclic; collect leftovers from earlier tests before defining a twin.
    gc.collect()

    class ForwardMismatchLoudFG845r(FeatureChainParserMixin, FeatureGroup):
        """Binds the probe type from the feature name, so a contradicting forwarded value is a mismatch."""

        PREFIX_PATTERN = r".*__([\w]+)_loudprobe845r$"

        PROPERTY_MAPPING = {
            PROBE_TYPE_KEY: property_spec(
                "Probe operation subtype.",
                strict=True,
                allowed_values={"median": "Median value", "sum": "Sum of values"},
                context=True,
            ),
        }

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {LoudProbeFw845r}

    return ForwardMismatchLoudFG845r


def _make_rival_fg() -> type[FeatureGroup]:
    """Rival candidate claiming the same feature name cleanly, with no interest in the forwarded option."""
    gc.collect()

    class RivalNameOwnerFG845r(FeatureGroup):
        """Matches the probe feature name unconditionally: the group that would silently win today."""

        @classmethod
        def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
            return {LoudProbeFw845r}

        @classmethod
        def match_feature_group_criteria(
            cls,
            feature_name: FeatureName | str,
            options: Options,
            data_access_collection: DataAccessCollection | None = None,
        ) -> bool:
            return str(feature_name) == LOUD_FEATURE

        def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
            return None

    return RivalNameOwnerFG845r


@dataclass(frozen=True)
class _SeamSnapshot:
    """Plain-data readout of one seam evaluation. Holds no class and no exception object."""

    escaped: str | None
    identified_names: tuple[str, ...]


def _evaluate_at_seam(options: Options, with_rival: bool) -> _SeamSnapshot:
    """Resolve the probe feature through the engine seam and read the outcome out as plain data."""
    mismatch_fg = _make_mismatch_fg()
    rival_fg = _make_rival_fg() if with_rival else None
    try:
        feature = Feature(LOUD_FEATURE, options=options)
        plugins: FeatureGroupEnvironmentMapping = {mismatch_fg: {LoudProbeFw845r}}
        if rival_fg is not None:
            plugins[rival_fg] = {LoudProbeFw845r}
        result, escaped = _capture(partial(evaluate_or_raise, feature, plugins, None))
        identified = () if result is None else tuple(sorted(fg.get_class_name() for fg in result.identified))
        snapshot = _SeamSnapshot(escaped=escaped, identified_names=identified)
        del result
        del plugins
        del feature
        return snapshot
    finally:
        del mismatch_fg
        del rival_fg
        gc.collect()


class TestForwardedMismatchStaysLoudAtTheSeam:
    """The mismatch guard is framework-owned: its raise must cross the match seam, not be contained."""

    def test_mismatch_raise_reaches_the_caller(self) -> None:
        """A contradicting forwarded value aborts the resolution with the guard's own ValueError."""
        snapshot = _evaluate_at_seam(_forwarded_mismatch_options(), with_rival=False)

        assert snapshot.escaped is not None, "the forwarded-name-mismatch guard must not be contained"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: "), (
            f"the guard's own ValueError must reach the caller, got: {snapshot.escaped}"
        )
        assert PROBE_TYPE_KEY in snapshot.escaped
        assert "forwarded" in snapshot.escaped.lower()
        assert snapshot.identified_names == ()

    def test_mismatch_is_not_dropped_when_a_rival_claims_the_name(self) -> None:
        """A rival matching the same name must not swallow the mismatch and win with the value ignored."""
        snapshot = _evaluate_at_seam(_forwarded_mismatch_options(), with_rival=True)

        assert snapshot.identified_names != (RIVAL_CLASS_NAME,), (
            "the rival must not silently win while the forwarded value is ignored"
        )
        assert snapshot.escaped is not None, "a rival candidate must not hide the forwarded-name-mismatch abort"
        assert snapshot.escaped.startswith(f"{RAISE_TYPE_NAME}: "), (
            f"the guard's own ValueError must reach the caller, got: {snapshot.escaped}"
        )
        assert PROBE_TYPE_KEY in snapshot.escaped
        assert "forwarded" in snapshot.escaped.lower()

    def test_probe_group_resolves_without_a_conflicting_forwarded_option(self) -> None:
        """Control: only the mismatch case aborts; the chain-parsed group is otherwise an ordinary winner."""
        snapshot = _evaluate_at_seam(Options(), with_rival=False)

        assert snapshot.escaped is None
        assert snapshot.identified_names == (MISMATCH_CLASS_NAME,)
