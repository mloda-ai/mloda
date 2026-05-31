"""RED-phase tests for ASOF link disambiguation in the link resolver.

Two ASOF links between the SAME (left_fg, right_fg) pair that share the same
by-key index but differ only in ``asof_config`` (e.g. direction) are both kept
in the ``links`` set (``Link.__eq__``/``__hash__`` include ``asof_config``).

Today ``ResolveLinks._find_matching_links`` narrows exact matches only by
``right_feature.index``; when both candidates share the same right index it
returns BOTH, causing a silent double-join. The target behavior is:

  * a Link pinned on a joining Feature (``feature.link``) disambiguates by ==,
  * if MORE THAN ONE ASOF link still remains after index + pinned narrowing,
    raise a clear ``ValueError`` telling the user to pin the intended link,
  * non-ASOF behavior is unchanged.

PART A exercises ``_find_matching_links`` directly (fast, no multiprocessing).
PART B drives the real path through ``mloda.run_all`` with the link pinned on
the LEFT feature, as in production.
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.components.link import Link
from mloda.core.prepare.graph.graph import Graph
from mloda.core.prepare.resolve_links import ResolveLinks
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda

# Importing the framework registers it as a ComputeFramework subclass, so
# ``compute_frameworks=["PandasDataFrame"]`` resolves even when this module runs
# in isolation.
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (  # noqa: F401
    PandasDataFrame,
)


# ============================================================================
# Mock feature groups (mirror tests/test_core/test_setup/test_asof_link.py)
# ============================================================================
class AsofFGa(FeatureGroup):
    """Mock left feature group with a single by-key index column 'k'."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("k",))]


class AsofFGb(FeatureGroup):
    """Mock right feature group with a single by-key index column 'k'."""

    def input_features(self, _options: Options, _feature_name: FeatureName) -> Optional[set[Any]]:
        return None

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        return [Index(("k",))]


def _link_backward() -> Link:
    return Link.asof(
        JoinSpec(AsofFGa, "k"),
        JoinSpec(AsofFGb, "k"),
        left_time_column="t",
        right_time_column="t",
        direction="backward",
    )


def _link_forward() -> Link:
    return Link.asof(
        JoinSpec(AsofFGa, "k"),
        JoinSpec(AsofFGb, "k"),
        left_time_column="t",
        right_time_column="t",
        direction="forward",
    )


# ============================================================================
# PART A - unit tests on ResolveLinks._find_matching_links
# ============================================================================
class TestAsofLinkDisambiguationUnit:
    def test_unpinned_ambiguous_raises(self) -> None:
        """Two ASOF links (same pair, same right index, differing direction) with no
        pin must FAIL LOUD.

        RED today: ``_find_matching_links`` returns BOTH links and does NOT raise.
        """
        l_back = _link_backward()
        l_fwd = _link_forward()
        resolver = ResolveLinks(Graph(), {l_back, l_fwd})

        right_feature = Feature("rv", index=Index(("k",)))

        with pytest.raises(ValueError):
            resolver._find_matching_links(AsofFGa, AsofFGb, right_feature)

    def test_pin_on_right_feature_selects_one(self) -> None:
        """A Link pinned on the right feature (equal to, but a distinct object from,
        the backward link) must narrow the result to exactly that link.

        RED today: index narrowing keeps both links; the pin is ignored.
        """
        l_back = _link_backward()
        l_fwd = _link_forward()
        resolver = ResolveLinks(Graph(), {l_back, l_fwd})

        right_feature = Feature("rv", link=_link_backward(), index=Index(("k",)))

        matched = resolver._find_matching_links(AsofFGa, AsofFGb, right_feature)

        assert len(matched) == 1
        assert matched[0] == l_back
        assert matched[0] != l_fwd

    def test_single_asof_link_still_returns_it(self) -> None:
        """The common case (exactly one ASOF link) must keep working: no raise.

        Regression guard. Likely GREEN today; guards against an over-broad raise
        being introduced in the Green phase.
        """
        l_back = _link_backward()
        resolver = ResolveLinks(Graph(), {l_back})

        right_feature = Feature("rv", index=Index(("k",)))

        matched = resolver._find_matching_links(AsofFGa, AsofFGb, right_feature)

        assert matched == [l_back]

    def test_sibling_pin_selects_one(self) -> None:
        """A pin carried on a sibling parent feature (``pinned_links``) governs the
        whole feature-group-pair join, including an index pair that carries no pin
        of its own.

        This is the production case: the pin lives on the joining value feature, but
        the (by-key) index pair is also evaluated and must honor the same pin.
        """
        l_back = _link_backward()
        l_fwd = _link_forward()
        resolver = ResolveLinks(Graph(), {l_back, l_fwd})

        # The index feature itself carries no pin; the pin arrives via pinned_links.
        right_feature = Feature("rv", index=Index(("k",)))

        matched = resolver._find_matching_links(AsofFGa, AsofFGb, right_feature, pinned_links=[_link_backward()])

        assert len(matched) == 1
        assert matched[0] == l_back
        assert matched[0] != l_fwd

    def test_conflicting_sibling_pins_raise(self) -> None:
        """Two DIFFERENT ASOF links pinned by different sibling parents cannot be
        auto-selected: the fail-loud raise must still fire (no silent double-join).
        """
        l_back = _link_backward()
        l_fwd = _link_forward()
        resolver = ResolveLinks(Graph(), {l_back, l_fwd})

        right_feature = Feature("rv", index=Index(("k",)))

        with pytest.raises(ValueError):
            resolver._find_matching_links(
                AsofFGa, AsofFGb, right_feature, pinned_links=[_link_backward(), _link_forward()]
            )

    def test_non_asof_multiple_exact_unaffected(self) -> None:
        """Two NON-asof links for the same pair (kept distinct via differing right
        index) must both be returned without raising.

        Documents that the new raise is ASOF-scoped. GREEN today and must stay
        green after the Green phase. ``right_feature=None`` skips index narrowing,
        so both inner links survive to the return.
        """
        l_inner_k = Link.inner(JoinSpec(AsofFGa, "k"), JoinSpec(AsofFGb, "k"))
        l_inner_k2 = Link.inner(JoinSpec(AsofFGa, "k"), JoinSpec(AsofFGb, "k2"))
        resolver = ResolveLinks(Graph(), {l_inner_k, l_inner_k2})

        matched = resolver._find_matching_links(AsofFGa, AsofFGb, None)

        assert len(matched) == 2
        assert set(matched) == {l_inner_k, l_inner_k2}


# ============================================================================
# PART B - integration through mloda.run_all (link pinned on the LEFT feature)
# ============================================================================
# Small data where forward vs backward give DIFFERENT (and non-null) matches so
# the two pin tests cannot both pass by accident.
#   left:  (k,t) = (1,10), (1,20), (2,15)
#   right: (k,t,rv) = (1,5,1), (1,18,2), (1,25,5), (2,5,3), (2,30,4)
# backward: (1,10)->rv1, (1,20)->rv2, (2,15)->rv3   => ["1|10|1","1|20|2","2|15|3"]
# forward:  (1,10)->rv2, (1,20)->rv5, (2,15)->rv4   => ["1|10|2","1|20|5","2|15|4"]
_LEFT_ROWS = {"k": [1, 1, 2], "t": [10, 20, 15], "lv": [100, 200, 300]}
_RIGHT_ROWS = {"k": [1, 1, 1, 2, 2], "t": [5, 18, 25, 5, 30], "rv": [1, 2, 5, 3, 4]}

_EXPECTED_BACKWARD = sorted(["1|10|1", "1|20|2", "2|15|3"])
_EXPECTED_FORWARD = sorted(["1|10|2", "1|20|5", "2|15|4"])


def _disambig_link_backward() -> Link:
    return Link.asof(
        JoinSpec(DisambigLeftFeature, Index(("k",))),
        JoinSpec(DisambigRightFeature, Index(("k",))),
        left_time_column="t",
        right_time_column="t",
        direction="backward",
    )


def _disambig_link_forward() -> Link:
    return Link.asof(
        JoinSpec(DisambigLeftFeature, Index(("k",))),
        JoinSpec(DisambigRightFeature, Index(("k",))),
        left_time_column="t",
        right_time_column="t",
        direction="forward",
    )


class DisambigLeftFeature(FeatureGroup):
    """Left side: emits by-key ``k``, time ``t`` and value ``lv``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"k", "t", "lv"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return dict(_LEFT_ROWS)


class DisambigRightFeature(FeatureGroup):
    """Right side: emits by-key ``k``, time ``t`` and value ``rv``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"k", "t", "rv"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return dict(_RIGHT_ROWS)


class DisambigBackwardJoinedFeature(FeatureGroup):
    """Parent FG pinning the BACKWARD link on the joining ``lv`` feature."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name="lv", link=_disambig_link_backward(), index=Index(("k",))),
            Feature(name="rv", index=Index(("k",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        records = data.to_dict("records")
        return {cls.get_class_name(): [f"{r['k']}|{r['t']}|{r['rv']}" for r in records]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


class DisambigForwardJoinedFeature(FeatureGroup):
    """Parent FG pinning the FORWARD link on the joining ``lv`` feature."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return {
            Feature(name="lv", link=_disambig_link_forward(), index=Index(("k",))),
            Feature(name="rv", index=Index(("k",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        records = data.to_dict("records")
        return {cls.get_class_name(): [f"{r['k']}|{r['t']}|{r['rv']}" for r in records]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_ENABLED = PluginCollector.enabled_feature_groups(
    {
        DisambigLeftFeature,
        DisambigRightFeature,
        DisambigBackwardJoinedFeature,
        DisambigForwardJoinedFeature,
    }
)

# Both ASOF links are registered; only the pin on the joined FG's left feature
# should decide which one fires.
_BOTH_LINKS = {_disambig_link_backward(), _disambig_link_forward()}


def _run_joined(joined_fg: type[FeatureGroup], flight_server: Any) -> list[str]:
    result = mloda.run_all(
        [Feature(name=joined_fg.get_class_name())],
        links=_BOTH_LINKS,
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=_ENABLED,
        flight_server=flight_server,
        parallelization_modes={ParallelizationMode.SYNC},
    )
    assert len(result) == 1
    records = result[0].to_dict("records")
    return sorted(str(r[joined_fg.get_class_name()]) for r in records)


class TestAsofLinkDisambiguationRunAll:
    def test_pinned_backward_selects_backward(self, flight_server: Any) -> None:
        """Pinning the backward link on the left feature must yield backward rows.

        RED today: both ASOF links are registered (the left-feature pin is ignored
        by the resolver call site), so the join double-registers / mis-joins rather
        than honoring the pin.
        """
        assert _run_joined(DisambigBackwardJoinedFeature, flight_server) == _EXPECTED_BACKWARD

    def test_pinned_forward_selects_forward(self, flight_server: Any) -> None:
        """Pinning the forward link on the left feature must yield forward rows
        (distinct from backward), proving the pin actually controls selection.

        RED today: the forward pin is ignored; the default-backward double-join can
        never produce the forward rows.
        """
        assert _run_joined(DisambigForwardJoinedFeature, flight_server) == _EXPECTED_FORWARD
