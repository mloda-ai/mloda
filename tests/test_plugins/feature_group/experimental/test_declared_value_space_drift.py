"""Pin declared value spaces to the dispatch that serves them (issue #774).

A finite ``allowed_values`` space and the branch table that consumes it live in different
places and drift independently. The sklearn ``feature_engineering`` defect (#797) is the
worked example: the value was declared and validated, but no branch handled it, so it fell
through to the scaling pipeline and looked like it worked.

These checks are the reusable form of that assertion, plus the cross-plugin sweeps for the
shared ``algorithm`` and ``time_unit`` discriminator spaces.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import pytest

from mloda.provider import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.clustering.base import ClusteringFeatureGroup
from mloda_plugins.feature_group.experimental.dimensionality_reduction.base import (
    DimensionalityReductionFeatureGroup,
)
from mloda_plugins.feature_group.experimental.forecasting.base import ForecastingFeatureGroup
from mloda_plugins.feature_group.experimental.geo_distance.base import GeoDistanceFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.pipeline.base import SklearnPipelineFeatureGroup
from mloda_plugins.feature_group.experimental.time_reference_mixin import TimeReferenceMixin
from mloda_plugins.feature_group.experimental.time_window.base import TimeWindowFeatureGroup

# ---------------------------------------------------------------------------
# Reusable assertion
# ---------------------------------------------------------------------------


def assert_values_reach_distinct_branches(
    *,
    feature_group: str,
    key: str,
    declared: Iterable[Any],
    resolve: Callable[[Any], Any],
    aliases: Sequence[frozenset[Any]] = (),
) -> None:
    """Every declared value must reach a branch, and branches must not silently collide.

    ``resolve`` maps one declared value to a hashable signature of the branch that served
    it. Two values sharing a signature is drift unless they are listed in ``aliases`` as a
    deliberate synonym group.

    Failures name the feature group, the key and the offending value, so a drift is
    actionable without reading this file.
    """
    unreachable: list[str] = []
    signatures: dict[Any, Any] = {}

    for value in declared:
        try:
            signatures[value] = resolve(value)
        except Exception as exc:  # noqa: BLE001 - the point is to report any failure to dispatch
            unreachable.append(f"    {value!r} -> {type(exc).__name__}: {exc}")

    assert not unreachable, (
        f"{feature_group}.{key}: declared value(s) with no working dispatch branch:\n"
        + "\n".join(unreachable)
        + "\n  Either implement the branch or remove the value from the declared space."
    )

    by_signature: dict[Any, list[Any]] = {}
    for value, signature in signatures.items():
        by_signature.setdefault(signature, []).append(value)

    collisions = [
        sorted(values, key=repr)
        for signature, values in by_signature.items()
        if len(values) > 1 and frozenset(values) not in aliases
    ]

    assert not collisions, (
        f"{feature_group}.{key}: declared value(s) share a dispatch branch without being "
        f"declared synonyms: {collisions}.\n"
        "  A value that silently resolves to another value's behaviour is the #797 defect. "
        "Give it its own branch, or add it to `aliases` if the synonym is intended."
    )


# ---------------------------------------------------------------------------
# Per-plugin drift checks
# ---------------------------------------------------------------------------


class TestSklearnPipelineTypesDoNotDrift:
    """Mandatory coverage for the declaration repaired in #797."""

    def test_every_pipeline_type_reaches_its_own_branch(self) -> None:
        pytest.importorskip("sklearn")

        def resolve(pipeline_name: str) -> tuple[str, ...]:
            config = SklearnPipelineFeatureGroup._create_default_pipeline_config(pipeline_name)
            return tuple(name for name, _ in config["steps"])

        assert_values_reach_distinct_branches(
            feature_group="SklearnPipelineFeatureGroup",
            key=SklearnPipelineFeatureGroup.PIPELINE_NAME,
            declared=SklearnPipelineFeatureGroup.PIPELINE_TYPES,
            resolve=resolve,
        )

    def test_an_undeclared_type_is_rejected(self) -> None:
        """The counterpart: dispatch must not invent behaviour for an undeclared value."""
        pytest.importorskip("sklearn")

        with pytest.raises(ValueError, match="Unsupported pipeline type"):
            SklearnPipelineFeatureGroup._create_default_pipeline_config("not_declared")


class TestClusteringAlgorithmsDoNotDrift:
    """Dispatch is asserted by which branch runs, not by clustering quality."""

    BRANCHES = {
        "kmeans": "_perform_kmeans_clustering",
        "hierarchical": "_perform_hierarchical_clustering",
        "dbscan": "_perform_dbscan_clustering",
        "spectral": "_perform_spectral_clustering",
        "affinity": "_perform_affinity_clustering",
        "agglomerative": "_perform_hierarchical_clustering",
    }

    def test_every_declared_algorithm_has_a_known_branch(self) -> None:
        """The branch map above must stay in step with the declared space."""
        declared = set(ClusteringFeatureGroup.CLUSTERING_ALGORITHMS)
        mapped = set(self.BRANCHES)

        assert declared == mapped, (
            f"ClusteringFeatureGroup.{ClusteringFeatureGroup.ALGORITHM}: declared space and the "
            f"dispatch map disagree. Only declared: {sorted(declared - mapped)}; "
            f"only mapped: {sorted(mapped - declared)}."
        )

    def test_dispatch_routes_each_algorithm_to_its_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pandas = pytest.importorskip("pandas")
        pytest.importorskip("sklearn")

        from mloda_plugins.feature_group.experimental.clustering.pandas import (
            PandasClusteringFeatureGroup,
        )

        methods = sorted(set(self.BRANCHES.values()))
        for method in methods:
            monkeypatch.setattr(
                PandasClusteringFeatureGroup,
                method,
                classmethod(lambda cls, *a, _m=method, **kw: _m),
                raising=True,
            )

        data = pandas.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 3.0, 2.0]})

        def resolve(algorithm: str) -> Any:
            return PandasClusteringFeatureGroup._perform_clustering(data, algorithm, 2, ["a", "b"])

        # hierarchical and agglomerative deliberately share one implementation.
        assert_values_reach_distinct_branches(
            feature_group="PandasClusteringFeatureGroup",
            key=ClusteringFeatureGroup.ALGORITHM,
            declared=ClusteringFeatureGroup.CLUSTERING_ALGORITHMS,
            resolve=resolve,
            aliases=(frozenset({"hierarchical", "agglomerative"}),),
        )

    def test_an_undeclared_algorithm_is_rejected(self) -> None:
        pandas = pytest.importorskip("pandas")
        pytest.importorskip("sklearn")

        from mloda_plugins.feature_group.experimental.clustering.pandas import (
            PandasClusteringFeatureGroup,
        )

        data = pandas.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
        with pytest.raises(ValueError, match="Unsupported clustering algorithm"):
            PandasClusteringFeatureGroup._perform_clustering(data, "not_declared", 2, ["a", "b"])


def _scalar(result: Any) -> float:
    """Aggregations return a numpy scalar here, but a Series would be just as valid."""
    if hasattr(result, "iloc"):
        return float(result.iloc[0])
    return float(result)


class TestAggregationTypesDoNotDrift:
    def test_every_aggregation_type_reaches_a_branch(self) -> None:
        pandas = pytest.importorskip("pandas")

        from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import (
            PandasAggregatedFeatureGroup,
        )

        data = pandas.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})

        def resolve(aggregation_type: str) -> Any:
            result = PandasAggregatedFeatureGroup._perform_aggregation(data, aggregation_type, ["value"])
            return (aggregation_type, _scalar(result))

        # Signatures include the type itself, so this asserts reachability rather than
        # distinctness; avg/mean are declared synonyms and would otherwise collide.
        assert_values_reach_distinct_branches(
            feature_group="PandasAggregatedFeatureGroup",
            key="aggregation_type",
            declared=AggregatedFeatureGroup.AGGREGATION_TYPES,
            resolve=resolve,
        )

    def test_declared_synonyms_agree(self) -> None:
        """avg and mean are declared with the same description, so they must compute the same."""
        pandas = pytest.importorskip("pandas")

        from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import (
            PandasAggregatedFeatureGroup,
        )

        data = pandas.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
        avg = PandasAggregatedFeatureGroup._perform_aggregation(data, "avg", ["value"])
        mean = PandasAggregatedFeatureGroup._perform_aggregation(data, "mean", ["value"])

        assert _scalar(avg) == _scalar(mean)


# ---------------------------------------------------------------------------
# Cross-plugin discriminator sweeps
# ---------------------------------------------------------------------------


class TestDiscriminatorSpacesDoNotCollide:
    # Annotated as Any: the shared ALGORITHM key is declared per plugin, not on FeatureGroup.
    ALGORITHM_OWNERS: list[tuple[str, Any]] = [
        ("ClusteringFeatureGroup", ClusteringFeatureGroup),
        ("DimensionalityReductionFeatureGroup", DimensionalityReductionFeatureGroup),
        ("ForecastingFeatureGroup", ForecastingFeatureGroup),
    ]

    def test_algorithm_spaces_are_disjoint_across_feature_groups(self) -> None:
        """`algorithm` is a shared key name, so an overlapping value would be ambiguous.

        On the configuration path the value is what tells these groups apart, so two groups
        claiming the same algorithm token would make resolution depend on ordering.
        """
        spaces = {
            name: frozenset(group.PROPERTY_MAPPING[group.ALGORITHM].allowed_values or ())
            for name, group in self.ALGORITHM_OWNERS
        }

        overlaps = []
        names = sorted(spaces)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = spaces[left] & spaces[right]
                if shared:
                    overlaps.append(f"{left} and {right} both declare algorithm {sorted(shared)}")

        assert not overlaps, "Ambiguous `algorithm` discriminator values:\n  " + "\n  ".join(overlaps)

    def test_every_algorithm_owner_declares_a_non_empty_space(self) -> None:
        for name, group in self.ALGORITHM_OWNERS:
            space = group.PROPERTY_MAPPING[group.ALGORITHM].allowed_values
            assert space, f"{name}.algorithm declares an empty value space, so nothing constrains it"

    @pytest.mark.parametrize(
        "name,group",
        [
            ("ForecastingFeatureGroup", ForecastingFeatureGroup),
            ("TimeWindowFeatureGroup", TimeWindowFeatureGroup),
        ],
    )
    def test_time_unit_spaces_come_from_the_shared_mixin(self, name: str, group: Any) -> None:
        """`time_unit` must stay one shared space rather than divergent per-plugin copies.

        Identity, not equality: equal copies can drift apart later without any test noticing.
        """
        declared = group.PROPERTY_MAPPING[group.TIME_UNIT].allowed_values

        assert declared is TimeReferenceMixin.TIME_UNITS, (
            f"{name}.{group.TIME_UNIT} does not use TimeReferenceMixin.TIME_UNITS. "
            "A private copy drifts silently; reference the shared mapping instead."
        )


# ---------------------------------------------------------------------------
# geo_distance strict in_features validator
# ---------------------------------------------------------------------------


class TestGeoDistanceInFeaturesValidator:
    """The strict validator judges one unpacked point-feature name; #750 had no rejection case."""

    @staticmethod
    def _validator() -> Callable[[Any], Any]:
        spec = GeoDistanceFeatureGroup.PROPERTY_MAPPING[DefaultOptionKeys.in_features]
        validator = spec.element_validator
        assert validator is not None, "geo_distance in_features should declare an element_validator"
        return validator

    @pytest.mark.parametrize("point", ["customer_location", "store_location", "a"])
    def test_accepts_point_feature_names(self, point: str) -> None:
        assert self._validator()(point) is True

    @pytest.mark.parametrize("point", [1, 1.5, None, True, ["a"], ("a",), {"a": 1}, object()])
    def test_rejects_non_string_points(self, point: Any) -> None:
        """Anything that is not a feature name must be rejected, not coerced."""
        assert self._validator()(point) is False

    def test_arity_is_enforced_separately(self) -> None:
        """The validator sees one point, so the exactly-two rule lives in MIN/MAX_IN_FEATURES."""
        assert GeoDistanceFeatureGroup.MIN_IN_FEATURES == 2
        assert GeoDistanceFeatureGroup.MAX_IN_FEATURES == 2
