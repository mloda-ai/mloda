"""Tests for the subtype universe on AggregatedFeatureGroup (issue #639).

Contract under test:

- ``AggregatedFeatureGroup.SUBTYPE_KEY = AGGREGATION_TYPE``, so the family's subtype
  universe is exactly the nine declared aggregation types, pinned literally here: a
  test that re-derives the expectation from ``AGGREGATION_TYPES`` cannot catch drift.
- The reported universe equals what strict validation accepts: every subtype matches on
  the string path and on the config path, and a value outside it is rejected.
- The abstract base declares the universe but NO support matrix. Only the concrete,
  per-framework subclasses carry the matrix.
- ``resolve_subtype`` never raises, not even for a name that matches PREFIX_PATTERN
  without a source feature.
"""

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.polars_lazy import (
    PolarsLazyAggregatedFeatureGroup,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pyarrow import PyArrowAggregatedFeatureGroup


# The aggregation universe, pinned literally. Adding or renaming an aggregation type
# must fail here, which is the point of not reading AGGREGATION_TYPES.
AGGREGATION_UNIVERSE = frozenset({"sum", "min", "max", "avg", "mean", "count", "std", "var", "median"})


def test_subtype_key_is_the_aggregation_type() -> None:
    assert AggregatedFeatureGroup.SUBTYPE_KEY == "aggregation_type"
    assert AggregatedFeatureGroup.SUBTYPE_KEY == AggregatedFeatureGroup.AGGREGATION_TYPE


def test_universe_is_the_nine_declared_aggregation_types() -> None:
    assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE


def test_universe_is_inherited_by_the_framework_subclasses() -> None:
    assert PandasAggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
    assert PyArrowAggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
    assert PolarsLazyAggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE


def test_resolve_subtype_from_the_feature_name() -> None:
    assert AggregatedFeatureGroup.resolve_subtype("sales__sum_aggr", Options()) == "sum"


def test_resolve_subtype_from_the_config_based_path() -> None:
    options = Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "max"})

    assert AggregatedFeatureGroup.resolve_subtype("placeholder", options) == "max"


def test_resolve_subtype_returns_none_for_an_unrelated_name() -> None:
    assert AggregatedFeatureGroup.resolve_subtype("sales", Options()) is None


def test_resolve_subtype_never_raises_for_a_name_without_a_source_feature() -> None:
    """'__sum_aggr' matches PREFIX_PATTERN but carries no source feature."""
    assert AggregatedFeatureGroup.resolve_subtype("__sum_aggr", Options()) is None


def test_capability_hook_never_raises_for_a_name_without_a_source_feature() -> None:
    """A match-time capability hook must return a bool, never raise."""
    assert AggregatedFeatureGroup.supports_compute_framework("__sum_aggr", Options(), PandasDataFrame) is True


def test_supported_subtypes_default_to_the_full_universe() -> None:
    assert PandasAggregatedFeatureGroup.supported_subtypes(PandasDataFrame) == AGGREGATION_UNIVERSE
    assert PandasAggregatedFeatureGroup.supported_subtypes(PythonDictFramework) == AGGREGATION_UNIVERSE


def test_change_is_behavior_neutral_for_supports_compute_framework() -> None:
    """Every declared aggregation type stays supported on every framework."""
    for aggregation_type in sorted(AGGREGATION_UNIVERSE):
        feature_name = f"sales__{aggregation_type}_aggr"
        assert (
            PandasAggregatedFeatureGroup.supports_compute_framework(feature_name, Options(), PandasDataFrame) is True
        ), f"'{aggregation_type}' must stay supported on PandasDataFrame"


def test_abstract_base_declares_the_universe_but_no_support_matrix() -> None:
    """AggregatedFeatureGroup is abstract: it must not claim support on every installed framework."""
    assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
    assert AggregatedFeatureGroup.subtype_support_matrix() == {}, (
        "The abstract base declares the universe, not the support. Frameworks without an "
        "aggregation implementation (SqliteFramework, SparkFramework, ...) must not appear."
    )


def test_concrete_framework_subclasses_carry_the_support_matrix() -> None:
    assert PandasAggregatedFeatureGroup.subtype_support_matrix() == {"PandasDataFrame": AGGREGATION_UNIVERSE}
    assert PyArrowAggregatedFeatureGroup.subtype_support_matrix() == {"PyArrowTable": AGGREGATION_UNIVERSE}
    assert PolarsLazyAggregatedFeatureGroup.subtype_support_matrix() == {"PolarsLazyDataFrame": AGGREGATION_UNIVERSE}


def test_compute_framework_definitions_back_the_concrete_matrices() -> None:
    assert PandasAggregatedFeatureGroup.compute_framework_definition() == {PandasDataFrame}
    assert PyArrowAggregatedFeatureGroup.compute_framework_definition() == {PyArrowTable}
    assert PolarsLazyAggregatedFeatureGroup.compute_framework_definition() == {PolarsLazyDataFrame}


class TestUniverseEqualsWhatStrictValidationAccepts:
    """The load-bearing invariant of issue #639: no drift between universe and strict validation."""

    def test_every_subtype_matches_on_the_string_path(self) -> None:
        for aggregation_type in sorted(AGGREGATION_UNIVERSE):
            feature_name = f"sales__{aggregation_type}_aggr"
            assert AggregatedFeatureGroup.match_feature_group_criteria(feature_name, Options()) is True, (
                f"'{aggregation_type}' is in the reported universe, so '{feature_name}' must match"
            )

    def test_every_subtype_matches_on_the_config_path(self) -> None:
        for aggregation_type in sorted(AGGREGATION_UNIVERSE):
            options = Options(
                context={
                    AggregatedFeatureGroup.AGGREGATION_TYPE: aggregation_type,
                    DefaultOptionKeys.in_features: "sales",
                }
            )
            assert AggregatedFeatureGroup.match_feature_group_criteria("placeholder", options) is True, (
                f"'{aggregation_type}' is in the reported universe, so strict validation must accept it"
            )

    def test_a_value_outside_the_universe_is_rejected_on_the_config_path(self) -> None:
        options = Options(
            context={
                AggregatedFeatureGroup.AGGREGATION_TYPE: "p95",
                DefaultOptionKeys.in_features: "sales",
            }
        )

        assert "p95" not in AGGREGATION_UNIVERSE
        assert AggregatedFeatureGroup.match_feature_group_criteria("placeholder", options) is False

    def test_every_subtype_resolves_back_to_itself(self) -> None:
        for aggregation_type in sorted(AGGREGATION_UNIVERSE):
            resolved = AggregatedFeatureGroup.resolve_subtype(f"sales__{aggregation_type}_aggr", Options())
            assert resolved == aggregation_type
