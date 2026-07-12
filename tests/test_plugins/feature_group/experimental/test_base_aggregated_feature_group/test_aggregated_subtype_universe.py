"""Tests for the subtype universe on AggregatedFeatureGroup (issue #639).

Contract under test (to be implemented by the Green agent):

11. ``AggregatedFeatureGroup.SUBTYPE_KEY = AGGREGATION_TYPE``, so the family's
    subtype universe is ``AGGREGATION_TYPES`` and both the string-based and the
    config-based path resolve a concrete feature's subtype. The change is
    behavior-neutral: the default ``supported_subtypes()`` is the full universe,
    so every framework still supports every aggregation type.

All tests fail until the feature exists.
"""

from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


def test_subtype_key_is_the_aggregation_type() -> None:
    assert AggregatedFeatureGroup.SUBTYPE_KEY == AggregatedFeatureGroup.AGGREGATION_TYPE
    assert AggregatedFeatureGroup.SUBTYPE_KEY == "aggregation_type"


def test_universe_is_the_declared_aggregation_types() -> None:
    assert AggregatedFeatureGroup.subtype_universe() == frozenset(AggregatedFeatureGroup.AGGREGATION_TYPES)


def test_universe_is_inherited_by_the_framework_subclass() -> None:
    assert PandasAggregatedFeatureGroup.subtype_universe() == frozenset(AggregatedFeatureGroup.AGGREGATION_TYPES)


def test_resolve_subtype_from_the_feature_name() -> None:
    assert AggregatedFeatureGroup.resolve_subtype("sales__sum_aggr", Options()) == "sum"


def test_resolve_subtype_from_the_config_based_path() -> None:
    options = Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "max"})

    assert AggregatedFeatureGroup.resolve_subtype("placeholder", options) == "max"


def test_resolve_subtype_returns_none_for_an_unrelated_name() -> None:
    assert AggregatedFeatureGroup.resolve_subtype("sales", Options()) is None


def test_supported_subtypes_default_to_the_full_universe() -> None:
    universe = AggregatedFeatureGroup.subtype_universe()

    assert PandasAggregatedFeatureGroup.supported_subtypes(PandasDataFrame) == universe
    assert PandasAggregatedFeatureGroup.supported_subtypes(PythonDictFramework) == universe


def test_change_is_behavior_neutral_for_supports_compute_framework() -> None:
    """Every declared aggregation type stays supported on every framework."""
    for aggregation_type in AggregatedFeatureGroup.AGGREGATION_TYPES:
        feature_name = f"sales__{aggregation_type}_aggr"
        assert (
            PandasAggregatedFeatureGroup.supports_compute_framework(feature_name, Options(), PandasDataFrame) is True
        ), f"'{aggregation_type}' must stay supported on PandasDataFrame"


def test_subtype_support_matrix_gives_every_framework_the_full_universe() -> None:
    universe = PandasAggregatedFeatureGroup.subtype_universe()
    matrix = PandasAggregatedFeatureGroup.subtype_support_matrix()

    assert matrix, "The matrix must enumerate the declared compute framework(s)"
    assert all(supported == universe for supported in matrix.values())
