"""Failing tests pinning the subtype-universe dogfooding of AggregatedFeatureGroup (issue #639, Phase 2)."""

import inspect

from mloda.core.abstract_plugins.components.options import Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.pandas import PandasAggregatedFeatureGroup


AGGREGATION_UNIVERSE = frozenset(AggregatedFeatureGroup.AGGREGATION_TYPES)


class TestAggregatedSubtypeDeclaration:
    """AggregatedFeatureGroup declares its aggregation types as its subtype universe."""

    def test_subtype_key_is_aggregation_type(self) -> None:
        assert AggregatedFeatureGroup.SUBTYPE_KEY == AggregatedFeatureGroup.AGGREGATION_TYPE
        assert AggregatedFeatureGroup.SUBTYPE_KEY == "aggregation_type"

    def test_universe_matches_declared_aggregation_types(self) -> None:
        assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE

    def test_pandas_subclass_inherits_universe(self) -> None:
        assert PandasAggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE


class TestAggregatedResolveSubtype:
    """resolve_subtype resolves aggregation types from the name and the config path."""

    def test_resolves_from_feature_name(self) -> None:
        assert AggregatedFeatureGroup.resolve_subtype("sales__sum_aggr", Options()) == "sum"

    def test_resolves_from_config_path(self) -> None:
        options = Options(context={AggregatedFeatureGroup.AGGREGATION_TYPE: "max"})
        assert AggregatedFeatureGroup.resolve_subtype("aggr_config_placeholder", options) == "max"

    def test_unrelated_name_resolves_to_none(self) -> None:
        # Anchor on the declaration so this test is red until SUBTYPE_KEY lands.
        assert AggregatedFeatureGroup.SUBTYPE_KEY == AggregatedFeatureGroup.AGGREGATION_TYPE
        assert AggregatedFeatureGroup.resolve_subtype("sales_total", Options()) is None


class TestAggregatedSupportedSubtypes:
    """supported_subtypes defaults to the full universe on the concrete Pandas class."""

    def test_default_is_full_universe_for_any_framework(self) -> None:
        assert PandasAggregatedFeatureGroup.supported_subtypes(PandasDataFrame) == AGGREGATION_UNIVERSE
        assert PandasAggregatedFeatureGroup.supported_subtypes(PythonDictFramework) == AGGREGATION_UNIVERSE


class TestAggregatedBehaviorNeutral:
    """The declaration is behavior-neutral: every aggregation type stays supported on PandasDataFrame."""

    def test_every_aggregation_type_stays_supported_on_pandas(self) -> None:
        # Anchor on the declared universe so this test is red until SUBTYPE_KEY lands.
        assert PandasAggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
        for aggregation_type in AggregatedFeatureGroup.AGGREGATION_TYPES:
            supported = PandasAggregatedFeatureGroup.supports_compute_framework(
                f"sales__{aggregation_type}_aggr", Options(), PandasDataFrame
            )
            assert supported is True, f"'{aggregation_type}' must stay supported on PandasDataFrame"


class TestAggregatedSubtypeSupportMatrix:
    """subtype_support_matrix reflects abstractness and gives every framework the full universe."""

    def test_abstract_base_declares_universe_but_no_matrix(self) -> None:
        assert inspect.isabstract(AggregatedFeatureGroup)
        # Anchor on the declared universe so this test is red until SUBTYPE_KEY lands.
        assert AggregatedFeatureGroup.subtype_universe() == AGGREGATION_UNIVERSE
        assert AggregatedFeatureGroup.subtype_support_matrix() == {}

    def test_pandas_matrix_gives_every_framework_the_full_universe(self) -> None:
        matrix = PandasAggregatedFeatureGroup.subtype_support_matrix()
        assert matrix, "the concrete Pandas class must have a non-empty support matrix"
        declared = {cfw.get_class_name() for cfw in PandasAggregatedFeatureGroup.compute_framework_definition()}
        assert set(matrix) == declared
        assert matrix == {PandasDataFrame.get_class_name(): AGGREGATION_UNIVERSE}
