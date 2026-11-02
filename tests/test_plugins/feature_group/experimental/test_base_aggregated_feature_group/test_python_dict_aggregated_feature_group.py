from typing import Any

import pandas as pd
import pytest

from mloda.provider import FeatureSet
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup
from mloda_plugins.feature_group.experimental.aggregated_feature_group.python_dict import PythonDictAggregation

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.aggregated_add_result_test_mixin import (
    AggregatedAddResultTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.aggregated_zero_row_test_mixin import (
    AggregatedZeroRowTestMixin,
)
from tests.test_plugins.feature_group.experimental.test_base_aggregated_feature_group.test_aggregated_utils import (
    PythonDictAggregatedTestDataCreator,
    validate_aggregated_features,
)
from tests.test_plugins.feature_group.experimental.zero_row_result_type_test_mixin import PythonDictZeroRowAdapter


@pytest.fixture
def sample_data() -> dict[str, list[Any]]:
    return {"sales": [100, 200, 300, 400, 500], "price": [10.0, 9.5, 9.0, 8.5, 8.0]}


class TestPythonDictAggregation:
    def test_compute_framework_rule(self) -> None:
        assert PythonDictAggregation.compute_framework is PythonDictFramework
        assert PythonDictAggregation.feature_group is AggregatedFeatureGroup
        assert PythonDictFramework in AggregatedFeatureGroup.compute_framework_definition()

    def test_perform_aggregation_invalid(self, sample_data: dict[str, list[Any]]) -> None:
        with pytest.raises(ValueError, match="Unsupported aggregation type: invalid"):
            PythonDictAggregation._perform_aggregation(sample_data, "invalid", ["sales"])

    def test_perform_aggregation_multi_column_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            PythonDictAggregation._perform_aggregation({"m~0": [1, 2, 3], "m~1": [1]}, "sum", ["m~0", "m~1"])

    def test_calculate_feature_multiple(self, sample_data: dict[str, list[Any]]) -> None:
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__sum_aggr"))
        feature_set.add(Feature("price__avg_aggr"))

        result = PythonDictAggregation.calculate(AggregatedFeatureGroup, sample_data, feature_set, None)

        assert result["sales__sum_aggr"] == [1500] * 5
        assert result["price__avg_aggr"] == [pytest.approx(9.0)] * 5
        assert result["sales"] == [100, 200, 300, 400, 500]

    def test_calculate_leaves_the_input_dict_unchanged(self, sample_data: dict[str, list[Any]]) -> None:
        feature_set = FeatureSet()
        feature_set.add(Feature("sales__sum_aggr"))
        before = {k: list(v) for k, v in sample_data.items()}

        PythonDictAggregation.calculate(AggregatedFeatureGroup, sample_data, feature_set, None)

        assert sample_data == before

    def test_calculate_feature_missing_source(self, sample_data: dict[str, list[Any]]) -> None:
        error = assert_core_rejects_missing_input(PythonDictAggregation, sample_data, "missing__sum_aggr", ("missing",))
        assert "sales" in error.available_columns


class TestPythonDictAggregatedZeroRow(PythonDictZeroRowAdapter, AggregatedZeroRowTestMixin):
    feature_group_class = PythonDictAggregation


class TestPythonDictAggregationAddResult(PythonDictZeroRowAdapter, AggregatedAddResultTestMixin):
    feature_group_class = PythonDictAggregation


class TestAggPythonDictIntegration:
    def test_aggregation_with_data_creator(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PythonDictAggregatedTestDataCreator, AggregatedFeatureGroup}
        )

        result = mloda.run_all(
            [
                "sales",
                "sales__sum_aggr",
                "price__avg_aggr",
                "discount__min_aggr",
                "customer_rating__max_aggr",
                "quantity__count_aggr",
                "discount__median_aggr",
            ],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=plugin_collector,
        )

        validate_aggregated_features([pd.DataFrame(r) for r in result])
