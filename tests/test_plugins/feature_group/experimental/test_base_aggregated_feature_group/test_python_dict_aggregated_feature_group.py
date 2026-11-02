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


@pytest.fixture
def multi_source_data() -> dict[str, list[Any]]:
    return {"metrics~0": [1, 2, 3, 4], "metrics~1": [10, 20, 30, 40]}


class TestPythonDictAggregation:
    def test_compute_framework_rule(self) -> None:
        assert PythonDictAggregation.compute_framework is PythonDictFramework
        assert PythonDictAggregation.feature_group is AggregatedFeatureGroup
        assert PythonDictFramework in AggregatedFeatureGroup.compute_framework_definition()

    def test_perform_aggregation_invalid(self, sample_data: dict[str, list[Any]]) -> None:
        with pytest.raises(ValueError, match="Unsupported aggregation type: invalid"):
            PythonDictAggregation._perform_aggregation(sample_data, "invalid", ["sales"])

    def test_perform_aggregation_multi_column_is_row_wise(self, multi_source_data: dict[str, list[Any]]) -> None:
        result = PythonDictAggregation._perform_aggregation(multi_source_data, "sum", ["metrics~0", "metrics~1"])
        assert list(result) == [11, 22, 33, 44]

    def test_perform_aggregation_multi_column_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            PythonDictAggregation._perform_aggregation({"m~0": [1, 2, 3], "m~1": [1]}, "sum", ["m~0", "m~1"])

    def test_add_result_to_data_broadcasts_a_scalar(self, sample_data: dict[str, list[Any]]) -> None:
        updated = PythonDictAggregation._add_result_to_data(sample_data, "sales__sum_aggr", 1500)
        assert updated["sales__sum_aggr"] == [1500] * 5
        assert updated["sales"] == [100, 200, 300, 400, 500]

    def test_add_result_to_data_sets_a_list_as_the_column(self, multi_source_data: dict[str, list[Any]]) -> None:
        updated = PythonDictAggregation._add_result_to_data(multi_source_data, "m__sum_aggr", [11, 22, 33, 44])
        assert updated["m__sum_aggr"] == [11, 22, 33, 44]

    def test_add_result_to_data_replaces_an_existing_column(self, sample_data: dict[str, list[Any]]) -> None:
        once = PythonDictAggregation._add_result_to_data(sample_data, "x__sum_aggr", 1)
        twice = PythonDictAggregation._add_result_to_data(once, "x__sum_aggr", 2)
        assert twice["x__sum_aggr"] == [2] * 5
        assert list(twice).count("x__sum_aggr") == 1

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
        keys_before = list(sample_data)

        PythonDictAggregation.calculate(AggregatedFeatureGroup, sample_data, feature_set, None)

        assert list(sample_data) == keys_before

    def test_calculate_feature_missing_source(self, sample_data: dict[str, list[Any]]) -> None:
        error = assert_core_rejects_missing_input(PythonDictAggregation, sample_data, "missing__sum_aggr", ("missing",))
        assert "sales" in error.available_columns


class TestPythonDictAggregatedZeroRow(PythonDictZeroRowAdapter, AggregatedZeroRowTestMixin):
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
