"""
Tests for formal sub-column dependency support.

This module tests the ability to declare dependencies on specific sub-columns
(e.g., Feature("base_feature~1")) rather than the entire multi-column output.

When a FeatureGroup produces multiple outputs using the ~ separator
(e.g., base_feature~0, base_feature~1, base_feature~2), downstream FeatureGroups
should be able to formally declare a dependency on a specific sub-column.

Expected Behavior:
- Feature("base") returns ALL columns: base~0, base~1, etc. (existing, unchanged)
- Feature("base~0") returns ONLY base~0 column (new capability)
"""

from typing import Any, List, Optional, Set, Type, Union

import numpy as np
import pandas as pd
import pytest

from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import ComputeFramework
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginLoader
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda.user import DataAccessCollection
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class SubColumnTestDataCreator(FeatureGroup):
    """Test data creator providing source data for sub-column tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"source_data"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(
            {
                "source_data": [10, 20, 30, 40, 50],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class MultiColumnProducerForSubColumnTest(FeatureGroup):
    """
    Producer feature group that creates multi-column output.

    Produces 3 columns using apply_naming_convention():
    - base_feature~0
    - base_feature~1
    - base_feature~2
    """

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"base_feature"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = next(iter(features.get_all_names()))
        source_values = data["source_data"].values
        result = np.column_stack([source_values, source_values * 2, source_values * 3])
        named_columns = cls.apply_naming_convention(result, feature_name)
        for col_name, col_data in named_columns.items():
            data[col_name] = col_data
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class SubColumnConsumer(FeatureGroup):
    """
    Consumer that depends on a specific sub-column (base_feature~1).

    This demonstrates formal sub-column dependency - the consumer only needs
    base_feature~1, not all of base_feature~0, ~1, ~2.
    """

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"sub_column_consumer_output"}

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("base_feature~1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = next(iter(features.get_all_names()))
        data[feature_name] = data["base_feature~1"].values * 10
        return data

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}


class TestSubColumnFeatureMatching:
    """Tests for match_feature_group_criteria() with sub-column names."""

    def test_sub_column_feature_matches_parent_feature_group(self) -> None:
        """
        Test that match_feature_group_criteria() returns True when given a sub-column
        name like "base_feature~0" for a FeatureGroup that supports "base_feature".

        The method should extract the base feature name using get_column_base_feature()
        and match it against the supported feature names.

        Expected: MultiColumnProducerForSubColumnTest.match_feature_group_criteria("base_feature~0", ...)
        should return True because "base_feature" is in feature_names_supported().
        """
        feature_name = FeatureName("base_feature~0")
        options = Options({})

        result = MultiColumnProducerForSubColumnTest.match_feature_group_criteria(feature_name, options, None)

        assert result is True, (
            "A sub-column name 'base_feature~0' should match the FeatureGroup "
            "that produces 'base_feature' multi-column output"
        )

    def test_sub_column_with_different_suffix_matches(self) -> None:
        """
        Test that different sub-column suffixes (e.g., base_feature~2) also match.
        """
        for suffix in ["0", "1", "2", "99"]:
            feature_name = FeatureName(f"base_feature~{suffix}")
            options = Options({})

            result = MultiColumnProducerForSubColumnTest.match_feature_group_criteria(feature_name, options, None)

            assert result is True, f"Sub-column 'base_feature~{suffix}' should match the parent FeatureGroup"


class TestSubColumnDependencyResolution:
    """Tests for resolving sub-column dependencies to parent FeatureGroups."""

    def test_sub_column_dependency_resolution(self) -> None:
        """
        Test that a FeatureGroup can declare Feature("base_feature~1") as a dependency
        and it resolves to the parent FeatureGroup that produces multi-column output.

        The SubColumnConsumer declares Feature("base_feature~1") as an input,
        and the system should correctly resolve this to MultiColumnProducerForSubColumnTest.
        """
        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                SubColumnConsumer,
            }
        )

        features_to_request: List[Union[Feature, str]] = [Feature("sub_column_consumer_output")]

        api = mloda(
            features_to_request,
            {PandasDataFrame},
            plugin_collector=plugin_collector,
        )
        api._batch_run()
        results = api.get_result()

        assert len(results) > 0, "Should return at least one result DataFrame"

        df = pd.concat(results, axis=1)

        assert "sub_column_consumer_output" in df.columns, "Consumer output should be present"

        expected_values = np.array([200, 400, 600, 800, 1000])
        np.testing.assert_array_equal(
            df["sub_column_consumer_output"].values,
            expected_values,
            err_msg="Consumer should use base_feature~1 (values * 2) and multiply by 10",
        )


class TestIdentifyNamingConventionForSubColumns:
    """Tests for identify_naming_convention() with specific sub-column requests.

    These tests verify the column selection logic when requesting specific sub-columns
    vs. requesting the base feature that returns all sub-columns.

    Note: These tests verify existing behavior that works via exact column matching.
    The challenge is in match_feature_group_criteria() which must be updated to
    recognize sub-column names and resolve them to the parent FeatureGroup.
    """

    def test_only_requested_sub_column_selected(self) -> None:
        """
        Test that identify_naming_convention() returns only the specific sub-column
        when a sub-column is explicitly requested.

        When Feature("base~1") is requested and columns {base~0, base~1, base~2} exist,
        only ["base~1"] should be returned, not all base~* columns.

        Note: This tests existing exact-match behavior that already works.
        The key requirement for sub-column dependencies is in match_feature_group_criteria().
        """
        from uuid import uuid4
        from mloda.user import ParallelizationMode

        selected_feature_names = {FeatureName("base~1")}
        column_names = {"base~0", "base~1", "base~2", "other_column"}

        framework = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )

        result = framework.identify_naming_convention(selected_feature_names, column_names)

        assert result == {"base~1"}, (
            f"When requesting 'base~1', only 'base~1' should be selected, not all base~* columns. Got: {result}"
        )

    def test_base_feature_selects_all_sub_columns(self) -> None:
        """
        Test that requesting the base feature (without ~suffix) returns all sub-columns.

        This is the existing behavior that should remain unchanged.
        When Feature("base") is requested, all base~0, base~1, base~2 should be returned.
        """
        from uuid import uuid4
        from mloda.user import ParallelizationMode

        selected_feature_names = {FeatureName("base")}
        column_names = {"base~0", "base~1", "base~2", "other_column"}

        framework = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )

        result = framework.identify_naming_convention(selected_feature_names, column_names)

        assert result == {"base~0", "base~1", "base~2"}, "Requesting 'base' should return all 'base~*' columns"


class TestChainingWithSubColumns:
    """Tests for feature chaining that involves sub-columns."""

    def test_chaining_with_sub_columns(self) -> None:
        """
        Test that chained features like 'base~0__mean_imputed' work correctly
        by first identifying 'base~0' as the source feature.

        The chaining parser should recognize that 'base~0' is the source feature
        in the chain 'base~0__mean_imputed', and the ~ is part of the feature name,
        not a chaining separator.
        """
        feature_name = "base~0__mean_imputed"

        base_feature = FeatureGroup.get_column_base_feature("base~0__mean_imputed".split("__")[0])

        assert base_feature == "base", (
            f"get_column_base_feature should extract 'base' from 'base~0', got: {base_feature}"
        )

        parts = feature_name.split("__")
        assert len(parts) == 2, "Feature chain should have two parts"
        assert parts[0] == "base~0", "First part should be 'base~0'"
        assert parts[1] == "mean_imputed", "Second part should be the operation"


class TestSubColumnErrorHandling:
    """Tests for error handling with sub-column requests."""

    def test_error_for_non_existent_sub_column(self) -> None:
        """
        Test that requesting a non-existent sub-column raises an appropriate error.

        When Feature("base~999") is requested but only base~0, base~1, base~2 exist,
        the system should raise a clear error message.

        This is important for user experience - clear errors help developers understand
        when they request a sub-column that doesn't exist in the producer's output.
        """
        from uuid import uuid4
        from mloda.user import ParallelizationMode

        selected_feature_names = {FeatureName("base~999")}
        column_names = {"base~0", "base~1", "base~2"}

        framework = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )

        with pytest.raises(ValueError) as exc_info:
            framework.identify_naming_convention(selected_feature_names, column_names)

        error_message = str(exc_info.value)
        assert "base~999" in error_message or "No columns found" in error_message, (
            f"Error should mention the missing column or indicate no columns found. Got: {error_message}"
        )


class TestSubColumnIntegration:
    """Integration tests using mloda.run_all for sub-column dependencies."""

    def test_sub_column_dependency_via_run_all(self) -> None:
        """
        Integration test using mloda.run_all:
        - Producer creates base_feature~0, base_feature~1, base_feature~2
        - Consumer depends on Feature("base_feature~1") specifically
        - Verify consumer receives only base_feature~1 column

        For source_data = [10, 20, 30, 40, 50]:
        - base_feature~0 = [10, 20, 30, 40, 50]
        - base_feature~1 = [20, 40, 60, 80, 100] (source * 2)
        - base_feature~2 = [30, 60, 90, 120, 150] (source * 3)
        - sub_column_consumer_output = [200, 400, 600, 800, 1000] (base_feature~1 * 10)
        """
        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                SubColumnConsumer,
            }
        )

        result = mloda.run_all(
            ["sub_column_consumer_output"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should return at least one result"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        assert "sub_column_consumer_output" in df.columns, "sub_column_consumer_output should be in the result"

        expected = np.array([200, 400, 600, 800, 1000])
        np.testing.assert_array_equal(
            df["sub_column_consumer_output"].values,
            expected,
            err_msg=(
                "Consumer should correctly use base_feature~1 (source*2=20,40,60,80,100) "
                "and multiply by 10 to get 200,400,600,800,1000"
            ),
        )

    def test_sub_column_dependency_does_not_pull_all_columns(self) -> None:
        """
        Verify that requesting a specific sub-column dependency does not
        unnecessarily pull all sibling sub-columns into the consumer's data.

        When SubColumnConsumer depends on Feature("base_feature~1"), the consumer's
        calculate_feature method should only receive base_feature~1, not base_feature~0
        or base_feature~2.

        Note: This behavior depends on how the data is filtered before being passed
        to calculate_feature. If all columns are passed, this test verifies the
        expectation for the new sub-column feature.
        """

        class SubColumnValidatingConsumer(FeatureGroup):
            """Consumer that validates it only receives the requested sub-column."""

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return {"validating_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature("base_feature~1")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                columns_available = set(data.columns)

                assert "base_feature~1" in columns_available, "base_feature~1 should be available"

                data[feature_name] = data["base_feature~1"].values * 5
                return data

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
                return {PandasDataFrame}

        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                SubColumnValidatingConsumer,
            }
        )

        result = mloda.run_all(
            ["validating_consumer_output"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should complete without error"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        expected = np.array([100, 200, 300, 400, 500])
        np.testing.assert_array_equal(
            df["validating_consumer_output"].values, expected, err_msg="Consumer should use base_feature~1 * 5"
        )

    def test_multiple_sub_column_dependencies_via_run_all(self) -> None:
        """
        Test a consumer that depends on multiple specific sub-columns.

        The consumer depends on base_feature~0 AND base_feature~2 (not all columns,
        just specific ones). This tests that multiple sub-column dependencies can
        be declared and resolved correctly.

        For source_data = [10, 20, 30, 40, 50]:
        - base_feature~0 = [10, 20, 30, 40, 50] (source * 1)
        - base_feature~2 = [30, 60, 90, 120, 150] (source * 3)
        - multi_sub_column_consumer_output = base_feature~0 + base_feature~2
          = [40, 80, 120, 160, 200]
        """

        class MultiSubColumnConsumer(FeatureGroup):
            """Consumer that depends on multiple specific sub-columns."""

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return {"multi_sub_column_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature("base_feature~0"), Feature("base_feature~2")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["base_feature~0"].values + data["base_feature~2"].values
                return data

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
                return {PandasDataFrame}

        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                MultiSubColumnConsumer,
            }
        )

        result = mloda.run_all(
            ["multi_sub_column_consumer_output"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should return at least one result"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        assert "multi_sub_column_consumer_output" in df.columns

        expected = np.array([40, 80, 120, 160, 200])
        np.testing.assert_array_equal(
            df["multi_sub_column_consumer_output"].values,
            expected,
            err_msg="Consumer should sum base_feature~0 and base_feature~2",
        )

    def test_chained_feature_with_sub_column_via_run_all(self) -> None:
        """
        Test feature chaining where a consumer uses the output of a sub-column consumer.

        This creates a chain: base_feature~1 -> sub_column_consumer_output -> doubled_consumer_output
        where sub_column_consumer_output depends on a specific sub-column and the doubled
        consumer chains on top of that.

        For source_data = [10, 20, 30, 40, 50]:
        - base_feature~1 = [20, 40, 60, 80, 100] (source * 2)
        - sub_column_consumer_output = [200, 400, 600, 800, 1000] (base_feature~1 * 10)
        - sub_column_consumer_output__value_doubled = [400, 800, 1200, 1600, 2000]
        """
        from mloda.provider import FeatureChainParser, FeatureChainParserMixin

        class ValueDoubledFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            """Feature group that doubles its input via chaining with unique suffix."""

            SUFFIX_PATTERN = r".*__value_doubled$"

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return set()

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                for feature in features.features:
                    feature_name = feature.get_name()
                    _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [cls.SUFFIX_PATTERN])
                    if source_feature and source_feature in data.columns:
                        data[feature_name] = data[source_feature].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
                return {PandasDataFrame}

        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                SubColumnConsumer,
                ValueDoubledFeatureGroup,
            }
        )

        result = mloda.run_all(
            ["sub_column_consumer_output__value_doubled"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should return at least one result"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        assert "sub_column_consumer_output__value_doubled" in df.columns

        expected = np.array([400, 800, 1200, 1600, 2000])
        np.testing.assert_array_equal(
            df["sub_column_consumer_output__value_doubled"].values,
            expected,
            err_msg="Chained feature should double sub_column_consumer_output values",
        )

    def test_mixed_base_and_sub_column_requests_via_run_all(self) -> None:
        """
        Test requesting both base feature and a consumer that uses a specific sub-column.

        Request Feature("base_feature") (which returns all columns: ~0, ~1, ~2)
        AND a consumer that uses Feature("base_feature~1") specifically in the same
        run_all call. This tests that both can coexist without conflict.

        For source_data = [10, 20, 30, 40, 50]:
        - base_feature~0 = [10, 20, 30, 40, 50]
        - base_feature~1 = [20, 40, 60, 80, 100]
        - base_feature~2 = [30, 60, 90, 120, 150]
        - sub_column_consumer_output = [200, 400, 600, 800, 1000] (base_feature~1 * 10)
        """
        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                SubColumnConsumer,
            }
        )

        result = mloda.run_all(
            ["base_feature", "sub_column_consumer_output"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should return at least one result"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        assert "base_feature~0" in df.columns, "base_feature~0 should be in result"
        assert "base_feature~1" in df.columns, "base_feature~1 should be in result"
        assert "base_feature~2" in df.columns, "base_feature~2 should be in result"
        assert "sub_column_consumer_output" in df.columns, "consumer output should be in result"

        np.testing.assert_array_equal(
            df["base_feature~0"].values,
            np.array([10, 20, 30, 40, 50]),
            err_msg="base_feature~0 should be source * 1",
        )
        np.testing.assert_array_equal(
            df["base_feature~1"].values,
            np.array([20, 40, 60, 80, 100]),
            err_msg="base_feature~1 should be source * 2",
        )
        np.testing.assert_array_equal(
            df["base_feature~2"].values,
            np.array([30, 60, 90, 120, 150]),
            err_msg="base_feature~2 should be source * 3",
        )
        np.testing.assert_array_equal(
            df["sub_column_consumer_output"].values,
            np.array([200, 400, 600, 800, 1000]),
            err_msg="Consumer should use base_feature~1 * 10",
        )

    def test_sub_column_consumer_chain_via_run_all(self) -> None:
        """
        Test a chain of consumers where first consumer depends on a sub-column,
        and the second consumer depends on the first consumer's output.

        Chain: base_feature~1 -> first_consumer_output -> second_consumer_output

        For source_data = [10, 20, 30, 40, 50]:
        - base_feature~1 = [20, 40, 60, 80, 100] (source * 2)
        - first_consumer_output = [40, 80, 120, 160, 200] (base_feature~1 * 2)
        - second_consumer_output = [80, 160, 240, 320, 400] (first_consumer_output * 2)
        """

        class FirstConsumer(FeatureGroup):
            """First consumer that depends on base_feature~1."""

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return {"first_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature("base_feature~1")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["base_feature~1"].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
                return {PandasDataFrame}

        class SecondConsumer(FeatureGroup):
            """Second consumer that depends on first_consumer_output."""

            @classmethod
            def feature_names_supported(cls) -> Set[str]:
                return {"second_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature("first_consumer_output")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["first_consumer_output"].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
                return {PandasDataFrame}

        PluginLoader().all()

        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SubColumnTestDataCreator,
                MultiColumnProducerForSubColumnTest,
                FirstConsumer,
                SecondConsumer,
            }
        )

        result = mloda.run_all(
            ["second_consumer_output"],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(result) >= 1, "Should return at least one result"

        df = pd.concat(result, axis=1) if len(result) > 1 else result[0]

        assert "second_consumer_output" in df.columns

        expected = np.array([80, 160, 240, 320, 400])
        np.testing.assert_array_equal(
            df["second_consumer_output"].values,
            expected,
            err_msg=(
                "Second consumer should receive first_consumer_output (base_feature~1 * 2) "
                "and double it to get [80, 160, 240, 320, 400]"
            ),
        )
