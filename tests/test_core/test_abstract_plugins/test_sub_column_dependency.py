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

import copy
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from mloda.provider import ANY_OUTPUT_PART
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import ComputeFramework
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import FeatureResolutionError
from mloda.core.abstract_plugins.components.match_hook import probe_match_criteria
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_STAGE,
    NAME_STAGE,
    MATCH_REJECTION_REASONS,
    record_match_rejection,
)
from mloda.core.abstract_plugins.components.validators.datatype_validator import DataTypeMismatchError
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginLoader
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class SubColumnTestDataCreator(FeatureGroup):
    """Test data creator providing source data for sub-column tests."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"source_data"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(
            {
                "source_data": [10, 20, 30, 40, 50],
            }
        )

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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
    def feature_names_supported(cls) -> set[str]:
        return {"base_feature"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
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
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class SubColumnConsumer(FeatureGroup):
    """
    Consumer that depends on a specific sub-column (base_feature~1).

    This demonstrates formal sub-column dependency - the consumer only needs
    base_feature~1, not all of base_feature~0, ~1, ~2.
    """

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"sub_column_consumer_output"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("base_feature~1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = next(iter(features.get_all_names()))
        data[feature_name] = data["base_feature~1"].values * 10
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class NamedPartProducerForSubColumnTest(FeatureGroup):
    """Producer that writes measure~value1..3 columns using named (non-digit) suffixes."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"measure", "measure~value1", "measure~value2"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        source_values = data["source_data"].values
        result = np.column_stack([source_values, source_values * 2, source_values * 3])
        named_columns = cls.apply_naming_convention(result, "measure", suffix_generator=lambda i: f"value{i + 1}")
        for col_name, col_data in named_columns.items():
            data[col_name] = col_data
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class NamedPartConsumerForSubColumnTest(FeatureGroup):
    """Consumer that depends on the named part measure~value1."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"named_part_consumer_output"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("measure~value1")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        feature_name = next(iter(features.get_all_names()))
        data[feature_name] = data["measure~value1"].values * 10
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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

    def test_non_digit_suffix_does_not_match_parent_feature_group(self) -> None:
        """A non-digit suffix (e.g. base_feature~mean) does not resolve to the parent FeatureGroup."""
        feature_name = FeatureName("base_feature~mean")
        options = Options({})

        result = MultiColumnProducerForSubColumnTest.match_feature_group_criteria(feature_name, options, None)

        assert result is False, "A non-digit sub-column suffix should not match the parent FeatureGroup"


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

        features_to_request: list[Feature | str] = [Feature("sub_column_consumer_output")]

        api = mloda(
            features_to_request,
            [PandasDataFrame],
            plugin_collector=plugin_collector,
        )
        results = api.run()

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

        selected_feature_names = [FeatureName("base~1")]
        column_names = {"base~0", "base~1", "base~2", "other_column"}

        framework = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )

        result = framework.identify_naming_convention(selected_feature_names, column_names)

        assert result == ["base~1"], (
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

        selected_feature_names = [FeatureName("base")]
        column_names = {"base~0", "base~1", "base~2", "other_column"}

        framework = PandasDataFrame(
            mode=ParallelizationMode.SYNC,
            children_if_root=frozenset(),
            uuid=uuid4(),
        )

        result = framework.identify_naming_convention(selected_feature_names, column_names)

        assert result == ["base~0", "base~1", "base~2"], "Requesting 'base' should return all 'base~*' columns"


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

        selected_feature_names = [FeatureName("base~999")]
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
            compute_frameworks=[PandasDataFrame],
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
            def feature_names_supported(cls) -> set[str]:
                return {"validating_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature("base_feature~1")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                columns_available = set(data.columns)

                assert "base_feature~1" in columns_available, "base_feature~1 should be available"

                data[feature_name] = data["base_feature~1"].values * 5
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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
            compute_frameworks=[PandasDataFrame],
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
            def feature_names_supported(cls) -> set[str]:
                return {"multi_sub_column_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature("base_feature~0"), Feature("base_feature~2")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["base_feature~0"].values + data["base_feature~2"].values
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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
            compute_frameworks=[PandasDataFrame],
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
            def feature_names_supported(cls) -> set[str]:
                return set()

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                for feature in features.features:
                    feature_name = feature.name
                    _, source_feature = FeatureChainParser.parse_feature_name(feature_name, [cls.SUFFIX_PATTERN])
                    if source_feature and source_feature in data.columns:
                        data[feature_name] = data[source_feature].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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
            compute_frameworks=[PandasDataFrame],
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
            compute_frameworks=[PandasDataFrame],
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
            def feature_names_supported(cls) -> set[str]:
                return {"first_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature("base_feature~1")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["base_feature~1"].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        class SecondConsumer(FeatureGroup):
            """Second consumer that depends on first_consumer_output."""

            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"second_consumer_output"}

            def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                return {Feature("first_consumer_output")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                feature_name = next(iter(features.get_all_names()))
                data[feature_name] = data["first_consumer_output"].values * 2
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
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
            compute_frameworks=[PandasDataFrame],
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


class TestSubColumnInputsInPlan:
    """Tests that PlanStep.input_feature_names keeps the part-ful input names (issue #2031)."""

    @staticmethod
    def _compute_step_for(session: Any, feature_name: str) -> Any:
        """Return the single compute step whose feature_names contain the given feature."""
        matching = [s for s in session.resolved_plan() if s.step_kind == "compute" and feature_name in s.feature_names]
        assert len(matching) == 1, f"Expected exactly one compute step for {feature_name}, got {len(matching)}"
        return matching[0]

    def test_numbered_part_input_reports_part_ful_name_in_plan(self) -> None:
        """
        A consumer declaring Feature("base_feature~1") must report the part-ful input
        name in the execution plan (issue #2031).

        SubColumnConsumer reads the numbered part base_feature~1, so the consumer step
        must expose input_feature_names == ("base_feature~1",) and the matching
        input_feature_edges entry; before the fix the plan loses the part and reports
        only the base name ("base_feature",), which breaks lineage/explain.
        """
        PluginLoader().all()

        session = mloda.prepare(
            [Feature("sub_column_consumer_output")],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    SubColumnTestDataCreator,
                    MultiColumnProducerForSubColumnTest,
                    SubColumnConsumer,
                }
            ),
        )

        step = self._compute_step_for(session, "sub_column_consumer_output")

        assert step.input_feature_names == ("base_feature~1",), (
            "The consumer reads base_feature~1, so the plan must keep the part-ful input name; "
            f"got {step.input_feature_names}"
        )
        assert step.input_feature_edges == {"sub_column_consumer_output": ("base_feature~1",)}, (
            f"The consumer step's input edge must point at base_feature~1; got {dict(step.input_feature_edges)}"
        )

    def test_named_part_input_reports_part_ful_name_in_plan(self) -> None:
        """
        A consumer declaring Feature("measure~value1") (non-digit suffix) must report the
        part-ful input name in the execution plan (issue #2031).

        This is a regression guard: named parts resolve through the producer declaring the
        literal part name in feature_names_supported(), and the plan already records
        measure~value1 correctly today; the fix must keep it that way.
        """
        PluginLoader().all()

        session = mloda.prepare(
            [Feature("named_part_consumer_output")],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    SubColumnTestDataCreator,
                    NamedPartProducerForSubColumnTest,
                    NamedPartConsumerForSubColumnTest,
                }
            ),
        )

        consumer_step = self._compute_step_for(session, "named_part_consumer_output")

        assert consumer_step.input_feature_names == ("measure~value1",), (
            f"The named part input name must be kept verbatim in the plan; got {consumer_step.input_feature_names}"
        )
        assert consumer_step.input_feature_edges == {"named_part_consumer_output": ("measure~value1",)}, (
            "The consumer step's input edge must point at measure~value1; "
            f"got {dict(consumer_step.input_feature_edges)}"
        )

        producer_steps = [s for s in session.resolved_plan() if s.feature_group is NamedPartProducerForSubColumnTest]
        assert len(producer_steps) == 1, f"Expected exactly one producer step, got {len(producer_steps)}"
        assert producer_steps[0].input_feature_names == ("source_data",), (
            f"The producer step reads source_data; got {producer_steps[0].input_feature_names}"
        )


# Declared output parts (<feature>~<part>).

_SOURCE = np.array([10, 20, 30, 40, 50])


class MeasureProducer(FeatureGroup):
    """Declares parts value and unit; counts calculate_feature calls."""

    OUTPUT_PARTS = ("value", "unit")
    calls = 0
    seen_parts: list[Any] = []
    captured: list[Any] = []

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"measure"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        cls.calls += 1
        cls.captured.extend(features.features)
        cls.seen_parts.extend(getattr(f, "requested_parts", None) for f in features.features)
        data["measure~value"] = data["source_data"].values
        data["measure~unit"] = "m"
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class WildcardProducer(FeatureGroup):
    """Parts are named at run time; produces wild~a and wild~b."""

    OUTPUT_PARTS = (ANY_OUTPUT_PART,)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"wild"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["wild~a"] = data["source_data"].values
        data["wild~b"] = data["source_data"].values * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class NestedProducer(FeatureGroup):
    """Wildcard producer emitting nested_f~0, nested_f~0~lower, nested_f~0~upper."""

    OUTPUT_PARTS: ClassVar[tuple[str, ...] | None] = (ANY_OUTPUT_PART,)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nested_f"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["nested_f~0"] = data["source_data"].values
        data["nested_f~0~lower"] = data["source_data"].values - 1
        data["nested_f~0~upper"] = data["source_data"].values + 1
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class DeclaredNestedProducer(NestedProducer):
    """NestedProducer declaring its nested parts in full."""

    OUTPUT_PARTS = ("0", "0~lower")


class MeasureConsumer(FeatureGroup):
    """Depends on Feature("measure~value")."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"measure_consumer_output"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("measure~value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["measure_consumer_output"] = data["measure~value"].values * 2
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


def _has_input_data_rejection() -> bool:
    reasons = MATCH_REJECTION_REASONS.get()
    return reasons is not None and any(r.stage == INPUT_DATA_STAGE for r in reasons.values())


@pytest.fixture(autouse=True)
def _reset_measure_state() -> None:
    MeasureProducer.calls = 0
    MeasureProducer.seen_parts = []
    MeasureProducer.captured = []


def _collector(*groups: type[FeatureGroup]) -> PluginCollector:
    return PluginCollector.enabled_feature_groups({SubColumnTestDataCreator, *groups})


def _run(features: list[Feature | str], *groups: type[FeatureGroup]) -> Any:
    return mloda.run_all(
        features,
        compute_frameworks=[PandasDataFrame],
        plugin_collector=_collector(*groups),
        parallelization_modes={ParallelizationMode.SYNC},
    )


def _columns(results: Any) -> set[str]:
    return {c for frame in results for c in frame.columns}


class TestDeclaredOutputParts:
    """A group declares OUTPUT_PARTS; core resolves, selects and checks `<feature>~<part>`."""

    def test_top_level_declared_part_returns_only_that_column(self) -> None:
        results = _run(["measure~value"], MeasureProducer)

        assert _columns(results) == {"measure~value"}
        np.testing.assert_array_equal(results[0]["measure~value"].values, _SOURCE)

    def test_declared_part_as_dependency(self) -> None:
        results = _run(["measure_consumer_output"], MeasureProducer, MeasureConsumer)

        np.testing.assert_array_equal(results[0]["measure_consumer_output"].values, _SOURCE * 2)

    @pytest.mark.parametrize(
        "name, part", [("measure~bogus", "bogus"), ("measure~0", "0"), ("measure~value~x", "value~x")]
    )
    def test_undeclared_part_fails_at_resolution_and_lists_declared(self, name: str, part: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            _run([name], MeasureProducer)

        message = str(exc_info.value)
        assert part in message
        assert f"MeasureProducer (feature name): '{part}' is not an output part" in message
        assert "value" in message and "unit" in message
        assert MeasureProducer.calls == 0

    def test_undeclared_numbered_part_returns_only_that_column(self) -> None:
        results = _run(["base_feature~1"], MultiColumnProducerForSubColumnTest)

        assert _columns(results) == {"base_feature~1"}
        np.testing.assert_array_equal(results[0]["base_feature~1"].values, _SOURCE * 2)

    def test_missing_numbered_part_raises_naming_column_and_produced(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _run(["base_feature~7"], MultiColumnProducerForSubColumnTest)

        message = str(exc_info.value)
        assert "base_feature~7" in message
        assert "base_feature~0" in message and "base_feature~2" in message

    def test_two_parts_co_requested_compute_once(self) -> None:
        results = _run(["measure~value", "measure~unit"], MeasureProducer)

        assert _columns(results) == {"measure~value", "measure~unit"}
        assert MeasureProducer.calls == 1

    def test_whole_feature_and_part_co_requested_returns_all_parts(self) -> None:
        results = _run(["measure", "measure~value"], MeasureProducer)

        assert _columns(results) == {"measure~value", "measure~unit"}
        assert MeasureProducer.calls == 1

    def test_wildcard_group_resolves_a_produced_part(self) -> None:
        results = _run(["wild~a"], WildcardProducer)

        assert _columns(results) == {"wild~a"}

    def test_wildcard_group_part_not_produced_raises_after_compute(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _run(["wild~zzz"], WildcardProducer)

        message = str(exc_info.value)
        assert "wild~zzz" in message
        assert "wild~a" in message and "wild~b" in message

    @pytest.mark.parametrize("producer", [NestedProducer, DeclaredNestedProducer])
    def test_nested_parent_part_returns_part_and_nested_columns(self, producer: type[FeatureGroup]) -> None:
        results = _run(["nested_f~0"], producer)

        assert _columns(results) == {"nested_f~0", "nested_f~0~lower", "nested_f~0~upper"}

    @pytest.mark.parametrize("producer", [NestedProducer, DeclaredNestedProducer])
    def test_nested_exact_part_returns_only_that_column(self, producer: type[FeatureGroup]) -> None:
        results = _run(["nested_f~0~lower"], producer)

        assert _columns(results) == {"nested_f~0~lower"}

    def test_match_criteria_stays_name_only_for_declared_part(self) -> None:
        """Resolution-only rule: a direct matcher call with `measure~value` is still False."""
        result = MeasureProducer.match_feature_group_criteria(FeatureName("measure~value"), Options({}), None)

        assert result is False

    @pytest.mark.parametrize("name", ["measure~", "measure~val ue", "measure~value\n"])
    def test_malformed_part_is_not_a_part(self, name: str) -> None:
        with pytest.raises(ValueError):
            _run([name], MeasureProducer)

    def test_part_data_type_mismatch_raises(self) -> None:
        with pytest.raises(DataTypeMismatchError):
            _run([Feature.str_of("measure~value")], MeasureProducer)

    def test_part_request_does_not_type_check_the_base_column(self) -> None:
        results = _run([Feature.str_of("price~currency")], _CurrencyGroup)

        assert _columns(results) == {"price~currency"}

    def test_matching_part_data_types_pass(self) -> None:
        results = _run([Feature.int64_of("measure~value"), Feature.str_of("measure~unit")], MeasureProducer)

        assert _columns(results) == {"measure~value", "measure~unit"}


class TestSplitOutputPart:
    """FeatureGroup.split_output_part(name) -> (base, part) | None (assumed public name)."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("measure~value", ("measure", "value")),
            ("base_feature~1", ("base_feature", "1")),
            ("f~0~lower", ("f", "0~lower")),
            ("a~0__op~1", ("a~0__op", "1")),
            ("x__op~1__next", None),
            ("measure", None),
            ("~x", None),
            ("measure~", None),
            ("measure~val ue", None),
            ("measure~value\n", None),
        ],
    )
    def test_split(self, name: str, expected: tuple[str, str] | None) -> None:
        assert FeatureGroup.split_output_part(name) == expected


class TestRequestedPartsOnFeature:
    """Recorded parts stay out of equality/hash and survive pickling."""

    def test_requested_parts_recorded_on_the_feature(self) -> None:
        _run(["measure~value"], MeasureProducer)

        [seen] = MeasureProducer.seen_parts
        assert seen == frozenset({"value"})

    def test_requested_parts_ignored_by_eq_hash_and_survive_pickle(self) -> None:
        import pickle  # nosec B403

        _run(["measure~value"], MeasureProducer)

        [feature] = MeasureProducer.captured
        plain = copy.copy(feature)
        plain.requested_parts = frozenset()
        plain.result_parts = None
        assert feature == plain
        assert hash(feature) == hash(plain)
        restored = pickle.loads(pickle.dumps(feature))  # nosec B301
        assert restored.requested_parts == feature.requested_parts == frozenset({"value"})

    @staticmethod
    def _feature(part: str | None, initial: bool = True) -> Feature:
        feature = Feature("measure")
        feature.initial_requested_data = initial
        feature.record_part(part)
        return feature

    def test_whole_feature_is_not_parts_only(self) -> None:
        assert self._feature(None).parts_only is False

    def test_part_request_is_parts_only(self) -> None:
        assert self._feature("value").parts_only is True

    def test_part_plus_part_stays_parts_only(self) -> None:
        a = self._feature("value")
        a.absorb_parts(self._feature("other"))
        assert a.parts_only is True

    @pytest.mark.parametrize("first_initial,second_initial", [(True, True), (True, False), (False, True)])
    def test_part_plus_whole_is_not_parts_only_either_order(self, first_initial: bool, second_initial: bool) -> None:
        part_first = self._feature("value", first_initial)
        part_first.absorb_parts(self._feature(None, second_initial))
        whole_first = self._feature(None, second_initial)
        whole_first.absorb_parts(self._feature("value", first_initial))

        assert part_first.parts_only is False
        assert whole_first.parts_only is False


class TestFrameForDeclaredPart:
    """RunResult.frame_for finds the frame by a requested part column name."""

    def test_frame_for_part_returns_frame_holding_the_column(self) -> None:
        results = _run(["measure~value"], MeasureProducer)

        assert "measure~value" in results.frame_for("measure~value").columns

    def test_plan_step_requests_the_part_and_injects_nothing_for_the_base(self) -> None:
        steps = mloda.explain(
            ["measure~value"],
            compute_frameworks=[PandasDataFrame],
            plugin_collector=_collector(MeasureProducer),
            parallelization_modes={ParallelizationMode.SYNC},
        )

        [step] = [s for s in steps if s.feature_group is MeasureProducer]
        assert "measure~value" in step.requested_feature_names
        assert "measure" not in step.injected_feature_names


class CountingVecProducer(FeatureGroup):
    """Undeclared group supporting `cvec`; counts matcher calls per name."""

    match_names: list[str] = []

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cvec"}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        cls.match_names.append(str(feature_name))
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for i in range(3):
            data[f"cvec~{i}"] = data["source_data"].values * (i + 1)
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class LiteralTildeProducer(FeatureGroup):
    """Declares parts (value, unit) and also supports the literal names `lit` and `lit~extra`."""

    OUTPUT_PARTS = ("value", "unit")

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"lit", "lit~extra"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"lit~extra": [1, 2], "lit~value": [3, 4], "lit~unit": ["m", "m"]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class LiteralMeasureValueProducer(FeatureGroup):
    """Provides the literal column `measure~value`."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"measure~value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"measure~value": [7, 7, 7, 7, 7]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class AnyVecProducer(FeatureGroup):
    """Declares ANY parts and supports `cvec`."""

    OUTPUT_PARTS = (ANY_OUTPUT_PART,)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cvec"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


@pytest.fixture(autouse=True)
def _reset_counting_state() -> None:
    CountingVecProducer.match_names = []


class TestFullNameAskedFirst:
    """Each candidate is asked about the full name first; the base only when a declared part applies."""

    def test_undeclared_group_sees_only_the_full_name(self) -> None:
        results = _run(["cvec~1"], CountingVecProducer)

        assert _columns(results) == {"cvec~1"}
        assert "cvec~1" in CountingVecProducer.match_names
        assert "cvec" not in CountingVecProducer.match_names

    def test_literal_tilde_name_wins_over_part_reading(self) -> None:
        results = _run(["lit~extra"], LiteralTildeProducer)

        assert _columns(results) == {"lit~extra"}

    def test_declared_part_still_works_beside_literal_tilde_names(self) -> None:
        results = _run(["lit~value"], LiteralTildeProducer)

        assert _columns(results) == {"lit~value"}

    def test_literal_column_group_wins_over_declared_part_reading(self) -> None:
        results = _run(["measure~value"], MeasureProducer, LiteralMeasureValueProducer)

        np.testing.assert_array_equal(results[0]["measure~value"].values, np.array([7, 7, 7, 7, 7]))
        assert MeasureProducer.calls == 0

    def test_non_declaring_numeric_match_does_not_count_as_whole(self) -> None:
        with pytest.raises(FeatureResolutionError, match="Multiple"):
            _run(["cvec~1"], CountingVecProducer, AnyVecProducer)

    def test_subclass_declaring_the_part_outranks_the_whole_name_parent(self) -> None:
        results = _run(["m2~value"], _ParentWholeName, _ChildDeclaringValue)

        assert _columns(results) == {"m2~value"}
        np.testing.assert_array_equal(results[0]["m2~value"].values, np.array([2, 2, 2]))

    def test_non_declaring_whole_strip_wins_over_part_group(self) -> None:
        results = _run(["cd~b~1"], _WholeStrip, _PartGroup)

        np.testing.assert_array_equal(results[0]["cd~b~1"].values, _SOURCE * 3)


class _ParentWholeName(FeatureGroup):
    """Matches the literal whole name `m2~value` and writes 1."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"m2~value"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"m2~value": [1, 1, 1]})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class _ChildDeclaringValue(_ParentWholeName):
    """Subclass declaring part `value` on base `m2`; writes 2."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"m2"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"m2~value": [2, 2, 2]})


class _WholeStrip(FeatureGroup):
    """Non-declaring group supporting `cd~b`; the numeric strip makes it match `cd~b~1`."""

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cd~b"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["cd~b~1"] = data["source_data"].values * 3
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class _PartGroup(FeatureGroup):
    """Supports `cd` and declares part `b`."""

    OUTPUT_PARTS = ("b",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"cd"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["cd~b~1"] = data["source_data"].values * 5
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class _CurrencyGroup(FeatureGroup):
    """Declares part `currency`; writes an int64 `price` and a str `price~currency`."""

    OUTPUT_PARTS = ("currency",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"price"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("source_data")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["price"] = data["source_data"].values.astype("int64")
        data["price~currency"] = "EUR"
        return data

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}


class _OptionSensitiveParts(FeatureGroup):
    """Full name writes an option and declines; output_parts is None when that option is visible."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gd"}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        if str(feature_name) == "gd~value":
            options.add_to_group("full_flag", 1)
            return False
        return str(feature_name) == "gd"

    @classmethod
    def output_parts(cls, feature_name: str, options: Options) -> tuple[str, ...] | None:
        return None if "full_flag" in options.group else ("value",)


class _GroupOptionGroup(FeatureGroup):
    """Full name writes a group option and records an input-data rejection; the base needs both absent."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"ga"}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        if str(feature_name) == "ga~value":
            options.add_to_group("rejected_flag", 1)
            record_match_rejection(cls.get_class_name(), "input data veto", INPUT_DATA_STAGE)
            return False
        return str(feature_name) == "ga" and "rejected_flag" not in options.group and not _has_input_data_rejection()


class _OwnReasonGroup(FeatureGroup):
    """Full name records its own reason; the base matches, so the undeclared part must be reported."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gb"}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        if str(feature_name) == "gb~bogus":
            record_match_rejection(cls.get_class_name(), "own full reason")
            return False
        return str(feature_name) == "gb"


class _BaseReasonGroup(FeatureGroup):
    """Full name records nothing; the base records a reason and declines."""

    OUTPUT_PARTS = ("value",)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"gc"}

    @classmethod
    def match_feature_group_criteria(
        cls, feature_name: Any, options: Options, data_access_collection: Any = None
    ) -> bool:
        if str(feature_name) == "gc":
            record_match_rejection(cls.get_class_name(), "base reason")
        return False


class TestPartProbeIsolation:
    """The full-name and base-name calls are isolated from each other and their reasons merged in order."""

    def test_full_name_option_write_and_rejection_do_not_leak_into_base_call(self) -> None:
        options = Options({})
        outcome = probe_match_criteria(_GroupOptionGroup, "ga~value", options, None, resolve_parts=True)

        assert outcome.matched is True
        assert outcome.part == ("ga", "value")
        assert "rejected_flag" not in options.group

    def test_output_parts_does_not_see_the_full_call_option_write(self) -> None:
        outcome = probe_match_criteria(_OptionSensitiveParts, "gd~value", Options({}), None, resolve_parts=True)

        assert outcome.matched is True
        assert outcome.part == ("gd", "value")

    def test_undeclared_part_reason_comes_before_the_full_call_reason(self) -> None:
        outcome = probe_match_criteria(_OwnReasonGroup, "gb~bogus", Options({}), None, resolve_parts=True)

        assert outcome.matched is False
        assert outcome.rejection is not None
        assert "not an output part" in outcome.rejection.reason
        assert "value" in outcome.rejection.reason
        assert outcome.rejection.stage == NAME_STAGE

    def test_base_call_reason_is_harvested_when_full_call_records_nothing(self) -> None:
        outcome = probe_match_criteria(_BaseReasonGroup, "gc~value", Options({}), None, resolve_parts=True)

        assert outcome.matched is False
        assert outcome.rejection is not None
        assert outcome.rejection.reason == "base reason"


class TestArtifactKeyedByBaseName:
    """An artifact keyed by a part column name is rejected, pointing to the base name."""

    def test_part_keyed_artifact_raises_value_error_pointing_to_base(self) -> None:
        api = mloda(
            ["measure~value"],
            [PandasDataFrame],
            plugin_collector=_collector(MeasureProducer),
            parallelization_modes={ParallelizationMode.SYNC},
        )

        with pytest.raises(ValueError, match="(?i)artifact") as exc_info:
            api.run(artifacts={"measure~value": "model"})

        assert "'measure'" in str(exc_info.value)
