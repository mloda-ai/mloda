import pyarrow as pa
import pyarrow.compute as pc
import pytest
from typing import List, Set, Type, Union

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.api.request import mlodaAPI

from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable
from mloda_plugins.feature_group.experimental.data_quality.missing_value.pyarrow import PyArrowMissingValueFeatureGroup


@pytest.fixture
def sample_table_with_missing() -> pa.Table:
    """Create a sample PyArrow Table with missing values for testing."""
    return pa.Table.from_pydict(
        {
            "income": [50000, None, 75000, None, 60000],
            "age": [30, 25, None, 45, None],
            "category": ["A", None, "B", "A", None],
            "temperature": [72.5, 68.3, None, None, 70.1],
            "group": ["X", "Y", "X", "Y", "X"],
        }
    )


@pytest.fixture
def feature_set_mean() -> FeatureSet:
    """Create a feature set with a mean imputation feature."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed_income"))
    return feature_set


@pytest.fixture
def feature_set_multiple() -> FeatureSet:
    """Create a feature set with multiple imputation features."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed_income"))
    feature_set.add(Feature("median_imputed_age"))
    feature_set.add(Feature("mode_imputed_category"))
    feature_set.add(Feature("ffill_imputed_temperature"))
    return feature_set


@pytest.fixture
def feature_set_constant() -> FeatureSet:
    """Create a feature set with a constant imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("constant_imputed_category"))
    feature_set.options = Options({"constant_value": "Unknown"})
    return feature_set


@pytest.fixture
def feature_set_grouped() -> FeatureSet:
    """Create a feature set with a grouped imputation feature and options."""
    feature_set = FeatureSet()
    feature_set.add(Feature("mean_imputed_income"))
    feature_set.options = Options({"group_by_features": ["group"]})
    return feature_set


class TestPyArrowMissingValueFeatureGroup:
    """Tests for the PyArrowMissingValueFeatureGroup class."""

    def test_compute_framework_rule(self) -> None:
        """Test compute_framework_rule method."""
        assert PyArrowMissingValueFeatureGroup.compute_framework_rule() == {PyarrowTable}

    def test_perform_imputation_mean(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with mean imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "mean", "income")
        # Mean of [50000, NaN, 75000, NaN, 60000] = 61666.67
        # Check that missing values are imputed
        assert not pc.is_null(result[1]).as_py()
        assert not pc.is_null(result[3]).as_py()
        # Original values should be preserved
        assert result[0].as_py() == 50000
        assert result[2].as_py() == 75000
        assert result[4].as_py() == 60000

    def test_perform_imputation_median(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with median imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "median", "income")
        # Median of [50000, NaN, 75000, NaN, 60000] = 60000
        # Check that missing values are imputed
        assert not pc.is_null(result[1]).as_py()
        assert not pc.is_null(result[3]).as_py()
        # Original values should be preserved
        assert result[0].as_py() == 50000
        assert result[2].as_py() == 75000
        assert result[4].as_py() == 60000

    def test_perform_imputation_mode(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with mode imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "mode", "category")
        # Mode of ["A", None, "B", "A", None] = "A"
        # Check that missing values are imputed
        assert not pc.is_null(result[1]).as_py()
        assert not pc.is_null(result[4]).as_py()
        # Original values should be preserved
        assert result[0].as_py() == "A"
        assert result[2].as_py() == "B"
        assert result[3].as_py() == "A"

    def test_perform_imputation_constant(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with constant imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(
            sample_table_with_missing, "constant", "category", constant_value="Unknown"
        )
        # Constant imputation with "Unknown"
        assert result[1].as_py() == "Unknown"
        assert result[4].as_py() == "Unknown"
        # Original values should be preserved
        assert result[0].as_py() == "A"
        assert result[2].as_py() == "B"
        assert result[3].as_py() == "A"

    def test_perform_imputation_ffill(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with forward fill imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "ffill", "temperature")
        # Forward fill [72.5, 68.3, None, None, 70.1]
        assert not pc.is_null(result[2]).as_py()  # Should be filled
        assert not pc.is_null(result[3]).as_py()  # Should be filled
        # Original values should be preserved
        assert result[0].as_py() == 72.5
        assert result[1].as_py() == 68.3
        assert result[4].as_py() == 70.1

    def test_perform_imputation_bfill(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with backward fill imputation."""
        result = PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "bfill", "temperature")
        # Backward fill [72.5, 68.3, None, None, 70.1]
        assert not pc.is_null(result[2]).as_py()  # Should be filled
        assert not pc.is_null(result[3]).as_py()  # Should be filled
        # Original values should be preserved
        assert result[0].as_py() == 72.5
        assert result[1].as_py() == 68.3
        assert result[4].as_py() == 70.1

    def test_perform_imputation_invalid(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_imputation method with invalid imputation type."""
        with pytest.raises(ValueError):
            PyArrowMissingValueFeatureGroup._perform_imputation(sample_table_with_missing, "invalid", "income")

    def test_perform_grouped_imputation_mean(self, sample_table_with_missing: pa.Table) -> None:
        """Test _perform_grouped_imputation method with mean imputation by group."""
        result = PyArrowMissingValueFeatureGroup._perform_grouped_imputation(
            sample_table_with_missing, "mean", "income", None, ["group"]
        )
        # Check that original values are preserved
        assert result[0].as_py() == 50000  # Original value in group X
        assert result[2].as_py() == 75000  # Original value in group X
        assert result[4].as_py() == 60000  # Original value in group X

        # Check that missing values are imputed
        assert not pc.is_null(result[1]).as_py()  # Should be imputed
        assert not pc.is_null(result[3]).as_py()  # Should be imputed

    def test_calculate_feature_single(self, sample_table_with_missing: pa.Table, feature_set_mean: FeatureSet) -> None:
        """Test calculate_feature method with a single imputation."""
        result = PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set_mean)

        # Check that the result contains the original data plus the imputed feature
        assert "mean_imputed_income" in result.schema.names
        # Check that missing values are imputed
        assert not pc.is_null(result["mean_imputed_income"][1]).as_py()
        assert not pc.is_null(result["mean_imputed_income"][3]).as_py()

        # Check that the original data is preserved
        assert "income" in result.schema.names
        assert "age" in result.schema.names
        assert "category" in result.schema.names
        assert "temperature" in result.schema.names
        assert "group" in result.schema.names

    def test_calculate_feature_multiple(
        self, sample_table_with_missing: pa.Table, feature_set_multiple: FeatureSet
    ) -> None:
        """Test calculate_feature method with multiple imputations."""
        result = PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set_multiple)

        # Check that the result contains all imputed features
        assert "mean_imputed_income" in result.schema.names
        assert not pc.is_null(result["mean_imputed_income"][1]).as_py()
        assert not pc.is_null(result["mean_imputed_income"][3]).as_py()

        assert "median_imputed_age" in result.schema.names
        assert not pc.is_null(result["median_imputed_age"][2]).as_py()
        assert not pc.is_null(result["median_imputed_age"][4]).as_py()

        assert "mode_imputed_category" in result.schema.names
        assert not pc.is_null(result["mode_imputed_category"][1]).as_py()
        assert not pc.is_null(result["mode_imputed_category"][4]).as_py()

        assert "ffill_imputed_temperature" in result.schema.names
        assert not pc.is_null(result["ffill_imputed_temperature"][2]).as_py()
        assert not pc.is_null(result["ffill_imputed_temperature"][3]).as_py()

        # Check that the original data is preserved
        assert "income" in result.schema.names
        assert "age" in result.schema.names
        assert "category" in result.schema.names
        assert "temperature" in result.schema.names
        assert "group" in result.schema.names

    def test_calculate_feature_constant(
        self, sample_table_with_missing: pa.Table, feature_set_constant: FeatureSet
    ) -> None:
        """Test calculate_feature method with constant imputation."""
        result = PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set_constant)

        # Check that the result contains the imputed feature
        assert "constant_imputed_category" in result.schema.names
        assert result["constant_imputed_category"][1].as_py() == "Unknown"
        assert result["constant_imputed_category"][4].as_py() == "Unknown"

        # Check that the original data is preserved
        assert "income" in result.schema.names
        assert "age" in result.schema.names
        assert "category" in result.schema.names
        assert "temperature" in result.schema.names
        assert "group" in result.schema.names

    def test_calculate_feature_grouped(
        self, sample_table_with_missing: pa.Table, feature_set_grouped: FeatureSet
    ) -> None:
        """Test calculate_feature method with grouped imputation."""
        result = PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set_grouped)

        # Check that the result contains the imputed feature
        assert "mean_imputed_income" in result.schema.names

        # Check that original values are preserved
        assert result["mean_imputed_income"][0].as_py() == 50000  # Original value in group X
        assert result["mean_imputed_income"][2].as_py() == 75000  # Original value in group X
        assert result["mean_imputed_income"][4].as_py() == 60000  # Original value in group X

        # Check that missing values are imputed
        assert not pc.is_null(result["mean_imputed_income"][1]).as_py()  # Should be imputed
        assert not pc.is_null(result["mean_imputed_income"][3]).as_py()  # Should be imputed

        # Check that the original data is preserved
        assert "income" in result.schema.names
        assert "age" in result.schema.names
        assert "category" in result.schema.names
        assert "temperature" in result.schema.names
        assert "group" in result.schema.names

    def test_calculate_feature_missing_source(self, sample_table_with_missing: pa.Table) -> None:
        """Test calculate_feature method with missing source feature."""
        feature_set = FeatureSet()
        feature_set.add(Feature("mean_imputed_missing"))

        with pytest.raises(ValueError, match="Source feature 'missing' not found in data"):
            PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set)

    def test_calculate_feature_constant_without_value(self, sample_table_with_missing: pa.Table) -> None:
        """Test calculate_feature method with constant imputation but no constant value."""
        feature_set = FeatureSet()
        feature_set.add(Feature("constant_imputed_category"))

        with pytest.raises(ValueError, match="Constant value must be provided for constant imputation method"):
            PyArrowMissingValueFeatureGroup.calculate_feature(sample_table_with_missing, feature_set)


class TestMissingValuePyArrowIntegration:
    """Integration tests for the missing value feature group using DataCreator."""

    def test_imputation_with_data_creator(self) -> None:
        """Test imputation features with mlodaAPI using DataCreator."""

        # Create a feature group that uses DataCreator to provide test data
        class TestDataCreator(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> BaseInputData:
                return DataCreator({"income", "age", "category", "temperature", "group"})

            @classmethod
            def calculate_feature(cls, data: pa.Table, features: FeatureSet) -> pa.Table:
                # Return the test data
                return pa.Table.from_pydict(
                    {
                        "income": [50000, None, 75000, None, 60000],
                        "age": [30, 25, None, 45, None],
                        "category": ["A", None, "B", "A", None],
                        "temperature": [72.5, 68.3, None, None, 70.1],
                        "group": ["X", "Y", "X", "Y", "X"],
                    }
                )

            @classmethod
            def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
                return {PyarrowTable}

        # Enable the necessary feature groups
        plugin_collector = PlugInCollector.enabled_feature_groups({TestDataCreator, PyArrowMissingValueFeatureGroup})

        options = Options({"constant_value": "Unknown"})
        feature_str = [
            "income",  # Source data with missing values
            "age",
            "category",
            "temperature",
            "group",
            "mean_imputed_income",  # Mean imputation
            "median_imputed_age",  # Median imputation
            "mode_imputed_category",  # Mode imputation
            "constant_imputed_category",  # Constant imputation
            "ffill_imputed_temperature",  # Forward fill imputation
        ]

        feature_list: List[str | Feature] = [Feature(name=feature, options=options) for feature in feature_str]

        # Run the API with multiple imputation features
        result = mlodaAPI.run_all(
            feature_list,
            compute_frameworks={PyarrowTable},
            plugin_collector=plugin_collector,
        )

        # Verify the results
        assert len(result) == 2  # Two Tables: one for source data, one for imputed features

        # Find the Table with the imputed features
        imputed_table = None
        for table in result:
            if "mean_imputed_income" in table.schema.names:
                imputed_table = table
                break

        assert imputed_table is not None, "Table with imputed features not found"

        # Verify the imputed features
        assert "mean_imputed_income" in imputed_table.schema.names
        assert not pc.is_null(imputed_table["mean_imputed_income"][1]).as_py()
        assert not pc.is_null(imputed_table["mean_imputed_income"][3]).as_py()

        assert "median_imputed_age" in imputed_table.schema.names
        assert not pc.is_null(imputed_table["median_imputed_age"][2]).as_py()
        assert not pc.is_null(imputed_table["median_imputed_age"][4]).as_py()

        assert "mode_imputed_category" in imputed_table.schema.names
        assert not pc.is_null(imputed_table["mode_imputed_category"][1]).as_py()
        assert not pc.is_null(imputed_table["mode_imputed_category"][4]).as_py()

        assert "constant_imputed_category" in imputed_table.schema.names
        assert imputed_table["constant_imputed_category"][1].as_py() == "Unknown"
        assert imputed_table["constant_imputed_category"][4].as_py() == "Unknown"

        assert "ffill_imputed_temperature" in imputed_table.schema.names
        assert not pc.is_null(imputed_table["ffill_imputed_temperature"][2]).as_py()
        assert not pc.is_null(imputed_table["ffill_imputed_temperature"][3]).as_py()
