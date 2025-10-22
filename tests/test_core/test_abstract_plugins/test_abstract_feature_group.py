from typing import Optional, Set, Union
import numpy as np
import pytest
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.options import Options


class BaseTestFeatureGroup1(AbstractFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "BaseTestFeature" in feature_name.name and "1" in feature_name.name:  # type: ignore
            return True
        return False


class BaseTestFeatureGroup2(AbstractFeatureGroup):
    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        if "BaseTestFeature" in feature_name.name and "2" in feature_name.name:  # type: ignore
            return True
        return False

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """This function should return the input features for the feature group
        if this feature is dependent on other features.
        Else, return None"""
        return {Feature.str_of("BaseTestFeature1"), Feature.int32_of("BaseTestFeature1")}

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> Optional[DataType]:
        if "BaseTestFeature" in feature.name and "2" in feature.name:
            return DataType.STRING
        return None


def test_apply_naming_convention_basic_multi_column() -> None:
    """Test that apply_naming_convention handles basic multi-column arrays correctly.

    Given a 2D numpy array with 3 columns and a feature name,
    the method should return a dictionary mapping column names to column data
    using the pattern: feature_name~0, feature_name~1, feature_name~2
    """
    # Arrange: Create a 2D array with 3 columns (simulating sklearn encoder output)
    result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    feature_name = "test_feature"

    # Act: Apply naming convention
    output = AbstractFeatureGroup.apply_naming_convention(result, feature_name)

    # Assert: Verify the output structure
    assert isinstance(output, dict), "Output should be a dictionary"
    assert len(output) == 3, "Should have 3 entries for 3 columns"

    # Verify column names follow the pattern
    assert "test_feature~0" in output
    assert "test_feature~1" in output
    assert "test_feature~2" in output

    # Verify column data is correct
    np.testing.assert_array_equal(output["test_feature~0"], np.array([1.0, 4.0, 7.0]))
    np.testing.assert_array_equal(output["test_feature~1"], np.array([2.0, 5.0, 8.0]))
    np.testing.assert_array_equal(output["test_feature~2"], np.array([3.0, 6.0, 9.0]))


@pytest.mark.parametrize(
    "column_name,expected_base_feature",
    [
        ("category~1", "category"),  # Basic case: strip suffix
        ("plain_name", "plain_name"),  # Edge case: no suffix
        ("feature~0", "feature"),  # Zero-indexed suffix
        ("complex_name~99", "complex_name"),  # Multi-digit suffix
    ],
)
def test_get_column_base_feature_strips_suffix(column_name: str, expected_base_feature: str) -> None:
    """Test that get_column_base_feature correctly extracts base feature name.

    The method should:
    - Strip the ~N suffix from column names (e.g., "category~1" -> "category")
    - Return the original name if no ~N suffix exists (e.g., "plain_name" -> "plain_name")
    """
    # Act: Extract base feature name
    result = AbstractFeatureGroup.get_column_base_feature(column_name)

    # Assert: Verify correct base feature name
    assert result == expected_base_feature, f"Expected {expected_base_feature}, got {result}"


def test_expand_feature_columns_generates_list() -> None:
    """Test that expand_feature_columns generates list of column names with suffixes.

    Given a feature name and the number of columns to generate,
    the method should return a list of column names following the pattern:
    ["feature~0", "feature~1", "feature~2", ...]
    """
    # Arrange: Feature name and number of columns
    feature_name = "category"
    num_columns = 3

    # Act: Generate column names
    result = AbstractFeatureGroup.expand_feature_columns(feature_name, num_columns)

    # Assert: Verify the output structure and content
    assert isinstance(result, list), "Output should be a list"
    assert len(result) == 3, "Should have 3 column names"
    assert result == ["category~0", "category~1", "category~2"], "Column names should follow the ~N suffix pattern"


def test_apply_naming_convention_with_custom_suffix_generator() -> None:
    """Test that apply_naming_convention uses custom suffix_generator when provided.

    When a suffix_generator callable is provided, it should be used instead of the
    default ~0, ~1, ~2 pattern. This is useful for custom naming schemes like
    dimensionality reduction where suffixes like ~dim1, ~dim2, ~dim3 are preferred.

    The suffix_generator takes a column index (int) and returns a suffix string.
    """
    # Arrange: Create a 2D array with 3 columns
    result = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    feature_name = "feature"

    # Custom suffix generator: produces "dim1", "dim2", "dim3" instead of "0", "1", "2"
    custom_suffix_generator = lambda i: f"dim{i + 1}"

    # Act: Apply naming convention with custom suffix generator
    output = AbstractFeatureGroup.apply_naming_convention(
        result, feature_name, suffix_generator=custom_suffix_generator
    )

    # Assert: Verify the output uses custom suffixes
    assert isinstance(output, dict), "Output should be a dictionary"
    assert len(output) == 3, "Should have 3 entries for 3 columns"

    # Verify column names use custom suffix pattern
    assert "feature~dim1" in output, "Should use custom suffix 'dim1'"
    assert "feature~dim2" in output, "Should use custom suffix 'dim2'"
    assert "feature~dim3" in output, "Should use custom suffix 'dim3'"

    # Verify column data is correct
    np.testing.assert_array_equal(output["feature~dim1"], np.array([1.0, 4.0, 7.0]))
    np.testing.assert_array_equal(output["feature~dim2"], np.array([2.0, 5.0, 8.0]))
    np.testing.assert_array_equal(output["feature~dim3"], np.array([3.0, 6.0, 9.0]))


def test_resolve_multi_column_feature_discovers_tilde_columns() -> None:
    """Test that resolve_multi_column_feature discovers multi-column features with ~ pattern.

    Given a feature name and a set of available columns that match the multi-column pattern,
    the method should return all matching columns sorted by their suffix index.

    For example, if "my_feature~0", "my_feature~1", "my_feature~2" exist in the columns,
    resolve_multi_column_feature("my_feature", columns) should return all three columns.
    """
    # Arrange: Set of available columns containing multi-column feature
    feature_name = "my_feature"
    available_columns = {"my_feature~0", "my_feature~1", "my_feature~2"}

    # Act: Resolve multi-column feature
    result = AbstractFeatureGroup.resolve_multi_column_feature(feature_name, available_columns)

    # Assert: Should return all matching columns sorted
    assert isinstance(result, list), "Output should be a list"
    assert len(result) == 3, "Should find all 3 columns"
    assert result == ["my_feature~0", "my_feature~1", "my_feature~2"], "Should return columns in sorted order"
