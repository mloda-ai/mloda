"""
Unit tests for the PandasEncoding class.
"""

from typing import Any
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from tests.test_plugins.feature_group.experimental.core_input_check import assert_core_rejects_missing_input
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.sklearn.encoding.base import EncodingFeatureGroup
from mloda_plugins.feature_group.experimental.sklearn.encoding.pandas import PandasEncoding
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class TestPandasEncoding:
    """Test cases for the PandasEncoding class."""

    def test_compute_framework(self) -> None:
        assert PandasEncoding.compute_framework is PandasDataFrame

    @pytest.mark.parametrize(
        ("name", "missing", "available"),
        [
            ("nonexistent__onehot_encoded", ("nonexistent",), ("category", "value")),
            ("nonexistent~1__onehot_encoded", ("nonexistent",), ("category", "value")),
        ],
        ids=["missing", "missing_with_part_suffix"],
    )
    def test_core_rejects_missing_input_and_lists_sorted_columns(
        self, name: str, missing: tuple[str, ...], available: tuple[str, ...]
    ) -> None:
        data = pd.DataFrame({"value": [1, 2, 3], "category": ["A", "B", "C"]})

        assert_core_rejects_missing_input(PandasEncoding, data, name, missing, available)

    def test_existing_sources_calculate_the_feature(self) -> None:
        pytest.importorskip("sklearn")
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})
        features = FeatureSet()
        features.add(Feature("category__label_encoded"))

        result = PandasEncoding.calculate(EncodingFeatureGroup, data, features, None)

        assert "category__label_encoded" in result.columns

    def test_extract_training_data_basic(self) -> None:
        """Test extracting training data from pandas DataFrame."""
        data = pd.DataFrame({"category": ["A", "B", "C", "A", "B"], "value": [1, 2, 3, 4, 5]})

        training_data = PandasEncoding._extract_training_data(data, "category")

        # Should return 1D array - base class handles reshaping based on encoder type
        assert training_data.shape == (5,)
        assert list(training_data) == ["A", "B", "C", "A", "B"]

    def test_extract_training_data_with_missing_values(self) -> None:
        """Test extracting training data with missing values."""
        data = pd.DataFrame({"category": ["A", "B", None, "A", "B"], "value": [1, 2, 3, 4, 5]})

        training_data = PandasEncoding._extract_training_data(data, "category")

        # Should fill NaN with "unknown" during training, matching the transform-time policy
        assert training_data.shape == (5,)
        assert list(training_data) == ["A", "B", "unknown", "A", "B"]

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_apply_encoder_label_encoder(self, mock_import: Any) -> None:
        """Test applying LabelEncoder to pandas DataFrame."""
        # Mock sklearn components
        mock_label_encoder = Mock()
        mock_label_encoder.__class__.__name__ = "LabelEncoder"
        mock_label_encoder.transform.return_value = np.array([0, 1, 2, 0, 1])

        data = pd.DataFrame({"category": ["A", "B", "C", "A", "B"], "value": [1, 2, 3, 4, 5]})

        result = PandasEncoding._apply_encoder(data, "category", mock_label_encoder)

        # Verify transform was called with correct data
        mock_label_encoder.transform.assert_called_once()
        call_args = mock_label_encoder.transform.call_args[0][0]
        assert list(call_args) == ["A", "B", "C", "A", "B"]

        # Check result
        assert list(result) == [0, 1, 2, 0, 1]

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_apply_encoder_onehot_encoder(self, mock_import: Any) -> None:
        """Test applying OneHotEncoder to pandas DataFrame."""
        # Mock sklearn components
        mock_onehot_encoder = Mock()
        mock_onehot_encoder.__class__.__name__ = "OneHotEncoder"
        # OneHotEncoder returns 2D array
        mock_onehot_encoder.transform.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

        data = pd.DataFrame({"category": ["A", "B", "C", "A", "B"], "value": [1, 2, 3, 4, 5]})

        result = PandasEncoding._apply_encoder(data, "category", mock_onehot_encoder)

        # Verify transform was called with correct data (2D)
        mock_onehot_encoder.transform.assert_called_once()
        call_args = mock_onehot_encoder.transform.call_args[0][0]
        assert call_args.shape == (5, 1)
        assert list(call_args.flatten()) == ["A", "B", "C", "A", "B"]

        # Check result shape
        assert result.shape == (5, 3)

    def test_apply_encoder_with_missing_values(self) -> None:
        """Test applying encoder with missing values in data."""
        # Mock encoder
        mock_encoder = Mock()
        mock_encoder.__class__.__name__ = "LabelEncoder"
        mock_encoder.transform.return_value = np.array([0, 1, 2, 0, 1])

        data = pd.DataFrame({"category": ["A", "B", None, "A", "B"], "value": [1, 2, 3, 4, 5]})

        _result = PandasEncoding._apply_encoder(data, "category", mock_encoder)

        # Verify transform was called with filled data
        mock_encoder.transform.assert_called_once()
        call_args = mock_encoder.transform.call_args[0][0]
        # Missing values should be filled with "unknown"
        assert list(call_args) == ["A", "B", "unknown", "A", "B"]

    def test_add_result_to_data_label_encoder(self) -> None:
        """Test adding LabelEncoder results to DataFrame."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})
        result = np.array([0, 1, 2])

        updated_data = PandasEncoding._add_result_to_data(data, "category__label_encoded", result)

        # Check that new column was added
        assert "category__label_encoded" in updated_data.columns
        assert list(updated_data["category__label_encoded"]) == [0, 1, 2]

        # Original data should be preserved
        assert list(updated_data["category"]) == ["A", "B", "C"]
        assert list(updated_data["value"]) == [1, 2, 3]

    def test_add_result_to_data_onehot_encoder(self) -> None:
        """Test adding OneHotEncoder results to DataFrame."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})
        # OneHotEncoder returns 2D array with multiple columns
        result = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        updated_data = PandasEncoding._add_result_to_data(data, "category__onehot_encoded", result)

        # Check that multiple columns were added with ~ separator
        assert "category__onehot_encoded~0" in updated_data.columns
        assert "category__onehot_encoded~1" in updated_data.columns
        assert "category__onehot_encoded~2" in updated_data.columns

        # Check values
        assert list(updated_data["category__onehot_encoded~0"]) == [1, 0, 0]
        assert list(updated_data["category__onehot_encoded~1"]) == [0, 1, 0]
        assert list(updated_data["category__onehot_encoded~2"]) == [0, 0, 1]

        # Original data should be preserved
        assert list(updated_data["category"]) == ["A", "B", "C"]
        assert list(updated_data["value"]) == [1, 2, 3]

    def test_add_result_to_data_onehot_encoder_single_column(self) -> None:
        """Test adding OneHotEncoder results when only one column is returned."""
        data = pd.DataFrame({"category": ["A", "A", "A"], "value": [1, 2, 3]})
        # OneHotEncoder with only one category
        result = np.array([[1], [1], [1]])

        updated_data = PandasEncoding._add_result_to_data(data, "category__onehot_encoded", result)

        # Should add single column without ~ separator
        assert "category__onehot_encoded" in updated_data.columns
        assert list(updated_data["category__onehot_encoded"]) == [1, 1, 1]

    def test_add_result_to_data_ordinal_encoder(self) -> None:
        """Test adding OrdinalEncoder results to DataFrame."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})
        # OrdinalEncoder returns 2D array but single column
        result = np.array([[0], [1], [2]])

        updated_data = PandasEncoding._add_result_to_data(data, "category__ordinal_encoded", result)

        # Check that new column was added (flattened from 2D)
        assert "category__ordinal_encoded" in updated_data.columns
        assert list(updated_data["category__ordinal_encoded"]) == [0, 1, 2]

        # Original data should be preserved
        assert list(updated_data["category"]) == ["A", "B", "C"]
        assert list(updated_data["value"]) == [1, 2, 3]

    def test_add_result_to_data_sparse_matrix(self) -> None:
        """Test adding results from sparse matrix (OneHotEncoder default)."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        # Mock sparse matrix
        mock_sparse_result = Mock()
        mock_sparse_result.toarray.return_value = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        updated_data = PandasEncoding._add_result_to_data(data, "category__onehot_encoded", mock_sparse_result)

        # Verify toarray was called
        mock_sparse_result.toarray.assert_called_once()

        # Check that multiple columns were added
        assert "category__onehot_encoded~0" in updated_data.columns
        assert "category__onehot_encoded~1" in updated_data.columns
        assert "category__onehot_encoded~2" in updated_data.columns

    def test_add_result_to_data_does_not_parse_a_part_suffix(self) -> None:
        """The writer no longer selects a column from `~N`; core selects the part. See TestEncodingOutputParts."""
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})
        result = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        updated_data = PandasEncoding._add_result_to_data(data, "category__onehot_encoded~1", result)

        assert "category__onehot_encoded~1~0" in updated_data.columns
        assert "category__onehot_encoded~1~2" in updated_data.columns

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_end_to_end_label_encoding(self, mock_import: Any) -> None:
        """Test end-to-end label encoding workflow."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return  # Skip test if sklearn not available

        # Mock sklearn components
        mock_import.return_value = {"LabelEncoder": LabelEncoder}

        # Create test data
        data = pd.DataFrame({"category": ["A", "B", "C", "A", "B"], "value": [1, 2, 3, 4, 5]})

        # Create feature set
        features = FeatureSet()
        features.add(Feature("category__label_encoded"))

        # Execute feature calculation
        result_data = PandasEncoding.calculate(EncodingFeatureGroup, data, features, None)

        # Check that encoded column was added
        assert "category__label_encoded" in result_data.columns
        # LabelEncoder should assign 0, 1, 2 to A, B, C respectively
        expected_values = [0, 1, 2, 0, 1]  # A=0, B=1, C=2
        assert list(result_data["category__label_encoded"]) == expected_values

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_end_to_end_label_encoding_with_nan_does_not_crash(self, mock_import: Any) -> None:
        """Fit fills NaN with 'unknown' (same as transform), so LabelEncoder.transform never sees an unseen label."""
        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return  # Skip test if sklearn not available

        mock_import.return_value = {"LabelEncoder": LabelEncoder}

        data = pd.DataFrame({"category": ["a", "b", None, "a", "b"]})

        features = FeatureSet()
        features.add(Feature("category__label_encoded"))

        result_data = PandasEncoding.calculate(EncodingFeatureGroup, data, features, None)

        assert "category__label_encoded" in result_data.columns

        # Ground truth: fit/transform a fresh LabelEncoder on the same fillna'd data.
        filled = data["category"].fillna("unknown")
        expected = LabelEncoder().fit(filled).transform(filled)

        assert list(result_data["category__label_encoded"]) == list(expected)

        # The row that had None must have a defined, non-null integer code.
        nan_row_code = result_data["category__label_encoded"].iloc[2]
        assert pd.notna(nan_row_code)
        assert 0 <= nan_row_code < len(set(filled))

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_end_to_end_label_encoding_with_numeric_nan_does_not_crash(self, mock_import: Any) -> None:
        """FIT's fillna("unknown") on a numeric column mixes floats and a string, crashing LabelEncoder.fit."""
        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            return  # Skip test if sklearn not available

        mock_import.return_value = {"LabelEncoder": LabelEncoder}

        data = pd.DataFrame({"category": [1.0, 2.0, None, 1.0]})

        features = FeatureSet()
        features.add(Feature("category__label_encoded"))

        result_data = PandasEncoding.calculate(EncodingFeatureGroup, data, features, None)

        assert "category__label_encoded" in result_data.columns

    @patch(
        "mloda_plugins.feature_group.experimental.sklearn.encoding.base.EncodingFeatureGroup._import_sklearn_components"
    )
    def test_end_to_end_onehot_encoding(self, mock_import: Any) -> None:
        """Test end-to-end one-hot encoding workflow."""
        # Skip test if sklearn not available
        try:
            from sklearn.preprocessing import OneHotEncoder
        except ImportError:
            return  # Skip test if sklearn not available

        # Mock sklearn components
        mock_import.return_value = {"OneHotEncoder": OneHotEncoder}

        # Create test data
        data = pd.DataFrame({"category": ["A", "B", "C"], "value": [1, 2, 3]})

        # Create feature set
        features = FeatureSet()
        features.add(Feature("category__onehot_encoded"))

        # Execute feature calculation
        result_data = PandasEncoding.calculate(EncodingFeatureGroup, data, features, None)

        # Check that multiple encoded columns were added
        onehot_columns = [col for col in result_data.columns if col.startswith("category__onehot_encoded~")]
        assert len(onehot_columns) == 3  # Should have 3 categories

        # Each row should have exactly one 1 and two 0s
        for i in range(len(data)):
            row_values = [result_data[col].iloc[i] for col in onehot_columns]
            assert sum(row_values) == 1  # Exactly one 1
            assert all(val in [0, 1] for val in row_values)  # Only 0s and 1s
