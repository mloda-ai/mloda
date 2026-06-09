from typing import Any

from mloda.user import ParallelizationMode
import pyarrow as pa
import pytest
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.test_plugins.compute_framework.base_implementations.datatype_validator_test_mixin import (
    DataTypeValidatorFrameworkTestMixin,
)
from tests.test_plugins.compute_framework.base_implementations.dtype_extraction_test_mixin import (
    DtypeExtractionTestMixin,
)


class TestPythonDictFramework:
    """Test suite for PythonDict compute framework."""

    def test_expected_data_framework(self) -> None:
        assert PythonDictFramework.expected_data_framework() is list

    def test_transform_dict_to_list(self) -> None:
        """Test transformation from columnar dict to list of dicts."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test columnar dict transformation
        input_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        expected = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}, {"col1": 3, "col2": "c"}]

        result = framework.transform(input_data, set())
        assert result == expected

    def test_transform_single_dict(self) -> None:
        """Test transformation from single dict to list of dicts."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test single dict transformation
        input_data = {"col1": 1, "col2": "a"}
        expected = [{"col1": 1, "col2": "a"}]

        result = framework.transform(input_data, set())
        assert result == expected

    def test_transform_list_passthrough(self) -> None:
        """Test that list of dicts passes through unchanged."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        # Test list passthrough
        input_data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        result = framework.transform(input_data, set())
        assert result == input_data

    def test_transform_empty_data_returns_empty_list(self) -> None:
        """Empty data is propagated as [] (the framework never judges emptiness)."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        assert framework.transform(None, set()) == []
        assert framework.transform([], set()) == []
        assert framework.transform({}, set()) == []

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise errors."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform("invalid", set())

    @pytest.mark.parametrize("falsy_value", [0, False, ""])
    def test_transform_falsy_unsupported_data_raises(self, falsy_value: Any) -> None:
        """Falsy values of unsupported types are rejected, not swallowed as empty.

        The empty short-circuit applies only to None, [] and {}. Other falsy
        values (0, False, "") must reach the unsupported-data tail and raise.
        """
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform(falsy_value, set())

    def test_select_data_by_column_names(self) -> None:
        """Test column selection functionality."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        data = [{"col1": 1, "col2": "a", "col3": 10}, {"col1": 2, "col2": "b", "col3": 20}]

        feature_names = {FeatureName("col1"), FeatureName("col2")}

        result = framework.select_data_by_column_names(data, feature_names)
        expected = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        assert result == expected

    def test_select_data_empty_returns_empty_list(self) -> None:
        """Empty data is propagated as [] in column selection (no judgement)."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        feature_names = {FeatureName("col1")}

        assert framework.select_data_by_column_names([], feature_names) == []

    def test_set_column_names(self) -> None:
        """Test setting column names from data."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b", "col3": 10}]

        framework.set_column_names()

        expected_columns = {"col1", "col2", "col3"}
        assert framework.column_names == expected_columns

    def test_set_column_names_empty_data(self) -> None:
        """Test that empty data produces an empty column set."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = []
        framework.set_column_names()
        assert framework.column_names == set()

    def test_merge_engine(self) -> None:
        """Test that merge engine returns correct type."""
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
            PythonDictMergeEngine,
        )

        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert framework.merge_engine() == PythonDictMergeEngine

    def test_filter_engine(self) -> None:
        """Test that filter engine returns correct type."""
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
            PythonDictFilterEngine,
        )

        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert framework.filter_engine() == PythonDictFilterEngine


class TestPythonDictDtypeExtraction(DtypeExtractionTestMixin):
    """Test PythonDictFramework._extract_column_dtype using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self) -> Any:
        return [
            {"int_col": 1, "str_col": "a", "float_col": 1.0},
            {"int_col": 2, "str_col": "b", "float_col": 2.0},
            {"int_col": 3, "str_col": "c", "float_col": 3.0},
        ]


class TestPythonDictDataTypeValidator(DataTypeValidatorFrameworkTestMixin):
    """Test DataTypeValidator enforcement on PythonDictFramework using shared mixin.

    Python types do not carry width information: ``type(int_val).__name__`` is just
    ``"int"`` regardless of value, and ``type(float_val).__name__`` is just ``"float"``
    (always 64-bit). ``datetime.datetime`` is always microsecond precision. Precision-
    narrowing tests that need per-width distinctions are skipped because they are
    statically un-enforceable from Python runtime types.
    """

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    def from_arrow(self, table: pa.Table) -> Any:
        return table.to_pylist()

    def test_int32_column_strict_int32_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python type.__name__ collapses int widths; the int32 column reports INT64, not INT32")

    def test_int64_column_strict_int32_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python type.__name__ collapses int widths; INT32 vs INT64 cannot be distinguished")

    def test_float64_column_strict_float_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python float is always 64-bit; FLOAT vs DOUBLE cannot be distinguished")

    def test_float32_column_strict_float_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python float is always 64-bit; the column reports DOUBLE, not FLOAT")

    def test_float32_column_strict_double_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python float is always 64-bit; the column would already be DOUBLE-typed")

    def test_timestamp_ms_column_strict_ms_passes(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python datetime is always microsecond precision; TIMESTAMP_MILLIS cannot be expressed")

    def test_timestamp_us_column_strict_ms_raises(self, framework_instance: Any, precision_sample_data: Any) -> None:
        pytest.skip("Python datetime is always microsecond precision; TIMESTAMP precision uniform")
