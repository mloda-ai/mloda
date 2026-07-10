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
        assert PythonDictFramework.expected_data_framework() is dict

    def test_transform_columnar_dict_passthrough(self) -> None:
        """A columnar dict with equal-length lists is returned as-is."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        input_data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}

        result = framework.transform(input_data, [])
        assert result == {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}

    def test_transform_single_dict_raises(self) -> None:
        """A single row dict (non-list values) is invalid columnar input and raises."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError):
            framework.transform({"col1": 1, "col2": "a"}, [])

    def test_transform_list_of_rows_becomes_columnar(self) -> None:
        """A list of row dicts pivots into a columnar dict."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        input_data = [{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]

        result = framework.transform(input_data, [])
        assert result == {"col1": [1, 2], "col2": ["a", "b"]}

    def test_transform_empty_data_returns_empty_dict(self) -> None:
        """The representational empties [] and {} normalize to the schema-less {}.
        None is NOT an empty: it raises, pinned by ``test_transform_none_raises``."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        assert framework.transform([], []) == {}
        assert framework.transform({}, []) == {}

    def test_transform_none_raises(self) -> None:
        """``None`` is a MISSING result (state A), not a representational empty.

        Every schema-bearing framework rejects ``None`` in ``transform``; PythonDict must
        do the same instead of silently converting it to ``[]``. Only ``[]`` and ``{}``
        are PythonDict's representational empties; ``None`` must reach the
        unsupported-data tail and raise.
        """
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform(None, [])

    def test_transform_invalid_data(self) -> None:
        """Test that invalid data types raise errors."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform("invalid", [])

    @pytest.mark.parametrize("falsy_value", [0, False, ""])
    def test_transform_falsy_unsupported_data_raises(self, falsy_value: Any) -> None:
        """Falsy values of unsupported types are rejected, not swallowed as empty.

        The empty short-circuit applies only to [] and {}. Other falsy
        values (0, False, "") must reach the unsupported-data tail and raise.
        """
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError, match="Data type .* is not supported"):
            framework.transform(falsy_value, [])

    def test_select_data_by_column_names(self) -> None:
        """Test columnar column selection functionality."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        data: dict[str, list[Any]] = {"col1": [1, 2], "col2": ["a", "b"], "col3": [10, 20]}

        feature_names = [FeatureName("col1"), FeatureName("col2")]

        result = framework.select_data_by_column_names(data, feature_names)
        assert result == {"col1": [1, 2], "col2": ["a", "b"]}

    def test_select_data_by_column_names_does_not_alias_internal_state(self) -> None:
        """Mutating the selection result must not mutate the framework's stored data.

        ``select_data_by_column_names`` currently returns a dict whose column lists are
        the SAME list objects held in ``framework.data``, so appending to the result
        corrupts the framework's internal state. The selection must not alias them.
        """
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = {"a": [1, 2], "b": [3, 4]}

        result = framework.select_data_by_column_names(framework.data, [FeatureName("a")])
        assert result == {"a": [1, 2]}

        result["a"].append(99)

        assert framework.data["a"] == [1, 2]

    def test_select_data_empty_returns_empty_dict(self) -> None:
        """Empty data ({}) is propagated as {} in column selection (no judgement)."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        feature_names = [FeatureName("col1")]

        assert framework.select_data_by_column_names({}, feature_names) == {}

    def test_set_column_names(self) -> None:
        """Test setting column names from columnar data."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = {"col1": [1, 2], "col2": ["a", "b"], "col3": [None, 10]}

        framework.set_column_names()

        expected_columns = {"col1", "col2", "col3"}
        assert framework.column_names == expected_columns

    def test_set_column_names_empty_data(self) -> None:
        """Test that the schema-less empty ({}) produces an empty column set."""
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        framework.data = {}
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


class TestFrameworkColumnarValidationAgreesWithHelper:
    def test_validate_native_data_and_validate_columnar_dict_agree_on_ragged_dict(self) -> None:
        """The framework's columnar validation and the utils validator reject the same ragged dict."""
        from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import (
            validate_columnar_dict,
        )

        ragged: dict[str, Any] = {"a": [1, 2], "b": [1]}
        framework = PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

        with pytest.raises(ValueError):
            framework.validate_native_data(ragged)
        with pytest.raises(ValueError):
            validate_columnar_dict(ragged)


class TestPythonDictDtypeExtraction(DtypeExtractionTestMixin):
    """Test PythonDictFramework._extract_column_dtype using shared mixin."""

    @pytest.fixture
    def framework_instance(self) -> Any:
        return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())

    @pytest.fixture
    def dtype_sample_data(self) -> Any:
        return {
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.0, 2.0, 3.0],
        }


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
        return table.to_pydict()

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
