from typing import Any

from mloda.user import ParallelizationMode
import pytest
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from mloda._testing.compute_framework.dtype_suite import DataTypeValidatorSuite, DtypeExtractionSuite
from mloda._testing.declarations import Cause
from mloda._testing.framework_adapter import FrameworkAdapter, PythonDictAdapter, to_arrow_table
from mloda._testing.typed_data import TypedTestData


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


class TestPythonDictDtypeExtractionSuite(DtypeExtractionSuite):
    adapter_class = PythonDictAdapter
    switched_off = {
        "test_extract_duplicate_column_dtype_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence": Cause.BACKEND,
        "test_extract_column_dtype_consistent_with_output_schema": Cause.BACKEND,
    }

    @pytest.mark.skip(reason="A Python dict cannot hold duplicate column names")
    def test_extract_duplicate_column_dtype_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="A Python dict cannot hold duplicate column names")
    def test_extract_duplicate_column_data_type_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="A Python dict cannot hold duplicate column names")
    def test_extract_duplicate_column_dtype_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="A Python dict cannot hold duplicate column names")
    def test_extract_duplicate_column_data_type_reversed_order_returns_first_occurrence(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...

    @pytest.mark.skip(reason="A Python dict cannot hold duplicate column names")
    def test_extract_column_dtype_consistent_with_output_schema(
        self, adapter: FrameworkAdapter, framework_instance: Any
    ) -> None: ...


_NO_INT_WIDTH = (Cause.BACKEND, "Python int has no width")


class TestPythonDictDataTypeValidatorSuite(DataTypeValidatorSuite):
    """Python types carry no width: precision-narrowing tests cannot be enforced from runtime types."""

    adapter_class = PythonDictAdapter
    unsupported_cases = {
        "int8_col": _NO_INT_WIDTH,
        "uint8_col": _NO_INT_WIDTH,
        "uint64_col": _NO_INT_WIDTH,
        "int16_col": _NO_INT_WIDTH,
        "uint16_col": _NO_INT_WIDTH,
        "uint32_col": _NO_INT_WIDTH,
    }
    switched_off = {
        "test_int32_column_strict_int32_passes": Cause.BACKEND,
        "test_int64_column_strict_int32_raises": Cause.BACKEND,
        "test_float64_column_strict_float_raises": Cause.BACKEND,
        "test_float32_column_strict_float_passes": Cause.BACKEND,
        "test_float32_column_strict_double_passes": Cause.BACKEND,
        "test_timestamp_ms_column_strict_ms_passes": Cause.BACKEND,
        "test_timestamp_us_column_strict_ms_raises": Cause.BACKEND,
    }

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        """Go through Arrow so nanosecond timestamps arrive as pandas Timestamps, as they do in a real run."""
        return adapter.from_arrow(to_arrow_table(data, self.arrow_types(name)))

    @pytest.mark.skip(reason="Python type.__name__ collapses int widths; the int32 column reports INT64, not INT32")
    def test_int32_column_strict_int32_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python type.__name__ collapses int widths; INT32 vs INT64 cannot be distinguished")
    def test_int64_column_strict_int32_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python float is always 64-bit; FLOAT vs DOUBLE cannot be distinguished")
    def test_float64_column_strict_float_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python float is always 64-bit; the column reports DOUBLE, not FLOAT")
    def test_float32_column_strict_float_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python float is always 64-bit; the column would already be DOUBLE-typed")
    def test_float32_column_strict_double_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python datetime is always microsecond precision; TIMESTAMP_MILLIS cannot be expressed")
    def test_timestamp_ms_column_strict_ms_passes(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...

    @pytest.mark.skip(reason="Python datetime is always microsecond precision; TIMESTAMP precision uniform")
    def test_timestamp_us_column_strict_ms_raises(self, adapter: FrameworkAdapter, framework_instance: Any) -> None: ...
