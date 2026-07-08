import datetime
import decimal
from collections.abc import Sequence
from typing import Any, Optional
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.provider import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_merge_engine import (
    PythonDictMergeEngine,
)
from mloda.user import FeatureName
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine, BaseMaskEngine
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_filter_engine import (
    PythonDictFilterEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_mask_engine import (
    PythonDictMaskEngine,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import rows_to_columnar


class PythonDictFramework(ComputeFramework):
    """
    PythonDict Compute Framework

    Uses a COLUMNAR ``dict[str, list[Any]]`` as the data structure for tabular data.
    This framework provides a simple, dependency-free implementation using
    native Python data structures.

    Data Structure:
        dict[str, list[Any]] - Each key is a column name, each value is that column's
        list of cell values. All value-lists share one length (= the row count). The
        schema is the set of keys, present even at zero rows.

    Example:
        {
            "col1": [1, 2],
            "col2": ["a", "b"],
        }

    ``{"a": []}`` is a valid schema-bearing zero-row frame (one known column). ``{}`` is
    the only schema-less value (zero columns).
    """

    @classmethod
    def expected_data_framework(cls) -> Any:
        return dict

    @classmethod
    def merge_engine(cls) -> type[BaseMergeEngine]:
        return PythonDictMergeEngine

    def select_data_by_column_names(
        self,
        data: dict[str, list[Any]],
        selected_feature_names: Sequence[FeatureName],
        column_ordering: Optional[str] = None,
    ) -> dict[str, list[Any]]:
        if not data:
            return {}

        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, set(data.keys()), ordering=column_ordering
        )

        # Copy the column lists so mutating the selection cannot mutate framework internals.
        return {k: list(data[k]) for k in _selected_feature_names if k in data}

    def _extract_column_names(self, data: Any) -> set[str]:
        if isinstance(data, dict):
            return set(data.keys())
        # Row-wise list[dict] (still an accepted pre-transform shape): columns are the keys.
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return set(data[0].keys())
        return set()

    def _is_schemaless_empty(self, data: Any) -> bool:
        """Only the empty dict ``{}`` (zero columns) is schema-less. A zero-row but
        column-bearing frame such as ``{"a": []}`` carries a schema and is NOT schema-less.
        """
        return isinstance(data, dict) and not data

    @staticmethod
    def _column_values(data: Any, column_name: str) -> list[Any]:
        """Return the values of ``column_name`` for either columnar dict or row-wise list[dict]."""
        if isinstance(data, list):
            return [row.get(column_name) for row in data if isinstance(row, dict)]
        return data.get(column_name, [])  # type: ignore[no-any-return]

    def _extract_column_dtype(self, data: Any, column_name: str) -> str | None:
        for value in self._column_values(data, column_name):
            if value is not None:
                return type(value).__name__
        return None

    def _extract_column_data_type(self, data: Any, column_name: str) -> Optional[DataType]:
        for val in self._column_values(data, column_name):
            if val is None:
                continue
            if isinstance(val, bool):
                return DataType.BOOLEAN
            if isinstance(val, int):
                return DataType.INT64
            if isinstance(val, float):
                return DataType.DOUBLE
            if isinstance(val, str):
                return DataType.STRING
            if isinstance(val, bytes):
                return DataType.BINARY
            if isinstance(val, datetime.datetime):
                return DataType.TIMESTAMP_MICROS
            if isinstance(val, datetime.date):
                return DataType.DATE
            if isinstance(val, decimal.Decimal):
                return DataType.DECIMAL
            return None
        return None

    @staticmethod
    def _validate_columnar_dict(data: dict[str, Any]) -> None:
        """Enforce the columnar contract on a dict: every value is a list and all value-lists
        share one length. ``{}`` is schema-less and always valid. Shared by ``transform`` and
        ``validate_native_data`` so both agree on what a valid columnar frame is.
        """
        if not data:
            return

        if not all(isinstance(v, list) for v in data.values()):
            raise ValueError(
                f"Columnar dict values must all be lists (rows must arrive as a list of dicts). Got: {data}"
            )

        lengths = {len(v) for v in data.values()}
        if len(lengths) > 1:
            raise ValueError(f"All columns must have the same length. Got column lengths {lengths}.")

    def validate_native_data(self, data: Any) -> None:
        """Enforce the columnar contract when a dict result bypasses ``transform``.

        A scalar/mixed dict such as ``{"a": 1}`` is the expected framework type (``dict``) and
        would otherwise be stored without validation. This raises the same ``ValueError``
        ``transform`` raises for a malformed columnar dict.
        """
        if isinstance(data, dict):
            self._validate_columnar_dict(data)

    def transform(self, data: Any, feature_names: Sequence[str]) -> dict[str, list[Any]]:
        """
        Transforms data to the COLUMNAR PythonDict framework format.

        Args:
            data: Input data to transform
            feature_names: Sequence of feature names being processed

        Returns:
            dict[str, list[Any]]: Data in columnar PythonDict format

        Raises:
            ValueError: If data type is not supported
        """

        # The representational empties [] and {} normalize to the schema-less {}. None is a
        # missing return value (state A), not an empty result, and falls through to the
        # unsupported-type rejection below.
        if isinstance(data, list) and not data:
            return {}
        if isinstance(data, dict) and not data:
            return {}

        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data  # type: ignore[no-any-return]

        if isinstance(data, dict):
            self._validate_columnar_dict(data)
            return data

        if isinstance(data, list):
            # List of row dicts -> pivot to columnar. Keys must be homogeneous across rows.
            return rows_to_columnar(data)

        raise ValueError(f"Data type {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def filter_engine(cls) -> type[BaseFilterEngine]:
        """
        Returns the filter engine for PythonDict framework.

        Returns:
            Type[BaseFilterEngine]: PythonDictFilterEngine class
        """
        return PythonDictFilterEngine

    @classmethod
    def mask_engine(cls) -> type[BaseMaskEngine]:
        return PythonDictMaskEngine
