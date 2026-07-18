"""Unit tests for the PyArrowFilterEngine class."""

from typing import Any

import pytest
import pyarrow as pa

from mloda.user import Feature
from mloda.user import SingleFilter
from mloda.user import FilterType
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_filter_engine import PyArrowFilterEngine

from tests.test_plugins.compute_framework.base_implementations.filter_engine_test_mixin import (
    FilterEngineTestMixin,
)


class TestPyArrowFilterEngine(FilterEngineTestMixin):
    """Unit tests for the PyArrowFilterEngine class using shared mixin."""

    @pytest.fixture
    def filter_engine(self) -> Any:
        """Return the PyArrowFilterEngine class."""
        return PyArrowFilterEngine

    @pytest.fixture
    def sample_data(self) -> Any:
        """Create a sample PyArrow table for testing."""
        return pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "age": [25, 30, 35, 40, 45],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "category": ["A", "B", "A", "C", "B"],
            }
        )

    @pytest.fixture
    def nullable_category_sample_data(self) -> Any:
        """Create a sample PyArrow table with null categories for testing."""
        return pa.Table.from_pydict({"id": [1, 2, 3, 4, 5], "category": ["A", None, "B", None, "C"]})

    def get_column_values(self, result: Any, column: str) -> list[Any]:
        """Extract column values from PyArrow table."""
        return result[column].to_pylist()  # type: ignore[no-any-return]

    def test_categorical_inclusion_on_dictionary_encoded_column(self, filter_engine: Any) -> None:
        """CATEGORICAL_INCLUSION must work on dictionary-encoded (categorical) columns.

        ``pa.Table.from_pandas`` on a pandas ``category`` dtype column produces a
        ``dictionary<large_string, int8>`` arrow column (same as dictionary-encoded
        parquet). Categorical inclusion is exactly the filter meant for such columns,
        so it must not crash. The engine builds its value-set with
        ``pa.array(values, type=data[column_name].type)``; when the column type is a
        dictionary type this raises ``ArrowNotImplementedError`` (DictionaryArray
        converter ... not implemented).
        """
        import pandas as pd

        pdf = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5], "category": pd.Series(["A", None, "B", None, "C"], dtype="category")}
        )
        table = pa.Table.from_pandas(pdf, preserve_index=False)

        # Document intent: the category column is dictionary-encoded.
        assert pa.types.is_dictionary(table["category"].type)

        cases: list[tuple[dict[str, Any], list[int]]] = [
            ({"values": ["A", None]}, [1, 2, 4]),  # "A" (id 1) plus nulls (ids 2, 4)
            ({"values": ["A"]}, [1]),  # only "A"; nulls dropped
            ({"values": []}, []),  # empty allowed-values -> empty result
            ({"values": [None]}, [2, 4]),  # only nulls kept
        ]

        for parameter, expected_ids in cases:
            single_filter = SingleFilter(Feature("category"), FilterType.CATEGORICAL_INCLUSION, parameter)

            result = filter_engine.do_categorical_inclusion_filter(table, single_filter)

            assert sorted(result["id"].to_pylist()) == expected_ids
