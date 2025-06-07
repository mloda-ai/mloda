from copy import deepcopy
from typing import Any
import pytest
from unittest.mock import patch
import pyarrow as pa

from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyarrowTable


class TestPyarrowTableAvailability:
    @patch("builtins.__import__")
    def test_is_available_when_pyarrow_not_installed(self, mock_import: Any) -> None:
        """Test that is_available() returns False when pyarrow import fails."""

        def side_effect(name: Any, *args: Any, **kwargs: Any) -> Any:
            if name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect
        assert PyarrowTable.is_available() is False


class TestPyarrowTableComputeFramework:
    pyarrow_table = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
    dict_data = {"column1": [1, 2, 3], "column2": [4, 5, 6]}
    expected_data = pa.table(dict_data)
    left_data = pa.table({"idx": [1, 3], "col1": ["a", "b"]})
    right_data = pa.table({"idx": [1, 2], "col2": ["x", "z"]})
    idx = Index(("idx",))

    def test_expected_data_framework(self) -> None:
        assert self.pyarrow_table.expected_data_framework() == pa.Table

    def test_transform_dict_to_table(self) -> None:
        assert self.pyarrow_table.transform(self.dict_data, set()) == self.expected_data

    def test_transform_arrays(self) -> None:
        chunked_array = pa.chunked_array([pa.array([1, 2]), pa.array([3])])
        pa_array = pa.array([1, 2, 3])

        for data in [chunked_array, pa_array]:
            _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
            _pytable.set_data(pa.table({"existing_column": [4, 5, 6]}))

            data = _pytable.transform(data=data, feature_names={"new_column"})
            assert data.equals(pa.table({"existing_column": [4, 5, 6], "new_column": [1, 2, 3]}))

    def test_transform_invalid_data(self) -> None:
        with pytest.raises(ValueError):
            self.pyarrow_table.transform(data=["a"], feature_names=set())

    def test_select_data_by_column_names(self) -> None:
        data = self.pyarrow_table.select_data_by_column_names(self.expected_data, {FeatureName("column1")})
        assert data.schema.names == ["column1"]

    def test_set_column_names(self) -> None:
        self.pyarrow_table.data = self.expected_data
        self.pyarrow_table.set_column_names()
        assert self.pyarrow_table.column_names == {"column1", "column2"}

    def test_merge_inner(self) -> None:
        _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pytable.data = self.left_data
        merge_engine = _pytable.merge_engine()
        result = merge_engine().merge(_pytable.data, self.right_data, JoinType.INNER, self.idx, self.idx)
        assert len(result) == 1
        expected = self.left_data.join(self.right_data, keys="idx", join_type="inner")
        assert result.equals(expected)

    def test_merge_left(self) -> None:
        _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pytable.data = self.left_data
        merge_engine = _pytable.merge_engine()
        result = merge_engine().merge(_pytable.data, self.right_data, JoinType.LEFT, self.idx, self.idx)
        expected = self.left_data.join(self.right_data, keys="idx", join_type="left outer")
        assert result.equals(expected)

    def test_merge_right(self) -> None:
        _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pytable.data = self.left_data
        merge_engine = _pytable.merge_engine()
        result = merge_engine().merge(_pytable.data, self.right_data, JoinType.RIGHT, self.idx, self.idx)
        expected = self.left_data.join(self.right_data, keys="idx", join_type="right outer")
        expected = expected.sort_by("idx")
        result = result.sort_by("idx")
        assert expected.column("col2") == result.column("col2")
        assert expected.column("col1") == result.column("col1")
        assert result.equals(expected)

    def test_merge_full_outer(self) -> None:
        _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pytable.data = deepcopy(self.left_data)
        merge_engine = _pytable.merge_engine()
        result = merge_engine().merge(_pytable.data, self.right_data, JoinType.OUTER, self.idx, self.idx)
        expected = self.left_data.join(self.right_data, keys="idx", join_type="full outer")
        expected = expected.sort_by("idx")
        result = result.sort_by("idx")
        assert result.equals(expected)

    def test_merge_append(self) -> None:
        _pytable = PyarrowTable(mode=ParallelizationModes.SYNC, children_if_root=frozenset())
        _pytable.data = self.left_data
        merge_engine = _pytable.merge_engine()
        right_data = pa.table({"idx": [1, 2], "col1": ["x", "z"]})
        result = merge_engine().merge_append(_pytable.data, right_data, self.idx, self.idx)
        expected = pa.concat_tables([self.left_data, right_data])
        assert result.equals(expected)
