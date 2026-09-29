"""Public ComputeFramework.extract_column_names delegates to the private _extract_column_names override point."""

from typing import Any

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


def _pandas_frame() -> Any:
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"a": [1], "b": [2]})


def test_pandas_subclass_public_method_follows_private_override() -> None:
    class ZzCustomColumnsPandasFramework(PandasDataFrame):
        @staticmethod
        def is_available() -> bool:
            return False

        def _extract_column_names(self, data: Any) -> set[str]:
            return {"custom"}

    framework = ZzCustomColumnsPandasFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    assert framework.extract_column_names(_pandas_frame()) == {"custom"}


def test_direct_subclass_public_method_returns_private_override() -> None:
    class ZzPrivateOnlyFramework(ComputeFramework):
        @staticmethod
        def is_available() -> bool:
            return False

        def _extract_column_names(self, data: Any) -> set[str]:
            return {str(key) for key in data}

    framework = ZzPrivateOnlyFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    assert framework.extract_column_names({"x": [1], "y": [2]}) == {"x", "y"}


def test_builtin_pandas_framework_returns_frame_columns() -> None:
    framework = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    assert framework.extract_column_names(_pandas_frame()) == {"a", "b"}
