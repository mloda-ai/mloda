from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from mloda._testing.compute_framework.mask_suite import MaskEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PandasAdapter
from mloda._testing.typed_data import TypedTestData
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)


class TestPandasMaskEngine(MaskEngineSuite):
    adapter_class = PandasAdapter

    def frame(self, adapter: FrameworkAdapter, name: str, data: TypedTestData) -> Any:
        """Strings use pandas' default inference as in production; NaN/null and decimal survive via ArrowDtype."""
        frame = super().frame(adapter, name, data)
        columns = data.to_python_dict()
        for column, data_type in data.types.items():
            if data_type == DataType.STRING:
                frame[column] = pd.Series(columns[column], index=frame.index)
        if name == "null":
            frame["ratio"] = pd.Series(pd.arrays.ArrowExtensionArray(pa.array(columns["ratio"], type=pa.float64())))
        if name == "decimal":
            frame["d"] = pd.Series(columns["d"], dtype=pd.ArrowDtype(pa.decimal128(10, 2)))
        return frame

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        return bool(mask.dtype == bool)

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        return data[mask]

    @pytest.mark.parametrize(
        "values",
        [np.array(["active"]), pd.Series(["active"])],
        ids=["numpy_array", "pandas_series"],
    )
    def test_is_in_rejects_numpy_and_series_values(self, sample_data: Any, values: Any) -> None:
        with pytest.raises(TypeError, match="list, tuple, set or frozenset"):
            PandasMaskEngine.is_in(sample_data, "status", values)
