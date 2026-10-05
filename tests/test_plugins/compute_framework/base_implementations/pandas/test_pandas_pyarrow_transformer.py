from decimal import Decimal

import numpy as np
import pytest

from mloda_plugins.compute_framework.base_implementations.pandas.pandas_pyarrow_transformer import (
    PandasPyArrowTransformer,
)

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


@pytest.mark.skipif(pd is None or pa is None, reason="Pandas or PyArrow is not installed. Skipping this test.")
class TestPandasPyArrowTransformerDecimal:
    def test_object_decimal_column_infers_precision(self) -> None:
        df = pd.DataFrame({"d": pd.Series([Decimal("12.34"), Decimal("5.50"), Decimal("99.99")], dtype=object)})

        table = PandasPyArrowTransformer.transform_fw_to_other_fw(df)

        assert table.schema.field("d").type == pa.decimal128(4, 2)

    def test_arrow_decimal_keeps_arrow_dtype(self) -> None:
        values = [Decimal("12.34"), Decimal("5.50")]
        table = pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))})

        df = PandasPyArrowTransformer.transform_other_fw_to_fw(table)

        assert df["d"].dtype == pd.ArrowDtype(pa.decimal128(10, 2))
        assert list(df["d"]) == values
        assert PandasPyArrowTransformer.transform_fw_to_other_fw(df).schema.field("d").type == pa.decimal128(10, 2)

    def test_non_decimal_columns_keep_numpy_dtypes(self) -> None:
        expected_string_dtype = pa.table({"s": ["a"]}).to_pandas()["s"].dtype
        values = [Decimal("12.34"), Decimal("5.50")]
        table = pa.table(
            {
                "i": pa.array([1, 2], type=pa.int64()),
                "f": pa.array([1.5, 2.5], type=pa.float64()),
                "s": pa.array(["a", "b"], type=pa.string()),
                "d": pa.array(values, type=pa.decimal128(10, 2)),
            }
        )

        df = PandasPyArrowTransformer.transform_other_fw_to_fw(table)

        assert df["i"].dtype == np.dtype("int64")
        assert df["f"].dtype == np.dtype("float64")
        assert df["s"].dtype == expected_string_dtype
