from decimal import Decimal

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

    def test_arrow_decimal_comes_back_as_object_dtype(self) -> None:
        values = [Decimal("12.34"), Decimal("5.50")]
        table = pa.table({"d": pa.array(values, type=pa.decimal128(10, 2))})

        df = PandasPyArrowTransformer.transform_other_fw_to_fw(table)

        assert df["d"].dtype == object
        assert list(df["d"]) == values
