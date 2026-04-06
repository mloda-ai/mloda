from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.core.filter.filter_mask_engine import BaseFilterMaskEngine


class PyArrowFilterMaskEngine(BaseFilterMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return pa.Table  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return pa.array([True] * data.num_rows)

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        return pc.and_(mask1, mask2)

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        return pc.equal(data[column], value)

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return pc.greater_equal(data[column], value)

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return pc.less_equal(data[column], value)

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return pc.less(data[column], value)

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        return pc.is_in(data[column], pa.array(values))
