from typing import Any

from mloda.core.abstract_plugins.components.mask.base_mask_engine import BaseMaskEngine
from mloda.core.abstract_plugins.components.mask.null_or_nan import is_null_or_nan, split_null_or_nan
from mloda.core.abstract_plugins.components.utils import require_value_collection
from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_value_set import value_set

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]
    pc = None


def _no_null(mask: Any) -> Any:
    return pc.fill_null(mask, False)


class PyArrowMaskEngine(BaseMaskEngine):
    @classmethod
    def supported_data_type(cls) -> type[Any]:
        return pa.Table  # type: ignore[no-any-return]

    @classmethod
    def all_true(cls, data: Any) -> Any:
        return pa.array([True] * data.num_rows, type=pa.bool_())

    @classmethod
    def combine(cls, mask1: Any, mask2: Any) -> Any:
        return pc.and_(mask1, mask2)

    @classmethod
    def equal(cls, data: Any, column: str, value: Any) -> Any:
        if is_null_or_nan(value):
            return pc.is_null(data[column], nan_is_null=True)
        return _no_null(pc.equal(data[column], value))

    @classmethod
    def greater_equal(cls, data: Any, column: str, value: Any) -> Any:
        return _no_null(pc.greater_equal(data[column], value))

    @classmethod
    def less_equal(cls, data: Any, column: str, value: Any) -> Any:
        return _no_null(pc.less_equal(data[column], value))

    @classmethod
    def less_than(cls, data: Any, column: str, value: Any) -> Any:
        return _no_null(pc.less(data[column], value))

    @classmethod
    def greater_than(cls, data: Any, column: str, value: Any) -> Any:
        return _no_null(pc.greater(data[column], value))

    @classmethod
    def is_in(cls, data: Any, column: str, values: Any) -> Any:
        require_value_collection(values, "is_in values")
        present, has_null_or_nan = split_null_or_nan(values)
        mask = pc.is_in(data[column], value_set(data[column], present))
        if has_null_or_nan:
            mask = pc.or_(mask, pc.is_null(data[column], nan_is_null=True))
        return mask
