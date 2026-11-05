from typing import Any

from mloda._testing.compute_framework.mask_suite import MaskEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PolarsAdapter

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


class TestPolarsMaskEngine(MaskEngineSuite):
    adapter_class = PolarsAdapter

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        return list(mask)

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        return bool(mask.dtype == pl.Boolean)

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        return data.filter(mask)
