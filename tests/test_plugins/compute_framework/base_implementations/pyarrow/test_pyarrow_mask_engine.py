from typing import Any

import pyarrow as pa

from mloda._testing.compute_framework.mask_suite import MaskEngineSuite
from mloda._testing.framework_adapter import FrameworkAdapter, PyArrowAdapter


class TestPyArrowMaskEngine(MaskEngineSuite):
    adapter_class = PyArrowAdapter

    def evaluate_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> list[bool]:
        return mask.to_pylist()  # type: ignore[no-any-return]

    def is_boolean_mask(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> bool:
        return bool(mask.type == pa.bool_())

    def select_rows(self, adapter: FrameworkAdapter, mask: Any, data: Any) -> Any:
        return data.filter(mask)
