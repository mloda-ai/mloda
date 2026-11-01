"""The transformer registry is complete regardless of import order.

The light transformer files auto-load once per process on first registry construction, and each
heavy backend's once its library is imported (refreshed on a lookup miss), so a stray module-scope import of a single
``BaseTransformer`` subclass before the first ``ComputeFrameworkTransformer()`` construction
cannot leave the registry partial (e.g. the ``(FileSource, pa.Table)`` edge missing, because
no production module imports ``FileSourcePyArrowTransformer``).

This module deliberately imports one transformer subclass at MODULE scope to "poison" the
registry, then asserts the registry is still complete.
"""

from __future__ import annotations

import copy
import pickle  # nosec B403
from typing import Any

import pandas as pd
import pyarrow as pa

from mloda.core.abstract_plugins.components.framework_transformer.base_transformer import BaseTransformer
from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
    _rebuild_transformer,
)
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

# Module-scope import of a BaseTransformer subclass: this is what poisons the registry.
# After this import, ``get_all_subclasses(BaseTransformer)`` is non-empty, so a load-only-
# when-empty glob in ``initilize_transformer`` never runs and other transformers are lost.
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_file_source_transformer import (  # noqa: F401,E501
    FileSourceDictTransformer,
)


def test_registry_is_complete_despite_module_scope_transformer_import() -> None:
    """Core transformer pairs must always be present with pyarrow installed."""
    transformer_map = ComputeFrameworkTransformer().transformer_map

    expected_pairs = {
        (FileSource, dict): "FileSource -> PythonDict (stdlib csv)",
        (FileSource, pa.Table): "FileSource -> pa.Table (pyarrow csv)",
        (dict, pa.Table): "PythonDict -> pa.Table",
        (pa.Table, dict): "pa.Table -> PythonDict",
        (pd.DataFrame, pa.Table): "pandas -> pa.Table",
        (pa.Table, pd.DataFrame): "pa.Table -> pandas",
    }

    missing = {pair for pair in expected_pairs if pair not in transformer_map}
    assert not missing, f"Registry incomplete after a module-scope transformer import; missing: {missing}"


def test_pickle_and_deepcopy_rebuild_the_registry_instead_of_shipping_it() -> None:
    t = ComputeFrameworkTransformer()

    for clone in (pickle.loads(pickle.dumps(t)), copy.deepcopy(t)):  # nosec B301
        assert clone is not t
        assert set(clone.transformer_map) == set(t.transformer_map)


def test_reduce_carries_only_transformers_outside_the_base_implementations() -> None:
    class _Left:
        pass

    class _Right:
        pass

    class _LocalTransformer(BaseTransformer):
        @classmethod
        def framework(cls) -> Any:
            return _Left

        @classmethod
        def other_framework(cls) -> Any:
            return _Right

        @classmethod
        def import_fw(cls) -> None:
            pass

        @classmethod
        def import_other_fw(cls) -> None:
            pass

        @classmethod
        def transform_fw_to_other_fw(cls, data: Any) -> Any:
            return data

        @classmethod
        def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
            return data

    rebuild, args = ComputeFrameworkTransformer().__reduce__()[:2]
    assert rebuild is _rebuild_transformer
    carried = args[0]

    assert _LocalTransformer in carried
    assert not [c for c in carried if c.__module__.startswith("mloda_plugins.compute_framework.base_implementations.")]
