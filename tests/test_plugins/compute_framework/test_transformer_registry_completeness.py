"""RED test: the transformer registry must be complete regardless of import order.

The two ``FileSource`` transformers currently register ONLY via
``ComputeFrameworkTransformer.initilize_transformer``'s auto-load glob, which fires only
when ``get_all_subclasses(BaseTransformer)`` is EMPTY. If any ``BaseTransformer`` subclass
is imported at module scope BEFORE the first ``ComputeFrameworkTransformer()`` construction,
the auto-load is skipped and the registry is left INCOMPLETE (e.g. the
``(FileSource, pa.Table)`` edge is missing, because no production module imports
``FileSourcePyArrowTransformer``).

This module deliberately imports one transformer subclass at MODULE scope to "poison" the
registry, then asserts the registry is still complete. It FAILS today (in isolation)
because the auto-load is skipped and several core pairs are absent.
"""

from __future__ import annotations

import pyarrow as pa
import pandas as pd

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource

# Module-scope import of a BaseTransformer subclass: this is what poisons the registry.
# After this import, ``get_all_subclasses(BaseTransformer)`` is non-empty, so the auto-load
# glob in ``initilize_transformer`` never runs and the FileSource transformers are lost.
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
