"""RED tests: a plain dict must use a framework's NATIVE ingestion, not be rerouted
through ``pa.Table`` by the multi-hop transformer chain.

Central bug (with pyarrow INSTALLED): ``apply_compute_framework_transformer`` builds a
2-hop chain ``dict -> pa.Table -> <framework>`` for any dict input. That hijacks the
framework's own native dict handling and, for mixed-type columns, raises
``pyarrow.ArrowInvalid`` before the native path is ever reached.

Target design: a multi-hop chain (through ``pa.Table``) may only be used when the input
is an input-data DESCRIPTOR marker (``FileSource``). A plain ``dict`` is not a descriptor,
so ``apply_compute_framework_transformer`` must NOT reroute it; the framework's own
``transform`` then handles the dict natively.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


def _pandas_fw() -> PandasDataFrame:
    return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


def _python_dict_fw() -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


class TestPandasNativeDictIngestion:
    """A plain dict handed to PandasDataFrame uses ``pd.DataFrame.from_dict``, not pa.Table."""

    def test_mixed_type_dict_is_not_rerouted_through_pyarrow(self) -> None:
        """PandasDataFrame.transform must build a DataFrame natively for a mixed-type column.

        Fails today: ``apply_compute_framework_transformer`` runs the
        ``dict -> pa.Table -> pandas`` chain first; ``dict -> pa.Table`` raises
        ``pyarrow.ArrowInvalid`` ("Could not convert 'x' with type str: tried to convert
        to int64") before the native ``from_dict`` branch runs.
        """
        fw = _pandas_fw()

        result = fw.transform({"a": [1, "x"]}, {"a"})

        assert isinstance(result, pd.DataFrame), f"Expected a pandas DataFrame, got {type(result)!r}"
        # Mixed int/str column stays object dtype (pandas' native behavior).
        assert str(result["a"].dtype) == "object"
        assert list(result["a"]) == [1, "x"]

    def test_apply_transformer_does_not_multihop_a_plain_dict(self) -> None:
        """The multi-hop (pa.Table) chain must not apply to a plain dict.

        ``apply_compute_framework_transformer`` must return ``None`` for a plain dict whose
        only registered path to pandas is the 2-hop ``dict -> pa.Table -> pandas`` chain, so
        the framework falls through to its native dict handling.

        Fails today: it executes the chain and raises ``pyarrow.ArrowInvalid`` instead of
        returning ``None``.
        """
        fw = _pandas_fw()

        assert fw.apply_compute_framework_transformer({"a": [1, "x"]}) is None


class TestPythonDictNativePassthrough:
    """PythonDict passthrough of a native columnar dict is unaffected by the fix (guard)."""

    def test_native_columnar_dict_passes_through_unchanged(self) -> None:
        fw = _python_dict_fw()
        data: dict[str, list[Any]] = {"a": [1, 2], "b": ["x", "y"]}

        result = fw.transform(data, {"a", "b"})

        assert result == {"a": [1, 2], "b": ["x", "y"]}
