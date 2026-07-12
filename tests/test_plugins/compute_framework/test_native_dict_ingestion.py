"""Regression pins for root ingestion in ``apply_compute_framework_transformer``.

These pass TODAY on this branch (root ingestion only applies a DIRECT transformer) and
must KEEP passing once descriptor-gated multi-hop materialization lands for FileSource:

  * Same-native-type input returns ``None`` (no spurious self-loop such as
    dict -> pa.Table -> dict discovered through the chain search).
  * A direct transformer still applies (pa.Table -> dict for PythonDict).
  * A plain dict is NEVER multi-hop routed through ``pa.Table``: only an input-data
    DESCRIPTOR (FileSource) may take the multi-hop chain. The framework's own native
    ``transform`` branch (e.g. ``pd.DataFrame.from_dict``) handles the dict instead.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa

from mloda.user import ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


def _pandas_fw() -> PandasDataFrame:
    return PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


def _python_dict_fw() -> PythonDictFramework:
    return PythonDictFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())


class TestSameNativeTypeReturnsNone:
    """Input already of the expected native type is not transformed (returns None)."""

    def test_pyarrow_table_input_into_pyarrow_framework_returns_none(self) -> None:
        fw = PyArrowTable(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert fw.apply_compute_framework_transformer(pa.table({"a": [1, 2]})) is None

    def test_dict_input_into_python_dict_framework_returns_none(self) -> None:
        fw = _python_dict_fw()
        assert fw.apply_compute_framework_transformer({"a": [1, 2]}) is None


class TestDirectTransformerStillApplies:
    def test_pyarrow_table_materializes_into_python_dict_via_direct_transformer(self) -> None:
        fw = _python_dict_fw()
        result = fw.apply_compute_framework_transformer(pa.table({"a": [1, 2]}))
        assert result == {"a": [1, 2]}


class TestPandasNativeDictIngestion:
    """A plain dict handed to PandasDataFrame uses ``pd.DataFrame.from_dict``, not pa.Table."""

    def test_mixed_type_dict_is_not_rerouted_through_pyarrow(self) -> None:
        """PandasDataFrame.transform must build a DataFrame natively for a mixed-type column.

        If the dict were rerouted through the ``dict -> pa.Table -> pandas`` chain, the
        first hop would raise ``pyarrow.ArrowInvalid`` ("Could not convert 'x' with type
        str") before the native ``from_dict`` branch runs.
        """
        fw = _pandas_fw()

        result = fw.transform({"a": [1, "x"]}, ["a"])

        assert isinstance(result, pd.DataFrame), f"Expected a pandas DataFrame, got {type(result)!r}"
        # Mixed int/str column stays object dtype (pandas' native behavior).
        assert str(result["a"].dtype) == "object"
        assert list(result["a"]) == [1, "x"]

    def test_apply_transformer_does_not_multihop_a_plain_dict(self) -> None:
        """The multi-hop (pa.Table) chain must not apply to a plain dict.

        ``apply_compute_framework_transformer`` must return ``None`` for a plain dict whose
        only registered path to pandas is the 2-hop ``dict -> pa.Table -> pandas`` chain, so
        the framework falls through to its native dict handling. Only an input-data
        DESCRIPTOR (FileSource) may be multi-hop materialized.
        """
        fw = _pandas_fw()

        assert fw.apply_compute_framework_transformer({"a": [1, "x"]}) is None


class TestPythonDictNativePassthrough:
    """PythonDict passthrough of a native columnar dict is unaffected by the seam (guard)."""

    def test_native_columnar_dict_passes_through_unchanged(self) -> None:
        fw = _python_dict_fw()
        data: dict[str, list[Any]] = {"a": [1, 2], "b": ["x", "y"]}

        result = fw.transform(data, ["a", "b"])

        assert result == {"a": [1, 2], "b": ["x", "y"]}
