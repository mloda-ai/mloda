"""RED test: a ``FileSource`` descriptor with no resolvable materialization chain must
raise an ACTIONABLE error, not silently leak the descriptor.

This test targets a dummy, unmaterializable data-framework type that has no registered
transformer. A CSV ``FileSource`` therefore cannot be materialized into it via any chain
(neither directly nor through ``pa.Table``), so the "no chain" condition holds regardless
of whether pyarrow is installed. Because it does not depend on pyarrow, pandas, duckdb, or
any optional framework, and runs in-process (no subprocess), it is portable across all tox
envs, including the genuine no-pyarrow env.

The fix in ``ComputeFramework._materialize_descriptor`` must raise a ``ValueError`` whose
message names the ``FileSource`` descriptor and points at installing pyarrow (substring
``pyarrow``), instead of leaking the ``FileSource`` object downstream.
"""

from __future__ import annotations

import pytest

from mloda.user import ParallelizationMode
from mloda.provider import ComputeFramework
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource


def test_file_source_without_chain_raises_actionable_pyarrow_error() -> None:
    # A dummy target type with no registered transformer: FileSource cannot be materialized
    # into it via any chain (direct or through pa.Table), so the actionable error must fire
    # whether or not pyarrow is installed. Both classes are defined locally so the throwaway
    # ComputeFramework subclass is not discovered by other tests' framework enumeration.
    class _Unmaterializable:
        pass

    class _NoChainFramework(ComputeFramework):
        @classmethod
        def expected_data_framework(cls) -> type:
            return _Unmaterializable

    fw = _NoChainFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
    source = FileSource(path="/nonexistent.csv", format="csv", columns=("A", "B"))

    with pytest.raises(ValueError) as excinfo:
        fw.transform(source, {"A", "B"})

    message = str(excinfo.value)
    assert "pyarrow" in message.lower(), message
    assert "FileSource" in message, message
