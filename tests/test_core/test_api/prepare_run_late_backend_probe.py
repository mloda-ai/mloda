"""Exercise a prepared session before pytest's plugin loading can import the late backend."""

import sys
import threading
from typing import Any
from unittest.mock import patch

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import ComputeFrameworkTransformer
from mloda.provider import ApiDataFG, SourceMatch
from mloda.user import ParallelizationMode
from tests.test_core.test_api.test_prepare_run import TestConcurrentRunsOnOneSession


class LatePolarsApiDataFG(ApiDataFG):
    """Materialize API input with a backend imported after session preparation."""

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        import polars as pl

        return pl.DataFrame(cls.api_values(match))


def check(mode: str) -> None:
    assert "polars" not in sys.modules
    case = TestConcurrentRunsOnOneSession()
    session = case._prepare({ParallelizationMode[mode]}, joined=True, api_group=LatePolarsApiDataFG)
    assert "polars" not in sys.modules
    assert session.engine is not None
    registry = session.engine.transformer
    assert "polars" not in registry._seen_lazy_dirs
    previous_map = registry.transformer_map
    previous_edges = dict(previous_map)

    import polars as pl
    import pyarrow as pa

    assert "polars" not in registry._seen_lazy_dirs
    assert registry.transformer_map is previous_map
    assert (pl.DataFrame, pa.Table) not in previous_map
    refresh = ComputeFrameworkTransformer._refresh
    readers = threading.Barrier(2, timeout=5)
    refreshed: list[ComputeFrameworkTransformer] = []

    def concurrent_refresh(self: ComputeFrameworkTransformer) -> bool:
        assert self is registry
        assert "polars" in sys.modules
        refreshed.append(self)
        readers.wait()
        return refresh(self)

    with case._unchanged_plan(session), patch.object(ComputeFrameworkTransformer, "_refresh", concurrent_refresh):
        got = case._concurrent_results(session, "batch", joined=True, modes={ParallelizationMode[mode]}, readers=2)
        assert got == {i: case._expected(i, joined=True) for i in range(2)}

    assert len(refreshed) == 2
    assert session.engine.transformer is registry
    assert "polars" in registry._seen_lazy_dirs
    assert (pl.DataFrame, pa.Table) not in previous_map
    assert (pl.DataFrame, pa.Table) in registry.transformer_map
    assert registry.transformer_map is not previous_map
    assert previous_map == previous_edges


if __name__ == "__main__":
    check(sys.argv[1])
