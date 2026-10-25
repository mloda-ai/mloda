"""Generic policy-conformance tooling for FeatureGroup-level policy flags.

Drives any pipeline result policy (such as the schema-based empty-result contract) end-to-end
through the public API ``mloda.run_all``, across every built-in compute framework and at
least one worker-based parallelization mode. A policy case is expressed as an expectation:
either ``PolicySuccess`` (run_all returns results that satisfy an assertion) or
``PolicyRaises`` (run_all raises an Exception whose message contains a substring).

This module is intentionally NOT collected as tests (no ``Test`` prefix). It is the shared
base that concrete per-framework conformance suites build on.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import GlobalFilter
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import RunResult
from mloda.user import mloda

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment, unused-ignore]


def records_from_frame(data: Any) -> list[dict[str, Any]]:
    """Extract native frame rows to plain Python row dicts across backends.

    Framework-agnostic: a zero-row result is simply ``len(records_from_frame(result)) == 0``.
    """
    records: list[dict[str, Any]]
    if pl is not None and isinstance(data, pl.LazyFrame):
        records = data.collect().to_dicts()
    elif pl is not None and isinstance(data, pl.DataFrame):
        records = data.to_dicts()
    elif pa is not None and isinstance(data, pa.Table):
        records = data.to_pylist()
    elif isinstance(data, dict):
        # PythonDict columnar frame: {"col": [v0, v1, ...]}. Zero columns -> zero rows.
        if not data:
            records = []
        else:
            length = len(next(iter(data.values())))
            records = [{k: data[k][i] for k in data} for i in range(length)]
    elif isinstance(data, list):
        records = data
    elif hasattr(data, "to_dicts"):
        records = data.to_dicts()
    elif hasattr(data, "to_dict"):
        records = data.to_dict("records")
    elif hasattr(data, "toPandas"):
        # Spark DataFrame
        records = [row.asDict() for row in data.collect()]
    else:
        records = data.df().to_dict("records")
    return records


@dataclass(frozen=True)
class PolicySuccess:
    """run_all returns results; ``assert_result`` is called on them."""

    assert_result: Callable[[RunResult], None]


@dataclass(frozen=True)
class PolicyRaises:
    """run_all raises an Exception whose ``str()`` contains ``match_substring`` (and ``also_contains``).

    ``error_type`` and ``cause_type`` optionally pin the raised type and its ``__cause__``.
    """

    match_substring: str
    error_type: type[BaseException] | None = None
    cause_type: type[BaseException] | None = None
    also_contains: tuple[str, ...] = ()


PolicyExpectation = PolicySuccess | PolicyRaises


class PolicyRunAllTestBase(ABC):
    """Drives a FeatureGroup-level policy end-to-end through ``run_all``.

    Subclasses implement ``compute_framework_name`` and, for connection-backed frameworks,
    ``get_connection`` (the connection travels in the DataAccessCollection).
    """

    @classmethod
    @abstractmethod
    def compute_framework_name(cls) -> str:
        """Return the compute framework name string for ``compute_frameworks=[...]``."""
        pass

    def get_connection(self) -> Any | None:
        """Return a framework connection object, or None when none is needed."""
        return None

    def _features_and_dac(self, feature_names: Sequence[str]) -> tuple[list[Feature], DataAccessCollection | None]:
        conn = self.get_connection()
        if conn is not None:
            features = [Feature(name=name) for name in feature_names]
            return features, DataAccessCollection(connections={conn})
        return [Feature(name=name) for name in feature_names], None

    def _feature_and_dac(self, feature_name: str) -> tuple[Feature, DataAccessCollection | None]:
        features, dac = self._features_and_dac([feature_name])
        return features[0], dac

    def assert_policy_case(
        self,
        *,
        feature_name: str | Sequence[str],
        plugin_collector: PluginCollector,
        expectation: PolicyExpectation,
        mode: ParallelizationMode,
        flight_server: Any,
        global_filter: GlobalFilter | None = None,
    ) -> RunResult | None:
        names = [feature_name] if isinstance(feature_name, str) else list(feature_name)
        features, dac = self._features_and_dac(names)

        if isinstance(expectation, PolicyRaises):
            with pytest.raises(Exception) as excinfo:
                mloda.run_all(
                    features,
                    compute_frameworks=[self.compute_framework_name()],
                    plugin_collector=plugin_collector,
                    parallelization_modes={mode},
                    flight_server=flight_server,
                    data_access_collection=dac,
                    global_filter=global_filter,
                )
            assert expectation.match_substring in str(excinfo.value)
            for substring in expectation.also_contains:
                assert substring in str(excinfo.value)
            if expectation.error_type is not None:
                assert isinstance(excinfo.value, expectation.error_type)
            if expectation.cause_type is not None:
                assert isinstance(excinfo.value.__cause__, expectation.cause_type)
            return None

        result = mloda.run_all(
            features,
            compute_frameworks=[self.compute_framework_name()],
            plugin_collector=plugin_collector,
            parallelization_modes={mode},
            flight_server=flight_server,
            data_access_collection=dac,
            global_filter=global_filter,
        )
        expectation.assert_result(result)
        return result
