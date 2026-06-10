"""Generic policy-conformance tooling for FeatureGroup-level policy flags.

Drives any FeatureGroup-declared policy flag (such as ``allow_empty_result()``) end-to-end
through the public API ``mloda.run_all``, across every built-in compute framework and at
least one worker-based parallelization mode. A policy case is expressed as an expectation:
either ``PolicySuccess`` (run_all returns results that satisfy an assertion) or
``PolicyRaises`` (run_all raises an Exception whose message contains a substring).

This module is intentionally NOT collected as tests (no ``Test`` prefix). It is the shared
base that concrete per-framework conformance suites build on.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest

from mloda.provider import FeatureGroup
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
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
    pa = None  # type: ignore[assignment]


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
    elif isinstance(data, list):
        records = data
    elif hasattr(data, "to_dicts"):
        records = data.to_dicts()
    elif hasattr(data, "to_dict"):
        records = data.to_dict("records")
    else:
        records = data.df().to_dict("records")
    return records


@dataclass(frozen=True)
class PolicySuccess:
    """run_all returns results; ``assert_result`` is called on them."""

    assert_result: Callable[[list[Any]], None]


@dataclass(frozen=True)
class PolicyRaises:
    """run_all raises an Exception whose ``str()`` contains ``match_substring``."""

    match_substring: str


PolicyExpectation = PolicySuccess | PolicyRaises


class PolicyRunAllTestBase(ABC):
    """Drives a FeatureGroup-level policy end-to-end through ``run_all``.

    Subclasses implement ``compute_framework_name`` and, for connection-backed frameworks,
    ``get_connection`` plus ``connection_keyed_feature_groups``.
    """

    @classmethod
    @abstractmethod
    def compute_framework_name(cls) -> str:
        """Return the compute framework name string for ``compute_frameworks=[...]``."""
        pass

    def get_connection(self) -> Optional[Any]:
        """Return a framework connection object, or None when none is needed."""
        return None

    @classmethod
    def connection_keyed_feature_groups(cls) -> set[type[FeatureGroup]]:
        """FeatureGroup classes whose ``get_class_name()`` keys the connection in options."""
        return set()

    def _feature_and_dac(self, feature_name: str) -> tuple[Feature, Optional[DataAccessCollection]]:
        conn = self.get_connection()
        if conn is not None:
            feature = Feature(
                name=feature_name,
                options={fg.get_class_name(): conn for fg in self.connection_keyed_feature_groups()},
            )
            return feature, DataAccessCollection(connections={conn})
        return Feature(name=feature_name), None

    def assert_policy_case(
        self,
        *,
        feature_name: str,
        plugin_collector: PluginCollector,
        expectation: PolicyExpectation,
        mode: ParallelizationMode,
        flight_server: Any,
    ) -> Optional[list[Any]]:
        feature, dac = self._feature_and_dac(feature_name)

        if isinstance(expectation, PolicyRaises):
            with pytest.raises(Exception) as excinfo:
                mloda.run_all(
                    [feature],
                    compute_frameworks=[self.compute_framework_name()],
                    plugin_collector=plugin_collector,
                    parallelization_modes={mode},
                    flight_server=flight_server,
                    data_access_collection=dac,
                )
            assert expectation.match_substring in str(excinfo.value)
            return None

        result = mloda.run_all(
            [feature],
            compute_frameworks=[self.compute_framework_name()],
            plugin_collector=plugin_collector,
            parallelization_modes={mode},
            flight_server=flight_server,
            data_access_collection=dac,
        )
        expectation.assert_result(result)
        return result
