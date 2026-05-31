"""
Shared base for ASOF (point-in-time) join integration tests driven through the
public API ``mloda.run_all``.

The engine-level shared tests exhaustively cover ASOF semantics per backend. These
tests only prove the plumbing fires:
``run_all -> JoinStep -> BaseMergeEngine.merge -> merge_asof`` and that the parent
feature group can consume the joined frame.

One representative scenario is used: ``direction="backward"``, single by-key,
reusing the ``backward_single_key`` data shape from the shared ASOF scenarios.

Subclasses implement only ``compute_framework_name`` (and, for connection-backed
frameworks such as DuckDB, ``get_connection`` + ``teardown_method``); the
parametrized test method lives in this base so adding a backend is a thin,
hooks-only addition.

This module is intentionally NOT collected as tests (no ``Test`` prefix).
"""

import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Optional

import pytest

from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import MatchData
from mloda.user import DataAccessCollection
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Index
from mloda.user import JoinSpec
from mloda.user import Link
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector
from mloda.user import mloda

from tests.test_plugins.compute_framework.test_tooling.asof.asof_scenarios import ASOF_SCENARIOS

import logging

logger = logging.getLogger(__name__)

try:
    import polars as pl
except ImportError:
    logger.warning("Polars is not installed. Some tests will be skipped.")
    pl = None  # type: ignore[assignment]

try:
    import duckdb
except ImportError:
    logger.warning("DuckDB is not installed. Some tests will be skipped.")
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    logger.warning("PyArrow is not installed. Some tests will be skipped.")
    pa = None  # type: ignore[assignment]


_SCENARIO = ASOF_SCENARIOS["backward_single_key"]

# Expected backward asof matches for backward_single_key, encoded as "k|t|rv":
#   (k=1, t=10) -> right t=5  -> rv=1
#   (k=1, t=20) -> right t=18 -> rv=2
#   (k=2, t=15) -> right t=5  -> rv=3
_EXPECTED_ENCODED = sorted(["1|10|1", "1|20|2", "2|15|3"])


def _columns_to_lists(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert a list of row dicts into a column-oriented dict for DataCreator."""
    keys = rows[0].keys()
    return {key: [row[key] for row in rows] for key in keys}


def _records_from_frame(data: Any) -> list[dict[str, Any]]:
    """Extract native frame rows to plain Python row dicts across backends."""
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


def _encode_records(records: list[dict[str, Any]]) -> list[str]:
    """Encode each joined row as ``"k|t|rv"`` so the asof match is verifiable."""
    return [f"{r['k']}|{r['t']}|{r['rv']}" for r in records]


def _asof_link() -> Link:
    return Link.asof(
        JoinSpec(AsofLeftFeature, Index(("k",))),
        JoinSpec(AsofRightFeature, Index(("k",))),
        left_time_column="t",
        right_time_column="t",
        direction="backward",
    )


class _AsofMatchData(MatchData):
    """Surfaces a DuckDB or sqlite3 connection when one is provided; inert otherwise.

    Under pandas/polars/python_dict/pyarrow no connection is passed, so this returns
    None and the framework selection falls back to the requested ``compute_frameworks``.
    """

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        raise NotImplementedError

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if feature_name not in cls.feature_names_supported():
            return None
        if duckdb is not None and isinstance(framework_connection_object, duckdb.DuckDBPyConnection):
            return framework_connection_object
        if isinstance(framework_connection_object, sqlite3.Connection):
            return framework_connection_object
        if data_access_collection is not None and data_access_collection.connections:
            for conn in data_access_collection.connections.values():
                if duckdb is not None and isinstance(conn, duckdb.DuckDBPyConnection):
                    return conn
                if isinstance(conn, sqlite3.Connection):
                    return conn
        return None


class AsofLeftFeature(FeatureGroup, _AsofMatchData):
    """Left side of the ASOF join: emits by-key ``k``, time ``t`` and value ``lv``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"k", "t", "lv"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _columns_to_lists(_SCENARIO["left"])

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"k", "t", "lv"}


class AsofRightFeature(FeatureGroup, _AsofMatchData):
    """Right side of the ASOF join: emits by-key ``k``, time ``t`` and value ``rv``."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"k", "t", "rv"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _columns_to_lists(_SCENARIO["right"])

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"k", "t", "rv"}


class AsofJoinedFeature(FeatureGroup, _AsofMatchData):
    """Parent feature group that requires the ASOF merge of the two leaf groups."""

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        link = _asof_link()
        return {
            Feature(name="lv", link=link, index=Index(("k",))),
            Feature(name="rv", index=Index(("k",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        records = _records_from_frame(data)
        return {cls.get_class_name(): _encode_records(records)}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_ENABLED = PluginCollector.enabled_feature_groups({AsofLeftFeature, AsofRightFeature, AsofJoinedFeature})


class AsofRunAllTestBase(ABC):
    """Drives a backward, single-by-key ASOF join end-to-end through run_all.

    Subclasses implement ``compute_framework_name`` and, for connection-backed
    frameworks, ``get_connection`` (plus a ``teardown_method`` to close it).
    """

    @classmethod
    @abstractmethod
    def compute_framework_name(cls) -> str:
        """Return the compute framework name string for ``compute_frameworks=[...]``."""
        pass

    def get_connection(self) -> Optional[Any]:
        """Return a framework connection object, or None when none is needed."""
        return None

    def _run(self, modes: set[ParallelizationMode], flight_server: Any) -> list[str]:
        conn = self.get_connection()

        if conn is not None:
            feature = Feature(
                name=AsofJoinedFeature.get_class_name(),
                options={
                    AsofLeftFeature.get_class_name(): conn,
                    AsofRightFeature.get_class_name(): conn,
                    AsofJoinedFeature.get_class_name(): conn,
                },
            )
            data_access_collection = DataAccessCollection(connections={conn})
        else:
            feature = Feature(name=AsofJoinedFeature.get_class_name())
            data_access_collection = None

        result = mloda.run_all(
            [feature],
            links={_asof_link()},
            compute_frameworks=[self.compute_framework_name()],
            plugin_collector=_ENABLED,
            flight_server=flight_server,
            parallelization_modes=modes,
            data_access_collection=data_access_collection,
        )

        assert len(result) == 1
        records = _records_from_frame(result[0])
        return sorted(str(r["AsofJoinedFeature"]) for r in records)

    @pytest.mark.parametrize("modes", [{ParallelizationMode.SYNC}, {ParallelizationMode.THREADING}])
    def test_backward_single_key(self, modes: set[ParallelizationMode], flight_server: Any) -> None:
        assert self._run(modes, flight_server) == _EXPECTED_ENCODED
