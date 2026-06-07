"""
Shared base for the FeatureGroup-declared ``allow_empty_result`` policy, driven
end-to-end through the public API ``mloda.run_all``.

A FeatureGroup declares whether it may legitimately yield an empty result by overriding
the ``allow_empty_result`` classmethod (default ``False``).

- A FeatureGroup that does not override it (default ``False``), requested as a final
  feature, must raise when it yields an empty result: empty is an error.
- A FeatureGroup that overrides it to ``True``, requested as a final feature, must succeed
  with an empty result. This is the knowledge-graph / search use case where zero matches is
  valid.

Both FeatureGroups emit a columnar dict with a single empty column, which ``transform``
turns into a zero-row frame in every framework. The framework's ``_is_empty`` predicate
detects the zero rows, and ``run_validate_output_features`` raises ``EmptyResultError``
(a ``ValueError``, message ``"Data cannot be empty"``) unless the FG allows it.

Subclasses implement only ``compute_framework_name`` (and, for connection-backed frameworks
such as DuckDB / SQLite / Spark / Iceberg, ``get_connection``). The connection is threaded
through ``Feature.options`` + a ``DataAccessCollection``, mirroring ``AsofRunAllTestBase``.

This module is intentionally NOT collected as tests (no ``Test`` prefix).
"""

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
from mloda.user import Options
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


def _records_from_frame(data: Any) -> list[dict[str, Any]]:
    """Extract native frame rows to plain Python row dicts across backends.

    Mirrors ``AsofRunAllTestBase._records_from_frame`` so the empty-result assertion is
    framework-agnostic: an empty result is simply ``len(_records_from_frame(result)) == 0``.
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


class _EmptyResultMatchData(MatchData):
    """Surfaces a framework connection when one is provided; inert otherwise.

    For connection-backed frameworks (DuckDB / SQLite / Spark / Iceberg) the connection is
    threaded through ``Feature.options`` and a ``DataAccessCollection``; this returns it so
    the framework can build its native (empty) frame. For pandas / polars / pyarrow /
    python_dict no connection is passed, so this returns ``None`` and feature resolution
    falls back to the ``DataCreator`` declared by ``input_data``.
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
        if framework_connection_object is not None:
            return framework_connection_object
        if data_access_collection is not None and data_access_collection.connections:
            for conn in data_access_collection.connections.values():
                return conn
        return None


class EmptyResultDefaultFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup that yields zero rows and does NOT allow empty results."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_default_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Columnar dict with one empty column -> zero rows after transform in every framework.
        return {"empty_result_default_col": []}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_default_col"}


class EmptyResultAllowedFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup that yields zero rows and DECLARES empty results are allowed."""

    @classmethod
    def allow_empty_result(cls) -> bool:
        return True

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_allowed_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Columnar dict with one empty column -> zero rows after transform in every framework.
        return {"empty_result_allowed_col": []}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_allowed_col"}


_ENABLED_DEFAULT = PluginCollector.enabled_feature_groups({EmptyResultDefaultFeatureGroup})
_ENABLED_ALLOWED = PluginCollector.enabled_feature_groups({EmptyResultAllowedFeatureGroup})


class EmptyResultRunAllTestBase(ABC):
    """Drives the ``allow_empty_result`` policy end-to-end through ``run_all``.

    Subclasses implement ``compute_framework_name`` and, for connection-backed frameworks,
    ``get_connection`` (plus a ``teardown_method`` to close it).
    """

    @classmethod
    @abstractmethod
    def compute_framework_name(cls) -> str:
        """Return the compute framework name string for ``compute_frameworks=[...]``."""
        pass

    def get_connection(self) -> Optional[Any]:
        """Return a framework connection object, or None when none is needed."""
        return None

    def _feature_and_dac(self, feature_name: str) -> tuple[Feature, Optional[DataAccessCollection]]:
        conn = self.get_connection()
        if conn is not None:
            feature = Feature(
                name=feature_name,
                options={
                    EmptyResultDefaultFeatureGroup.get_class_name(): conn,
                    EmptyResultAllowedFeatureGroup.get_class_name(): conn,
                },
            )
            return feature, DataAccessCollection(connections={conn})
        return Feature(name=feature_name), None

    def test_empty_result_default_raises(self, flight_server: Any) -> None:
        """A FeatureGroup that does not allow empty results raises on an empty final result."""
        feature, dac = self._feature_and_dac("empty_result_default_col")

        with pytest.raises(Exception, match="Data cannot be empty"):
            mloda.run_all(
                [feature],
                compute_frameworks=[self.compute_framework_name()],
                plugin_collector=_ENABLED_DEFAULT,
                parallelization_modes={ParallelizationMode.SYNC},
                flight_server=flight_server,
                data_access_collection=dac,
            )

    def test_empty_result_allowed_succeeds(self, flight_server: Any) -> None:
        """A FeatureGroup that overrides allow_empty_result()->True succeeds with an empty result."""
        feature, dac = self._feature_and_dac("empty_result_allowed_col")

        result = mloda.run_all(
            [feature],
            compute_frameworks=[self.compute_framework_name()],
            plugin_collector=_ENABLED_ALLOWED,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
            data_access_collection=dac,
        )

        assert len(result) == 1
        assert len(_records_from_frame(result[0])) == 0
