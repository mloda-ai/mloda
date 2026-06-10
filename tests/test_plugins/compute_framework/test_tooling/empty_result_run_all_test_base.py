"""
Shared base for the FeatureGroup-declared ``allow_empty_result`` policy, driven
end-to-end through the public API ``mloda.run_all``.

A feature computation must return a SCHEMA-BEARING result: at least one column. Rows are
optional. The guard in ``ComputeFramework.run_validate_output_features`` fires only for
FINAL requested features when ``feature_group.allow_empty_result()`` is False, and raises
``EmptyResultError`` for:

- State B: a schema-less result (zero columns, i.e. ``_extract_column_names`` returns an
  empty set).

State A, a ``None`` result (no result at all), never reaches the guard: the method returns
early when ``self.data is None``. A ``None`` result is rejected UPSTREAM instead: every
framework raises in ``transform`` (e.g. pandas: ``Data <class 'NoneType'> is not
supported``; python_dict: ``Data type <class 'NoneType'> is not supported by
PythonDictFramework``), so state A never reaches the guard on any backend.

State C, a schema-bearing result with ZERO ROWS, MUST SUCCEED. This is the key behavior:
a zero-row but column-bearing frame is a valid, well-typed empty result.

Both empty-producing FeatureGroups emit a columnar dict with a single empty column.
``transform`` turns this into a zero-row frame in every schema-bearing framework (PyArrow,
Pandas, Polars, DuckDB, SQLite, Spark, Iceberg): the schema is carried as metadata even at
zero rows, so these are state C and SUCCEED. For ``python_dict`` (``List[Dict[str, Any]]``)
``transform`` collapses ``{"col": []}`` to ``[]``, which has no schema: that is state B and
STILL RAISES (unless ``allow_empty_result()`` is True).

Subclasses implement only ``compute_framework_name`` (and, for connection-backed frameworks
such as DuckDB / SQLite / Spark / Iceberg, ``get_connection``). The python_dict subclass also
overrides ``default_empty_is_schemaless`` to opt into the "raises" expectation for the default
FeatureGroup.

This module is intentionally NOT collected as tests (no ``Test`` prefix).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.compute_framework import EmptyResultError
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
    framework-agnostic: a zero-row result is simply ``len(_records_from_frame(result)) == 0``.
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
        # Columnar dict with one empty column -> zero rows after transform.
        # Schema-bearing frameworks keep the column (state C); python_dict drops it (state B).
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


class EmptyResultSchemalessAllowedFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup that yields a ZERO-COLUMN result and DECLARES empty results are allowed.

    Unlike ``EmptyResultAllowedFeatureGroup`` (one empty column, schema-less only on
    python_dict), the ``{}`` returned here is schema-less (state B) on EVERY framework:
    ``transform`` produces a frame with zero columns. With ``allow_empty_result()`` True the
    output guard accepts it, and the framework must hand the zero-column result through to
    the caller unchanged (result selection has no columns to match against and must not
    re-judge what the guard already accepted).
    """

    @classmethod
    def allow_empty_result(cls) -> bool:
        return True

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_schemaless_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Zero columns -> schema-less (state B) in every framework, accepted via the opt-in.
        return {}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_schemaless_col"}


class EmptyResultNoneFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup whose ``calculate_feature`` returns ``None`` (state A)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_none_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # No result at all -> state A (rejected upstream of the guard, see module docstring).
        return None

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_none_col"}


_ENABLED_DEFAULT = PluginCollector.enabled_feature_groups({EmptyResultDefaultFeatureGroup})
_ENABLED_ALLOWED = PluginCollector.enabled_feature_groups({EmptyResultAllowedFeatureGroup})
_ENABLED_SCHEMALESS_ALLOWED = PluginCollector.enabled_feature_groups({EmptyResultSchemalessAllowedFeatureGroup})
_ENABLED_NONE = PluginCollector.enabled_feature_groups({EmptyResultNoneFeatureGroup})


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

    @classmethod
    def default_empty_is_schemaless(cls) -> bool:
        """Whether the default FG's empty result is schema-less (state B) on this framework.

        Schema-bearing frameworks keep the column at zero rows (state C -> success); the
        python_dict subclass overrides this to True because ``transform`` collapses the
        columnar dict to ``[]`` (state B -> raises).
        """
        return False

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

    def test_empty_result_default(self, flight_server: Any) -> None:
        """Default (not-allowed) FG requested as a final feature.

        Schema-bearing frameworks return a schema-bearing zero-row result (state C) and
        SUCCEED. python_dict drops the schema (state B) and raises ``EmptyResultError``.
        """
        feature, dac = self._feature_and_dac("empty_result_default_col")

        if self.default_empty_is_schemaless():
            # run_all wraps the framework-raised EmptyResultError in a plain Exception, so we
            # assert on the type NAME surfaced in the wrapped message (stable across the
            # green-phase message change) rather than the exception class or its text.
            with pytest.raises(Exception) as excinfo:
                mloda.run_all(
                    [feature],
                    compute_frameworks=[self.compute_framework_name()],
                    plugin_collector=_ENABLED_DEFAULT,
                    parallelization_modes={ParallelizationMode.SYNC},
                    flight_server=flight_server,
                    data_access_collection=dac,
                )
            assert EmptyResultError.__name__ in str(excinfo.value)
            return

        result = mloda.run_all(
            [feature],
            compute_frameworks=[self.compute_framework_name()],
            plugin_collector=_ENABLED_DEFAULT,
            parallelization_modes={ParallelizationMode.SYNC},
            flight_server=flight_server,
            data_access_collection=dac,
        )

        assert len(result) == 1
        assert len(_records_from_frame(result[0])) == 0

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
