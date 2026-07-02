"""
Shared base for the schema-based empty-result contract, driven end-to-end through the
public API ``mloda.run_all``. It is the canonical first consumer of the generic
``PolicyRunAllTestBase``.

A feature computation must return a SCHEMA-BEARING result: at least one column. Rows are
optional. The guard in ``ComputeFramework.run_validate_output_features`` fires only for
FINAL requested features, with no per-FeatureGroup opt-in, and raises ``EmptyResultError``
for:

- State B: a schema-less result (zero columns, i.e. ``_extract_column_names`` returns an
  empty set).

State A, a ``None`` result (no result at all), never reaches the guard: the method returns
early when ``self.data is None``. A ``None`` result is rejected UPSTREAM instead: every
framework raises in ``transform`` (e.g. pandas: ``Data <class 'NoneType'> is not
supported``; python_dict: ``Data type <class 'NoneType'> is not supported by
PythonDictFramework``), so state A never reaches the guard on any backend.

State C, a schema-bearing result with ZERO ROWS, MUST SUCCEED. This is the key behavior:
a zero-row but column-bearing frame is a valid, well-typed empty result.

Both zero-row FeatureGroups emit a columnar dict with a single empty column. ``transform``
turns this into a zero-row frame in every framework: the schema-bearing frameworks (PyArrow,
Pandas, Polars, DuckDB, SQLite, Spark, Iceberg) carry the schema as metadata even at zero
rows, and ``python_dict`` (columnar ``dict[str, list]``) keeps ``{"col": []}`` as a
schema-bearing zero-row frame. All of these are state C and SUCCEED. Only the schema-less
``{}`` (zero columns) is state B and RAISES, uniformly on every backend.

Subclasses implement only ``compute_framework_name`` (and, for connection-backed frameworks
such as DuckDB / SQLite / Spark / Iceberg, ``get_connection``). ``default_empty_is_schemaless``
remains as an override point but is False for every built-in framework.

This module is intentionally NOT collected as tests (no ``Test`` prefix).
"""

from typing import Any, Optional

import pytest

from mloda.core.abstract_plugins.compute_framework import EmptyResultError
from mloda.provider import BaseInputData
from mloda.provider import DataCreator
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.provider import MatchData
from mloda.user import DataAccessCollection
from mloda.user import Options
from mloda.user import ParallelizationMode
from mloda.user import PluginCollector

from tests.test_plugins.compute_framework.test_tooling.policy_run_all_test_base import (
    PolicyRaises,
    PolicyRunAllTestBase,
    PolicySuccess,
    records_from_frame,
)


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
    """Root FeatureGroup that yields a schema-bearing zero-row result under default behavior."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_default_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Columnar dict with one empty column -> zero rows after transform.
        # Every framework keeps the column at zero rows (state C), including python_dict.
        return {"empty_result_default_col": []}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"empty_result_default_col"}


class EmptyResultAllowedFeatureGroup(FeatureGroup, _EmptyResultMatchData):
    """Root FeatureGroup that yields a schema-bearing zero-row result (one empty column)."""

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
    """Root FeatureGroup that yields a ZERO-COLUMN (schema-less) ``{}`` result.

    The ``{}`` returned here is schema-less (state B) on EVERY framework: ``transform``
    produces a frame with zero columns. With ``allow_empty_result()`` retired, the output
    guard now RAISES ``EmptyResultError`` for this result on every backend.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(supports_features={"empty_result_schemaless_col"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Zero columns -> schema-less (state B) in every framework, now always raises.
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


def _assert_single_empty_result(result: list[Any]) -> None:
    """A run_all result that is exactly one schema-bearing frame with zero rows."""
    assert len(result) == 1
    assert len(records_from_frame(result[0])) == 0


class EmptyResultRunAllTestBase(PolicyRunAllTestBase):
    """Drives the schema-based empty-result contract end-to-end through ``run_all``.

    Subclasses implement ``compute_framework_name`` and, for connection-backed frameworks,
    ``get_connection`` (plus a ``teardown_method`` to close it).
    """

    @classmethod
    def connection_keyed_feature_groups(cls) -> set[type[FeatureGroup]]:
        return {EmptyResultDefaultFeatureGroup, EmptyResultAllowedFeatureGroup}

    @classmethod
    def default_empty_is_schemaless(cls) -> bool:
        """Whether the default FG's empty result is schema-less (state B) on this framework.

        Every backend now keeps the single empty column at zero rows (state C -> success),
        including python_dict under the columnar model, so this returns False everywhere.
        """
        return False

    def _run_default_case(self, mode: ParallelizationMode, flight_server: Any) -> None:
        if self.default_empty_is_schemaless():
            expectation: PolicySuccess | PolicyRaises = PolicyRaises(match_substring=EmptyResultError.__name__)
        else:
            expectation = PolicySuccess(assert_result=_assert_single_empty_result)

        self.assert_policy_case(
            feature_name="empty_result_default_col",
            plugin_collector=_ENABLED_DEFAULT,
            expectation=expectation,
            mode=mode,
            flight_server=flight_server,
        )

    def _run_allowed_case(self, mode: ParallelizationMode, flight_server: Any) -> None:
        self.assert_policy_case(
            feature_name="empty_result_allowed_col",
            plugin_collector=_ENABLED_ALLOWED,
            expectation=PolicySuccess(assert_result=_assert_single_empty_result),
            mode=mode,
            flight_server=flight_server,
        )

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_empty_result_default(self, mode: ParallelizationMode, flight_server: Any) -> None:
        """Default FG requested as a final feature.

        Every framework, python_dict included, returns a schema-bearing zero-row result
        (state C) and SUCCEEDS.
        SYNC-only frameworks (DuckDB / SQLite) override these methods to run SYNC only.
        """
        self._run_default_case(mode, flight_server)

    @pytest.mark.parametrize("mode", [ParallelizationMode.SYNC, ParallelizationMode.THREADING])
    def test_empty_result_allowed_succeeds(self, mode: ParallelizationMode, flight_server: Any) -> None:
        """A FeatureGroup returning a schema-bearing zero-row result succeeds on every backend."""
        self._run_allowed_case(mode, flight_server)
