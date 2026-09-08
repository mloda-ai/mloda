"""SYNC end-to-end inner join between two DuckDB DataCreator feature groups, connected only
through a ConnectionSpec(DuckDBFramework) in the DataAccessCollection (no live connection on
Feature.options). Mirrors test_python_dict_one_to_many_join_run_all.py's run_all shape.
"""

from typing import Any

import pytest

try:
    import duckdb
except ImportError:
    duckdb = None  # type: ignore[assignment]

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import (
    ConnectionSpec,
    DataAccessCollection,
    Feature,
    FeatureName,
    Index,
    JoinSpec,
    Link,
    Options,
    ParallelizationMode,
    PluginCollector,
    mloda,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework

pytestmark = pytest.mark.skipif(duckdb is None or pa is None, reason="DuckDB or PyArrow is not installed.")


def _spec_join_link() -> Link:
    return Link(
        "inner",
        JoinSpec(SpecJoinLeftFeature, Index(("user_id",))),
        JoinSpec(SpecJoinRightFeature, Index(("user_id",))),
    )


class SpecJoinLeftFeature(FeatureGroup):
    """Left side: DataCreator returning a plain dict, pinned to DuckDBFramework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"user_id", "uname"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"user_id": [1, 2], "uname": ["ann", "bob"]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"user_id", "uname"}


class SpecJoinRightFeature(FeatureGroup):
    """Right side: DataCreator returning a plain dict, pinned to DuckDBFramework."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(supports_features={"user_id", "amount"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"user_id": [1, 2], "amount": [10, 20]}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"user_id", "amount"}


class SpecJoinedFeature(FeatureGroup):
    """Parent joining left and right, encoding each joined row as 'uname|amount'."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        link = _spec_join_link()
        return {
            Feature(name="uname", link=link, index=Index(("user_id",))),
            Feature(name="amount", index=Index(("user_id",))),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return data.project(f"uname || '|' || CAST(amount AS VARCHAR) AS {cls.get_class_name()}")

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {cls.get_class_name()}


_ENABLED = PluginCollector.enabled_feature_groups({SpecJoinLeftFeature, SpecJoinRightFeature, SpecJoinedFeature})


class TestDuckDBSpecJoinRunAll:
    """A ConnectionSpec(DuckDBFramework) registered in the DataAccessCollection, with no live
    connection on Feature.options, must be enough for a SYNC join across two DuckDB feature groups."""

    def test_inner_join_via_connection_spec(self, flight_server: Any) -> None:
        feature = Feature(name=SpecJoinedFeature.get_class_name())
        data_access_collection = DataAccessCollection(connections={ConnectionSpec(DuckDBFramework)})

        result = mloda.run_all(
            [feature],
            links={_spec_join_link()},
            compute_frameworks={DuckDBFramework},
            plugin_collector=_ENABLED,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            data_access_collection=data_access_collection,
        )

        assert len(result) == 1
        final_data = result[0]
        assert isinstance(final_data, pa.Table)
        column = final_data.column(SpecJoinedFeature.get_class_name()).to_pylist()
        assert sorted(column) == ["ann|10", "bob|20"]
