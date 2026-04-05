import sqlite3
from typing import Any, Dict, List, Optional, Set, Type

import pytest

from mloda.provider import MatchData
from mloda.provider import ComputeFramework
from mloda.provider import FeatureGroup
from mloda.provider import FeatureSet
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Options
from mloda.user import PluginCollector
from mloda.user import mloda
from mloda.user import ParallelizationMode
from mloda.user import DataAccessCollection
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework, _regexp
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.test_plugins.integration_plugins.test_data_creator import ATestDataCreator


@pytest.fixture
def sqlite_conn() -> Any:
    conn = sqlite3.connect(":memory:")
    conn.create_function("REGEXP", 2, _regexp)
    yield conn
    conn.close()


sqlite_test_dict: Dict[str, Any] = {
    "id": [1, 2, 3, 4, 5],
    "value": [10, 20, 30, 40, 50],
    "category": ["A", "B", "A", "C", "B"],
    "score": [1.5, 2.5, 3.5, 4.5, 5.5],
}


class SqliteTestDataCreator(ATestDataCreator):
    compute_framework = SqliteFramework

    conversion = {
        **ATestDataCreator.conversion,
        SqliteFramework: lambda data: data,
    }

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        return sqlite_test_dict


class ATestSqliteFeatureGroup(FeatureGroup, MatchData):
    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        if not SqliteFramework.is_available():
            return None

        if feature_name not in cls.feature_names_supported():
            return None

        if isinstance(framework_connection_object, sqlite3.Connection):
            return framework_connection_object

        if data_access_collection is None:
            return None

        if data_access_collection.initialized_connection_objects is None:
            return None

        if data_access_collection.initialized_connection_objects:
            for conn in data_access_collection.initialized_connection_objects:
                if isinstance(conn, sqlite3.Connection):
                    return conn
        return None

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {SqliteFramework}


class SqliteSimpleTransformFeatureGroup(ATestSqliteFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str == "doubled_value":
            return {Feature("value")}
        elif feature_name_str == "score_plus_ten":
            return {Feature("score")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "doubled_value":
                result_data = result_data.select(_raw_sql="*, value * 2 AS doubled_value")
            elif feature_name == "score_plus_ten":
                result_data = result_data.select(_raw_sql="*, score + 10 AS score_plus_ten")

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"doubled_value", "score_plus_ten"}


class SqliteSecondTransformFeatureGroup(ATestSqliteFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return {Feature("doubled_value")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "quadrupled_value":
                result_data = result_data.select(_raw_sql="*, doubled_value * 2 AS quadrupled_value")

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"quadrupled_value"}


class SqliteAggregationFeatureGroup(ATestSqliteFeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["avg_value_by_category", "count_by_category"]:
            return {Feature("value"), Feature("category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        result_data = data

        for feat in features.features:
            feature_name = str(feat.name)

            if feature_name == "avg_value_by_category":
                result_data = result_data.select(
                    _raw_sql="*, AVG(value) OVER (PARTITION BY category) AS avg_value_by_category"
                )
            elif feature_name == "count_by_category":
                result_data = result_data.select(
                    _raw_sql="*, COUNT(*) OVER (PARTITION BY category) AS count_by_category"
                )

        return result_data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"avg_value_by_category", "count_by_category"}


class SqliteCheckData(FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        feature_name_str = feature_name.name if isinstance(feature_name, FeatureName) else str(feature_name)

        if feature_name_str in ["pyarrow_avg_value_by_category_sqlite"]:
            return {Feature("avg_value_by_category")}

        return set()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data = data.rename_columns(["id", "value", "category", "score", "pyarrow_avg_value_by_category_sqlite"])
        return data

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return {"pyarrow_avg_value_by_category_sqlite"}

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[ComputeFramework]]:
        return {PyArrowTable}


class TestSqliteIntegrationWithMlodaAPI:
    def test_basic_sqlite_feature_calculation(self, flight_server: Any, sqlite_conn: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {SqliteTestDataCreator, SqliteSimpleTransformFeatureGroup}
        )

        feature_list: List[Feature | str] = [
            Feature(name="doubled_value", options={"SqliteTestDataCreator": sqlite_conn}),
            Feature(name="score_plus_ten", options={"SqliteTestDataCreator": sqlite_conn}),
        ]

        data_access_collection = DataAccessCollection(initialized_connection_objects={sqlite_conn})

        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            plugin_collector=plugin_collector,
            data_access_collection=data_access_collection,
            compute_frameworks={SqliteFramework},
        )

        final_data = result[0]
        assert hasattr(final_data, "columns")

        assert "doubled_value" in final_data.columns
        assert "score_plus_ten" in final_data.columns

        final_df = final_data.df()
        doubled_values = final_df["doubled_value"].tolist()
        assert doubled_values == [20, 40, 60, 80, 100]

        score_plus_ten = final_df["score_plus_ten"].tolist()
        assert score_plus_ten == [11.5, 12.5, 13.5, 14.5, 15.5]

    def test_sqlite_aggregation_features(self, flight_server: Any, sqlite_conn: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {SqliteTestDataCreator, SqliteAggregationFeatureGroup}
        )

        feature_list: List[Feature | str] = [
            Feature(name="avg_value_by_category", options={"SqliteTestDataCreator": sqlite_conn}),
            Feature(name="count_by_category", options={"SqliteTestDataCreator": sqlite_conn}),
        ]

        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            parallelization_modes={ParallelizationMode.SYNC},
            compute_frameworks={SqliteFramework},
            plugin_collector=plugin_collector,
        )

        final_data = result[0]
        assert "avg_value_by_category" in final_data.columns
        assert "count_by_category" in final_data.columns

        final_df = final_data.df()
        avg_values = final_df["avg_value_by_category"].tolist()
        count_values = final_df["count_by_category"].tolist()

        assert len(set(avg_values)) <= 3
        assert len(set(count_values)) <= 3

    def test_multiple_feature_groups_sqlite_pipeline(self, flight_server: Any, sqlite_conn: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {SqliteTestDataCreator, SqliteSimpleTransformFeatureGroup, SqliteSecondTransformFeatureGroup}
        )

        feature_list: List[Feature | str] = [
            Feature(name="quadrupled_value", options={"SqliteTestDataCreator": sqlite_conn})
        ]

        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            compute_frameworks={SqliteFramework},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert len(result) == 1
        final_data = result[0]

        assert "quadrupled_value" in final_data.columns
        quadrupled_values = final_data.df()["quadrupled_value"].tolist()
        assert quadrupled_values == [40, 80, 120, 160, 200]

    def test_transform_to_pyarrow(self, flight_server: Any, sqlite_conn: Any) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {
                SqliteTestDataCreator,
                SqliteSimpleTransformFeatureGroup,
                SqliteSecondTransformFeatureGroup,
                SqliteAggregationFeatureGroup,
                SqliteCheckData,
            }
        )

        feature_list: List[Feature | str] = [
            Feature(name="pyarrow_avg_value_by_category_sqlite", options={"SqliteTestDataCreator": sqlite_conn})
        ]

        result = mloda.run_all(
            feature_list,
            flight_server=flight_server,
            compute_frameworks={SqliteFramework, PyArrowTable},
            plugin_collector=plugin_collector,
            parallelization_modes={ParallelizationMode.SYNC},
        )

        assert len(result) == 1
        assert "pyarrow_avg_value_by_category_sqlite" in result[0].column_names
