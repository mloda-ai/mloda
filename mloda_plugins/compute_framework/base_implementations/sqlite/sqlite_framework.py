import logging
import re
import sqlite3
from typing import Any, Optional, Set, Type

from mloda.provider import BaseMergeEngine
from mloda.provider import ComputeFramework
from mloda.provider import BaseFilterEngine
from mloda.user import FeatureName, ParallelizationMode
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_filter_engine import SqliteFilterEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_merge_engine import SqliteMergeEngine
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

logger = logging.getLogger(__name__)


def _regexp(pattern: str, string: Optional[str]) -> bool:
    if string is None:
        return False
    return bool(re.search(pattern, string))


class SqliteFramework(ComputeFramework):
    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        if framework_connection_object is None:
            raise ValueError("A sqlite3.Connection object is required.")
        if not isinstance(framework_connection_object, sqlite3.Connection):
            raise ValueError(f"Expected a sqlite3.Connection object, got {type(framework_connection_object)}")
        if self.framework_connection_object is not None:
            if self.framework_connection_object is not framework_connection_object:
                raise ValueError("A different connection is already set. Cannot replace an existing connection.")
            return  # same connection passed again — safe no-op
        framework_connection_object.create_function("REGEXP", 2, _regexp, deterministic=True)
        self.framework_connection_object = framework_connection_object

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def expected_data_framework(cls) -> Any:
        return SqliteRelation

    @classmethod
    def merge_engine(cls) -> Type[BaseMergeEngine]:
        return SqliteMergeEngine

    def select_data_by_column_names(
        self, data: Any, selected_feature_names: Set[FeatureName], column_ordering: Optional[str] = None
    ) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(
            selected_feature_names, column_names, ordering=column_ordering
        )

        selected_columns = list(_selected_feature_names)
        return data.select(*selected_columns).df()

    def set_column_names(self) -> None:
        self.column_names = set(self.data.columns)

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            if self.framework_connection_object is None:
                raise ValueError(
                    "Framework connection object is not set. Please call set_framework_connection_object() first."
                )
            return SqliteRelation.from_dict(self.framework_connection_object, data)

        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if hasattr(self.data, "columns") and feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the relation")

                temp_data = {feature_name: list(data) if hasattr(data, "__iter__") else [data]}

                if self.framework_connection_object is None:
                    raise ValueError(
                        "Framework connection object is not set. Please call set_framework_connection_object() first."
                    )
                return SqliteRelation.from_dict(self.framework_connection_object, temp_data)
            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    @classmethod
    def supported_parallelization_modes(cls) -> Set[ParallelizationMode]:
        return {ParallelizationMode.SYNC}

    @classmethod
    def filter_engine(cls) -> Type[BaseFilterEngine]:
        return SqliteFilterEngine
