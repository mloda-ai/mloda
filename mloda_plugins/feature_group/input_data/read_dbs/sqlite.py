import os
from typing import Any

import pyarrow as pa
import sqlite3

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.input_data.read_db import ReadDB


class SQLITEReader(ReadDB):
    @classmethod
    def db_path(cls) -> str:
        return "sqlite"

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        if isinstance(credentials, HashableDict):
            credential = credentials.data[cls.db_path()]
            return sqlite3.connect(credential)
        else:
            raise ValueError("Credentials must be an HashableDict.")

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        query = "select "

        for feature in features.get_all_names():
            query += f"{feature}, "

        query = query[:-2] + " "  # last comma is removed

        query += "from "
        query += f"{cls.get_table(features.options)};"

        if query is None:
            raise ValueError("query cannot be None")
        elif len(query) == 0:
            raise ValueError("query cannot be empty")
        return query

    @classmethod
    def is_valid_credentials(cls, credentials: Any) -> bool:
        """Checks if the given dictionary is a valid credentials object."""

        if isinstance(credentials, HashableDict):
            db_path = credentials.data.get(cls.db_path(), None)
        else:
            db_path = None

        if not isinstance(db_path, str):
            return False

        if not os.path.isfile(db_path):
            raise ValueError(f"Database file {db_path} does not exist, but key is given.")
        return True

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        query = cls.build_query(features)
        result, column_names = cls.read_db(data_access, query)
        return cls.read_as_pa_data(result, column_names)

    @classmethod
    def read_as_pa_data(cls, result: Any, column_names: Any) -> Any:
        schema = pa.schema([(column_names[i], DataType.infer_arrow_type(result[0][i])) for i in range(len(result[0]))])
        data_dicts = [{column_names[i]: row[i] for i in range(len(row))} for row in result]
        table = pa.Table.from_pylist(data_dicts, schema=schema)
        return table

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        # get tables in the database
        result, _ = cls.read_db(data_access, query="SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [table[0] for table in result]

        # check if the feature_name is in the tables
        for table in table_names:
            result, _ = cls.read_db(data_access, query=f"PRAGMA table_info({table});")
            column_names = [column[1] for column in result]
            if feature_name in column_names:
                cls.set_table_name(data_access, table)
                return True
        return False

    @classmethod
    def set_table_name(cls, data_access: Any, table_name: str) -> None:
        if data_access.data.get("table_name"):
            if data_access.data["table_name"] != table_name:
                raise ValueError(f"Table name is already set to {data_access.data['table_name']} and not {table_name}.")
            return

        data_access.data["table_name"] = table_name

    @classmethod
    def get_table(cls, options: Options | None) -> Any:
        if options is None:
            raise ValueError("Options were not set.")
        return options.get("BaseInputData")[1].data["table_name"]
