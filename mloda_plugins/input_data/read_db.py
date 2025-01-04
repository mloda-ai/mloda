from typing import Any, Dict, List, Optional, Tuple, Union
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.hashable_dict import HashableDict
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.options import Options


class ReadDB(BaseInputData):
    """
    ReadDB is responsible for loading and processing input data from databases.

    This class should be inherited by all classes that are responsible for reading databases.


    The following methods should be implemented in the child classes:
    - load_data
    - connect
    - build_query
    - is_valid_credentials
    - check_feature_in_data_access
    """

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """This function should be implemented from child classes."""
        raise NotImplementedError

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        """Builds a query to retrieve the data from the database."""
        raise NotImplementedError

    @classmethod
    def is_valid_credentials(cls, credentials: Dict[str, Any]) -> bool:
        """Checks if the given dictionary is a valid credentials object."""
        raise NotImplementedError

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        """Obligatory function to check if the feature is in the data access."""
        raise NotImplementedError

    def init_reader(self, options: Optional[Options]) -> Tuple["ReadDB", Any]:
        if options is None:
            raise ValueError("Options were not set.")

        reader_data_access = options.get("BaseInputData")

        if reader_data_access is None:
            raise ValueError("Reader data access was not set.")

        reader, data_access = reader_data_access
        return reader(), data_access

    def load(self, features: FeatureSet) -> Any:
        reader, data_access = self.init_reader(features.options)
        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: List[str]) -> Any:
        data_accesses: List[Any] = []

        if isinstance(data_access, DataAccessCollection):
            if data_access.credential_dicts:
                data_accesses.append(data_access.credential_dicts)
        elif isinstance(data_access, HashableDict):
            data_accesses.append(data_access)

        if not data_accesses:
            return False

        matched_data_access = cls.match_read_db_data_access(data_accesses, feature_names)
        if matched_data_access is False:
            return False
        return matched_data_access

    @classmethod
    def match_read_db_data_access(cls, data_accesses: List[Any], feature_names: List[str]) -> Any:
        if len(feature_names) > 1:
            raise ValueError("This should not happen.")

        for data_access in data_accesses:
            try:
                if cls.is_valid_credentials(data_access):
                    try:
                        if cls.check_feature_in_data_access(feature_names[0], data_access):
                            return data_access
                        continue
                    except NotImplementedError:
                        pass

                    return data_access
            except NotImplementedError:
                continue
        return False

    @classmethod
    def get_connection(cls, credentials: Any) -> Any:
        """Establishes a database connection using the provided credentials."""
        connection = None
        connection = cls.connect(credentials)
        if connection is None:
            raise ValueError("Connection to database failed.")
        return connection

    @classmethod
    def read_db(cls, credentials: Union[Dict[str, Any], str], query: str) -> Tuple[Any, Any]:
        with cls.get_connection(credentials) as conn:
            cursor = None
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                result = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
            finally:
                try:
                    cursor.close()
                except AttributeError:
                    pass

        if result is None:
            raise ValueError(f"No data was returned from the query: {query}.")
        return result, column_names
