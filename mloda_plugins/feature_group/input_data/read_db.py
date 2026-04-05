from typing import Any, Dict, List, Optional, Tuple, Union
from mloda.user import DataAccessCollection
from mloda.provider import FeatureSet, HashableDict, BaseInputData
from mloda.user import Options


class ReadDB(BaseInputData):
    """
    ReadDB is responsible for loading and processing input data from databases.

    _auto_load_group triggers lazy plugin discovery when no ReadDB subclasses
    are found in the process. Only the read_dbs subdirectory is loaded.

    To suppress auto-loading:
        PluginLoader.disable_auto_load("feature_group/input_data/read_dbs")

    The following methods should be implemented in the child classes:
    - load_data
    - connect
    - build_query
    - is_valid_credentials
    - check_feature_in_data_access
    """

    _auto_load_group: str = "feature_group/input_data/read_dbs"

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
            raise ValueError(
                f"Options were not set for {self.__class__.__name__}. "
                f"Provide an Options object containing a 'BaseInputData' key "
                f"with a (reader_class, data_access) tuple."
            )

        reader_data_access = options["BaseInputData"]

        if reader_data_access is None:
            raise ValueError(
                f"'BaseInputData' key is missing or None in the provided Options for {self.__class__.__name__}. "
                f"Set options with Options(group={{'BaseInputData': (ReaderClass, credentials_dict)}})."
            )

        reader, data_access = reader_data_access
        return reader(), data_access

    def load(self, features: FeatureSet) -> Any:
        _options = None
        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reader, data_access = self.init_reader(_options)

        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: List[str], options: Options) -> Any:
        data_accesses: List[Any] = []

        if isinstance(data_access, DataAccessCollection):
            if data_access.credential_dicts:
                data_accesses.append(data_access.credential_dicts)
        elif isinstance(data_access, HashableDict):
            data_accesses.append(data_access)

        if not data_accesses:
            return None

        matched_data_access = cls.match_read_db_data_access(data_accesses, feature_names)
        if matched_data_access is None:
            return None
        return matched_data_access

    @classmethod
    def match_read_db_data_access(cls, data_accesses: List[Any], feature_names: List[str]) -> Any:
        if len(feature_names) > 1:
            raise ValueError(
                f"ReadDB.match_read_db_data_access expects exactly one feature name, "
                f"but received {len(feature_names)}: {feature_names}.\n"
                "Each database feature should be resolved individually."
            )

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
        return None

    @classmethod
    def get_connection(cls, credentials: Any) -> Any:
        """Establishes a database connection using the provided credentials."""
        connection = cls.connect(credentials)
        if connection is None:
            raise ValueError(
                f"Connection to database failed for {cls.__name__}.\n"
                f"Credentials type: {type(credentials).__name__}.\n"
                "The connect() method returned None. Verify that:\n"
                "  - The database server is reachable.\n"
                "  - The credentials (host, port, user, password, database name) are correct.\n"
                "  - The required database driver is installed."
            )
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
