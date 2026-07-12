from typing import Any
from mloda.user import DataAccessCollection
from mloda.provider import FeatureSet, BaseInputData
from mloda.user import Options


class ReadDB(BaseInputData):
    """
    ReadDB is responsible for loading and processing input data from databases.

    _auto_load_group triggers lazy plugin discovery when no ReadDB subclasses
    are found in the process. Only the read_dbs subdirectory is loaded.

    To suppress auto-loading:
        PluginLoader.disable_auto_load("feature_group/input_data/read_dbs")

    load_data is a template method exposing an opt-in lifecycle seam: a new
    backend implements ``produce_rows``, ``connect``, and ``is_valid_credentials``
    (optionally ``prepare_credentials``/``build_query``/``check_feature_in_data_access``)
    instead of overriding ``load_data`` wholesale. Overriding ``load_data`` directly
    is still supported.
    """

    _auto_load_group: str = "feature_group/input_data/read_dbs"

    @classmethod
    def prepare_credentials(cls, data_access: Any, features: FeatureSet) -> Any:
        """Overridable hook to normalize/validate credentials before connecting; default is pass-through."""
        return data_access

    @classmethod
    def produce_rows(cls, connection: Any, features: FeatureSet) -> Any:
        """The per-backend row hook; implement THIS (plus connect and is_valid_credentials) instead of load_data.
        Return fully materialized rows: close_connection runs in the template's finally block, so a lazy
        cursor or generator dies after close. The template does not call build_query; call it here if needed."""
        raise NotImplementedError

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """Template method: probe final-reader classification against the dynamic anchor,
        prepare credentials, connect, produce rows, then close."""
        if not cls.is_final_reader():
            raise NotImplementedError

        credentials = cls.prepare_credentials(data_access, features)
        connection = cls.get_connection(credentials)
        try:
            return cls.produce_rows(connection, features)
        finally:
            cls.close_connection(connection)

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        """Hook-based discovery requires overriding connect itself; a backend that customizes
        get_connection but leaves connect abstract must override load_data wholesale or also
        override connect."""
        # Requiring connect alongside produce_rows screens out intermediate bases that
        # supply a shared produce_rows but leave connect abstract.
        return ("produce_rows", "connect")

    @classmethod
    def connect(cls, credentials: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def build_query(cls, features: FeatureSet) -> str:
        """Builds a query to retrieve the data from the database."""
        raise NotImplementedError

    @classmethod
    def is_valid_credentials(cls, credentials: dict[str, Any]) -> bool:
        """Checks if the given dictionary is a valid credentials object.

        Matcher exception contract: match_read_db_data_access treats only NotImplementedError
        as a soft no-match; any other exception propagates and aborts matching for every
        reader sharing the DataAccessCollection.
        """
        raise NotImplementedError

    @classmethod
    def check_feature_in_data_access(cls, feature_name: str, data_access: Any) -> bool:
        """Obligatory function to check if the feature is in the data access.

        Same matcher exception contract as is_valid_credentials: only NotImplementedError
        is a soft no-match, any other exception propagates.
        """
        raise NotImplementedError

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        data_accesses: list[Any] = []

        if isinstance(data_access, DataAccessCollection):
            hint = options.get("data_access_handle") if options is not None else None
            if hint is not None and data_access.handles().get(hint) not in (None, "credentials"):
                hint = None
            creds = data_access.resolve("credentials", hint=hint)
            if creds:
                data_accesses.append(creds)
        elif isinstance(data_access, dict):
            data_accesses.append(data_access)

        if not data_accesses:
            return None

        matched_data_access = cls.match_read_db_data_access(data_accesses, feature_names)
        if matched_data_access is None:
            return None
        return matched_data_access

    @classmethod
    def match_read_db_data_access(cls, data_accesses: list[Any], feature_names: list[str]) -> Any:
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
    def close_connection(cls, connection: Any) -> None:
        """Releases a connection returned by get_connection; override for non-PEP-249 drivers."""
        connection.close()

    @classmethod
    def read_db(cls, credentials: dict[str, Any] | str, query: str) -> tuple[Any, Any]:
        connection = cls.get_connection(credentials)
        try:
            with connection as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    column_names = [description[0] for description in cursor.description]
                finally:
                    cursor.close()
        finally:
            cls.close_connection(connection)

        if result is None:
            raise ValueError(f"No data was returned from the query: {query}.")
        return result, column_names
