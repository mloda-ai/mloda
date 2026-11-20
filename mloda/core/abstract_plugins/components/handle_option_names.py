"""Option names that carry a named data access or connection handle."""

DATA_ACCESS_HANDLE_OPTION = "data_access_handle"
CONNECTION_HANDLE_OPTION = "connection_handle"
HANDLE_OPTION_NAMES: frozenset[str] = frozenset({DATA_ACCESS_HANDLE_OPTION, CONNECTION_HANDLE_OPTION})
