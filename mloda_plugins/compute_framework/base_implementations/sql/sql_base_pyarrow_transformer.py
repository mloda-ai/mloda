from typing import Any, Optional

from mloda.provider import BaseTransformer

try:
    import pyarrow as pa
except ImportError:
    pa = None


class SqlBasePyArrowTransformer(BaseTransformer):
    """Shared PyArrow transformer logic for SQL-based compute frameworks.

    Subclasses must implement:
    - framework(): Return the native relation type
    - import_fw(): Import the framework module
    - _convert_to_arrow(data): Convert native relation to PyArrow Table
    - _convert_to_native(data, connection): Convert PyArrow Table to native relation
    - _validate_connection(connection): Validate the connection object
    """

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return None
        return pa.Table

    @classmethod
    def import_other_fw(cls) -> None:
        pass  # noqa: F811

    @classmethod
    def _convert_to_arrow(cls, data: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def _convert_to_native(cls, data: Any, connection: Any) -> Any:
        raise NotImplementedError

    @classmethod
    def _validate_connection(cls, connection: Any) -> None:
        raise NotImplementedError

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        return cls._convert_to_arrow(data)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Optional[Any] = None) -> Any:
        if framework_connection_object is None:
            raise ValueError("A connection object is required for this transformation.")

        cls._validate_connection(framework_connection_object)
        return cls._convert_to_native(data, framework_connection_object)
