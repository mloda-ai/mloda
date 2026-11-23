from typing import Any

from mloda.provider import BaseTransformer

try:
    from pyspark.sql import DataFrame, SparkSession
except ImportError:
    DataFrame = None
    SparkSession = None

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def _assert_utc_timezone(column: str, zone: str, session_zone: str) -> None:
    if zone != "UTC" or session_zone != "UTC":
        raise ValueError(
            f"Column '{column}' has time zone '{zone}' with Spark session time zone '{session_zone}'. "
            "Spark can only preserve 'UTC' columns in a 'UTC' session. Use another compute framework for other zones."
        )


class SparkPyArrowTransformer(BaseTransformer):
    """
    Transformer for converting between Spark DataFrame and PyArrow Table.

    This transformer handles bidirectional conversion between Spark DataFrame
    and PyArrow Table data structures, using Spark's native PyArrow integration
    for efficient zero-copy operations.
    """

    @classmethod
    def framework(cls) -> Any:
        if DataFrame is None:
            return NotImplementedError
        return DataFrame

    @classmethod
    def other_framework(cls) -> Any:
        if pa is None:
            return NotImplementedError
        return pa.Table

    @classmethod
    def import_fw(cls) -> None:
        from pyspark.sql import DataFrame  # noqa: F401

    @classmethod
    def import_other_fw(cls) -> None:
        import pyarrow as pa  # noqa: F401

    @classmethod
    def transform_fw_to_other_fw(cls, data: Any) -> Any:
        """Export a Spark frame, refusing a session zone that would change UTC column semantics."""
        from pyspark.sql.types import TimestampType

        session_zone = data.sparkSession.conf.get("spark.sql.session.timeZone")
        for field in data.schema:
            if isinstance(field.dataType, TimestampType):
                _assert_utc_timezone(field.name, "UTC", session_zone)
        return data.toArrow()

    @classmethod
    def validate_arrow_timezones(cls, data: Any, spark: Any) -> None:
        """Reject column zones Spark cannot retain, including schema-only columns."""
        session_zone = spark.conf.get("spark.sql.session.timeZone")
        for field in data.schema:
            if pa.types.is_timestamp(field.type) and field.type.tz is not None:
                _assert_utc_timezone(field.name, field.type.tz, session_zone)

    @classmethod
    def transform_other_fw_to_fw(cls, data: Any, framework_connection_object: Any | None = None) -> Any:
        """
        Transform a PyArrow Table to a Spark DataFrame.

        This method uses SparkSession's createDataFrame method with direct
        PyArrow Table support for efficient conversion.
        """
        if DataFrame is None or SparkSession is None:
            raise ImportError("PySpark is not installed. To be able to use this framework, please install pyspark.")

        # Get SparkSession from framework connection object or active session
        if framework_connection_object is not None:
            if not isinstance(framework_connection_object, SparkSession):
                raise ValueError(f"Expected a SparkSession object, got {type(framework_connection_object)}")
            spark = framework_connection_object
        else:
            # Try to get active SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                raise ValueError("No active SparkSession found and no framework connection object provided.")

        cls.validate_arrow_timezones(data, spark)

        # Create Spark DataFrame directly from PyArrow Table
        # Convert PyArrow Table to list of dictionaries
        pydict = data.to_pydict()

        # Convert columnar format to row format
        num_rows = data.num_rows
        if num_rows == 0:
            # Handle empty table
            return spark.createDataFrame([], schema=None)

        # Convert from columnar to row-based format
        rows = []
        for i in range(num_rows):
            row = {}
            for column_name in data.column_names:
                row[column_name] = pydict[column_name][i]
            rows.append(row)

        return spark.createDataFrame(rows)
