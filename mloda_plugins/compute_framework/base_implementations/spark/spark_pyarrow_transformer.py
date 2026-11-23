from typing import Any

from mloda.provider import BaseTransformer
from mloda_plugins.compute_framework.base_implementations.spark.spark_type_semantics import TIME_ZONE_METADATA_KEY

try:
    from pyspark.sql import DataFrame, SparkSession
except ImportError:
    DataFrame = None
    SparkSession = None

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment, unused-ignore]


def with_time_zones(sdf: Any, schema: Any) -> Any:
    """Record the zone of each zoned Arrow timestamp field in the Spark field metadata."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import TimestampType

    zones = {f.name: f.type.tz for f in schema if pa.types.is_timestamp(f.type) and f.type.tz}
    if not zones:
        return sdf
    columns = []
    for field in sdf.schema.fields:
        column = F.col("`" + field.name.replace("`", "``") + "`")
        if field.name in zones and isinstance(field.dataType, TimestampType):
            column = column.alias(field.name, metadata={**field.metadata, TIME_ZONE_METADATA_KEY: zones[field.name]})
        columns.append(column)
    return sdf.select(*columns)


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
        """
        Transform a Spark DataFrame to a PyArrow Table.

        This method uses Spark's native Arrow integration through toArrow()
        for efficient conversion.
        """
        table = data.toArrow()
        for index, (field, spark_field) in enumerate(zip(table.schema, data.schema.fields)):
            zone = spark_field.metadata.get(TIME_ZONE_METADATA_KEY)
            if zone is not None and pa.types.is_timestamp(field.type):
                table = table.set_column(
                    index, field.name, table.column(index).cast(pa.timestamp(field.type.unit, tz=zone))
                )
        return table

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

        return with_time_zones(spark.createDataFrame(rows), data.schema)
