import logging
from typing import Any, Set, Type, Optional
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import SparkMergeEngine
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
    import pyspark.sql.functions as F
except ImportError:
    SparkSession = None
    DataFrame = None
    StructType = None
    StructField = None
    StringType = None
    IntegerType = None
    DoubleType = None
    BooleanType = None
    F = None

logger = logging.getLogger(__name__)


class SparkFramework(ComputeFrameWork):
    """Spark framework implementation for ComputeFrameWork.

    This framework leverages Apache Spark for distributed data processing.
    It requires a SparkSession to be provided through the framework connection object.
    """

    def set_framework_connection_object(self, framework_connection_object: Optional[Any] = None) -> None:
        """Use given SparkSession connection."""
        if SparkSession is None:
            raise ImportError("PySpark is not installed. To be able to use this framework, please install pyspark.")

        if self.framework_connection_object is None:
            if framework_connection_object is not None:
                if not isinstance(framework_connection_object, SparkSession):
                    raise ValueError(f"Expected a SparkSession object, got {type(framework_connection_object)}")
                self.framework_connection_object = framework_connection_object
            else:
                # Create a default local SparkSession if none provided
                self.framework_connection_object = (
                    SparkSession.builder.appName("MLoda-Spark-Framework")
                    .master("local[*]")
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                    .getOrCreate()
                )

    @staticmethod
    def is_available() -> bool:
        """Check if PySpark is installed and available."""
        try:
            from pyspark.sql import SparkSession

            return True
        except ImportError:
            return False

    @staticmethod
    def expected_data_framework() -> Any:
        return SparkFramework.spark_dataframe()

    def merge_engine(self) -> Type[BaseMergeEngine]:
        return SparkMergeEngine

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        column_names = set(data.columns)
        _selected_feature_names = self.identify_naming_convention(selected_feature_names, column_names)
        return data.select(*list(_selected_feature_names))

    def set_column_names(self) -> None:
        if self.data is not None:
            self.column_names = set(self.data.columns)

    @staticmethod
    def spark_dataframe() -> Any:
        if DataFrame is None:
            raise ImportError("PySpark is not installed. To be able to use this framework, please install pyspark.")
        return DataFrame

    @staticmethod
    def spark_session() -> Any:
        if SparkSession is None:
            raise ImportError("PySpark is not installed. To be able to use this framework, please install pyspark.")
        return SparkSession

    def _infer_spark_type(self, value: Any) -> Any:
        """Infer Spark data type from Python value."""
        if isinstance(value, bool):
            return BooleanType()
        elif isinstance(value, int):
            return IntegerType()
        elif isinstance(value, float):
            return DoubleType()
        else:
            return StringType()

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
    ) -> Any:
        transformed_data = self.apply_compute_framework_transformer(data)
        if transformed_data is not None:
            return transformed_data

        if isinstance(data, dict):
            """Initial data: Transform dict to Spark DataFrame"""
            if self.framework_connection_object is None:
                self.set_framework_connection_object()

            spark = self.framework_connection_object

            # Handle empty dict
            if not data:
                return spark.createDataFrame([], StructType([]))  # type: ignore

            # Infer schema from the first row of data
            first_key = next(iter(data.keys()))
            if not data[first_key]:  # Empty list
                schema_fields = [StructField(col, StringType(), True) for col in data.keys()]
                schema = StructType(schema_fields)
                return spark.createDataFrame([], schema)  # type: ignore

            # Create schema based on first values
            schema_fields = []
            for col_name, col_values in data.items():
                if col_values:
                    spark_type = self._infer_spark_type(col_values[0])
                    schema_fields.append(StructField(col_name, spark_type, True))
                else:
                    schema_fields.append(StructField(col_name, StringType(), True))

            schema = StructType(schema_fields)

            # Convert dict to list of rows
            if data:
                num_rows = len(next(iter(data.values())))
                rows = []
                for i in range(num_rows):
                    row = tuple(data[col][i] for col in data.keys())
                    rows.append(row)
                return spark.createDataFrame(rows, schema)  # type: ignore
            else:
                return spark.createDataFrame([], schema)  # type: ignore

        if hasattr(data, "__iter__") and not isinstance(data, (str, bytes, DataFrame)):
            """Added data: Add column to DataFrame"""
            if len(feature_names) == 1:
                feature_name = next(iter(feature_names))

                if feature_name in self.data.columns:
                    raise ValueError(f"Feature {feature_name} already exists in the DataFrame")

                # Convert data to list if it's not already
                data_list = list(data) if hasattr(data, "__iter__") else [data]

                # Add row numbers to both DataFrames for joining
                from pyspark.sql.window import Window

                window_spec = Window.orderBy(F.monotonically_increasing_id())

                # Add row numbers to existing DataFrame
                existing_with_row_num = self.data.withColumn("__row_num", F.row_number().over(window_spec))

                # Create new DataFrame with the new column
                if self.framework_connection_object is None:
                    self.set_framework_connection_object()

                spark = self.framework_connection_object
                new_data_df = spark.createDataFrame(
                    [(i + 1, val) for i, val in enumerate(data_list)],
                    StructType(
                        [
                            StructField("__row_num", IntegerType(), False),
                            StructField(feature_name, self._infer_spark_type(data_list[0] if data_list else ""), True),
                        ]
                    ),
                )

                # Join the DataFrames and drop the row number column
                result = existing_with_row_num.join(new_data_df, "__row_num").drop("__row_num")
                return result

            raise ValueError(f"Only one feature can be added at a time: {feature_names}")

        raise ValueError(f"Data {type(data)} is not supported by {self.__class__.__name__}")

    def filter_engine(self) -> Type[BaseFilterEngine]:
        return SparkFilterEngine
