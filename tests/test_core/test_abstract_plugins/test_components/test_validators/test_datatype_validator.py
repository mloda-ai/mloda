from typing import Callable, Optional

import pytest
import pyarrow as pa

from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeMismatchError,
    DataTypeValidator,
)
from mloda.user import DataType
from mloda.user import Feature
from mloda.provider import FeatureSet


def _arrow_resolver(table: pa.Table) -> Callable[[str], Optional[DataType]]:
    def resolve(col: str) -> Optional[DataType]:
        if col not in table.schema.names:
            return None
        return DataType.from_arrow_type_safe(table.schema.field(col).type)

    return resolve


class TestDataTypeMismatchError:
    """Test DataTypeMismatchError exception class."""

    def test_datatype_mismatch_error_message(self) -> None:
        """Test DataTypeMismatchError contains useful info."""
        error = DataTypeMismatchError("age", DataType.INT32, DataType.STRING)
        msg = str(error)
        assert "age" in msg
        assert "INT32" in msg
        assert "STRING" in msg


class TestTypesCompatible:
    """Test the _types_compatible static method."""

    def test_types_compatible_exact_match(self) -> None:
        """Test exact type match is compatible."""
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.INT32) is True
        assert DataTypeValidator._types_compatible(DataType.STRING, DataType.STRING) is True
        assert DataTypeValidator._types_compatible(DataType.BOOLEAN, DataType.BOOLEAN) is True

    def test_types_compatible_widening(self) -> None:
        """Test type widening rules."""
        # INT32 -> INT64 is safe widening
        assert DataTypeValidator._types_compatible(DataType.INT64, DataType.INT32) is True
        # FLOAT -> DOUBLE is safe widening
        assert DataTypeValidator._types_compatible(DataType.DOUBLE, DataType.FLOAT) is True
        # INT32 -> DOUBLE is safe widening
        assert DataTypeValidator._types_compatible(DataType.DOUBLE, DataType.INT32) is True
        # TIMESTAMP_MILLIS -> TIMESTAMP_MICROS
        assert DataTypeValidator._types_compatible(DataType.TIMESTAMP_MICROS, DataType.TIMESTAMP_MILLIS) is True

    def test_types_incompatible(self) -> None:
        """Test incompatible types return False."""
        # STRING to INT is never compatible
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.STRING) is False
        # INT64 to INT32 (narrowing) is not allowed
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.INT64) is False
        # DOUBLE to FLOAT (narrowing) is not allowed
        assert DataTypeValidator._types_compatible(DataType.FLOAT, DataType.DOUBLE) is False
        # BOOLEAN to INT is not allowed
        assert DataTypeValidator._types_compatible(DataType.INT32, DataType.BOOLEAN) is False


class TestValidate:
    """Test the validate static method."""

    def test_validate_passes_on_match(self) -> None:
        """Test validation passes when types match."""
        # Create PyArrow table with INT32 column
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        feature = Feature.int32_of("age")
        feature_set = FeatureSet([feature])

        # Should not raise
        DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_validate_raises_on_mismatch(self) -> None:
        """Test validation raises DataTypeMismatchError on type mismatch."""
        # Create PyArrow table with STRING column
        table = pa.table({"age": pa.array(["25", "30"], type=pa.string())})

        feature = Feature.int32_of("age")  # Declared as INT32
        feature_set = FeatureSet([feature])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_validate_skips_untyped_features(self) -> None:
        """Test validation skips features without declared data_type."""
        table = pa.table({"age": pa.array(["25", "30"], type=pa.string())})

        feature = Feature.not_typed("age")  # No data_type
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should not raise - untyped features are skipped
        DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_validate_passes_with_widening(self) -> None:
        """Test validation passes when actual type can be widened to declared type."""
        # Create PyArrow table with INT32 column
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        feature = Feature.int64_of("age")  # Declared as INT64
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should not raise - INT32 can be widened to INT64
        DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_validate_multiple_features(self) -> None:
        """Test validation works with multiple features."""
        table = pa.table(
            {
                "age": pa.array([25, 30], type=pa.int32()),
                "name": pa.array(["Alice", "Bob"], type=pa.string()),
                "score": pa.array([95.5, 87.3], type=pa.float64()),
            }
        )

        feature_set = FeatureSet(
            [
                Feature.int32_of("age"),
                Feature.str_of("name"),
                Feature.double_of("score"),
            ]
        )

        # Should not raise - all types match
        DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_validate_raises_on_first_mismatch(self) -> None:
        """Test validation raises on first type mismatch."""
        table = pa.table(
            {
                "age": pa.array([25, 30], type=pa.int32()),
                "name": pa.array([123, 456], type=pa.int32()),  # Wrong type
            }
        )

        feature_set = FeatureSet()
        feature_set.add(Feature.int32_of("age"))
        feature_set.add(Feature.str_of("name"))  # Expects STRING

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, _arrow_resolver(table))


class TestValidateEnforcesOnPandas:
    """DataTypeValidator must enforce declared types against pandas-backed data.

    Before issue #432 was fixed, the validator silently no-op'd on any non-Arrow data,
    so declaring ``return_data_type_rule`` on a pandas FeatureGroup got no enforcement.
    """

    def test_validate_enforces_on_pandas_matching_type_passes(self) -> None:
        import pandas as pd

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        df = pd.DataFrame({"age": [25, 30]})
        fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        feature_set = FeatureSet([Feature.int64_of("age")])

        # int64 column declared as INT64 — must not raise.
        DataTypeValidator.validate(feature_set, lambda col: fw._extract_column_data_type(df, col))

    def test_validate_enforces_on_pandas_mismatch_raises(self) -> None:
        import pandas as pd

        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        df = pd.DataFrame({"name": ["alice", "bob"]})
        fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        feature_set = FeatureSet([Feature.int64_of("name")])

        # String column declared as INT64 — must raise even in lenient mode.
        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, lambda col: fw._extract_column_data_type(df, col))

    def test_typed_feature_group_runs_through_run_all_on_pandas_matching_passes(self) -> None:
        from typing import Any, Optional

        import pandas as pd

        from mloda.provider import ComputeFramework, DataCreator, FeatureGroup
        from mloda.user import Feature as UFeature
        from mloda.user import PluginCollector, mloda
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        class _Src(FeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[Any]:
                return DataCreator({"price"})

            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                return pd.DataFrame({"price": [1.0, 2.0, 3.0]})

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        class _TypedMatch(FeatureGroup):
            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"price_typed_match"}

            @classmethod
            def input_features(cls, options: Any, feature_name: Any) -> Any:
                return {UFeature("price")}

            @classmethod
            def return_data_type_rule(cls, feature: Any) -> Any:
                return DataType.DOUBLE

            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                data["price_typed_match"] = data["price"] * 2.0
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        pc = PluginCollector.enabled_feature_groups({_Src, _TypedMatch})
        results = mloda.run_all(
            [UFeature("price_typed_match")],
            compute_frameworks={PandasDataFrame},
            plugin_collector=pc,
        )
        df = next(d for d in results if "price_typed_match" in d.columns)
        assert df["price_typed_match"].tolist() == [2.0, 4.0, 6.0]

    def test_typed_feature_group_runs_through_run_all_on_pandas_mismatch_raises(self) -> None:
        from typing import Any, Optional

        import pandas as pd
        import pytest as _pytest

        from mloda.provider import ComputeFramework, DataCreator, FeatureGroup
        from mloda.user import Feature as UFeature
        from mloda.user import PluginCollector, mloda
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        class _Src(FeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[Any]:
                return DataCreator({"price"})

            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                return pd.DataFrame({"price": [1.0, 2.0, 3.0]})

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        class _TypedMismatch(FeatureGroup):
            """Declares STRING but returns float — enforcement must fire."""

            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"price_typed_mismatch"}

            @classmethod
            def input_features(cls, options: Any, feature_name: Any) -> Any:
                return {UFeature("price")}

            @classmethod
            def return_data_type_rule(cls, feature: Any) -> Any:
                return DataType.STRING

            @classmethod
            def calculate_feature(cls, data: Any, features: Any) -> Any:
                data["price_typed_mismatch"] = data["price"] * 2.0
                return data

            @classmethod
            def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
                return {PandasDataFrame}

        pc = PluginCollector.enabled_feature_groups({_Src, _TypedMismatch})
        # mlodaAPI wraps the DataTypeMismatchError in a generic Exception — match that.
        with _pytest.raises(Exception) as exc_info:
            mloda.run_all(
                [UFeature("price_typed_mismatch")],
                compute_frameworks={PandasDataFrame},
                plugin_collector=pc,
            )
        assert "STRING" in str(exc_info.value)
        assert "DOUBLE" in str(exc_info.value)
