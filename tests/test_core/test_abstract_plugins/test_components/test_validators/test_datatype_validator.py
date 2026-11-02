from datetime import datetime
from typing import Callable

import pytest
import pyarrow as pa

from mloda.core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeMismatchError,
    DataTypeValidator,
)
from mloda.user import DataType
from mloda.user import Feature
from mloda.provider import DefaultOptionKeys, FeatureSet


def _arrow_resolver(table: pa.Table) -> Callable[[str], DataType | None]:
    def resolve(col: str) -> DataType | None:
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


_D = DataType

_STRICT_WIDENINGS: list[tuple[DataType, DataType]] = [
    (_D.INT32, _D.INT8),
    (_D.INT32, _D.UINT8),
    (_D.INT64, _D.INT8),
    (_D.INT64, _D.UINT8),
    (_D.UINT64, _D.UINT8),
    (_D.DOUBLE, _D.INT8),
    (_D.DOUBLE, _D.INT64),
    (_D.DOUBLE, _D.UINT8),
    (_D.DOUBLE, _D.UINT64),
]


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

    def test_decimal_never_coerces(self) -> None:
        """DECIMAL never coerces to or from other numeric types, only to itself."""
        assert DataTypeValidator._types_compatible(DataType.DECIMAL, DataType.DOUBLE) is False
        assert DataTypeValidator._types_compatible(DataType.DOUBLE, DataType.DECIMAL) is False
        assert DataTypeValidator._types_compatible(DataType.DECIMAL, DataType.INT64) is False
        assert DataTypeValidator._types_compatible(DataType.INT64, DataType.DECIMAL) is False
        assert DataTypeValidator._types_compatible(DataType.DECIMAL, DataType.DECIMAL) is True

        assert DataTypeValidator._types_loosely_compatible(DataType.DECIMAL, DataType.DOUBLE) is False
        assert DataTypeValidator._types_loosely_compatible(DataType.DOUBLE, DataType.DECIMAL) is False
        assert DataTypeValidator._types_loosely_compatible(DataType.DECIMAL, DataType.INT64) is False
        assert DataTypeValidator._types_loosely_compatible(DataType.INT64, DataType.DECIMAL) is False
        assert DataTypeValidator._types_loosely_compatible(DataType.DECIMAL, DataType.DECIMAL) is True

    @pytest.mark.parametrize("declared,actual", _STRICT_WIDENINGS)
    def test_strict_widening_accepted(self, declared: DataType, actual: DataType) -> None:
        assert DataTypeValidator._types_compatible(declared, actual) is True

    @pytest.mark.parametrize(
        "declared,actual",
        [
            (_D.INT64, _D.UINT64),
            (_D.INT32, _D.UINT64),
            (_D.INT8, _D.UINT8),
            (_D.UINT8, _D.INT8),
            (_D.UINT64, _D.INT8),
            (_D.UINT64, _D.INT64),
            (_D.INT8, _D.INT32),
            (_D.UINT8, _D.UINT64),
            (_D.FLOAT, _D.INT8),
        ],
    )
    def test_strict_rejected(self, declared: DataType, actual: DataType) -> None:
        assert DataTypeValidator._types_compatible(declared, actual) is False

    def test_strict_naive_and_tz_never_widen_into_each_other(self) -> None:
        tz = DataType.TIMESTAMP_MICROS_TZ
        for naive in (DataType.TIMESTAMP_MILLIS, DataType.TIMESTAMP_MICROS):
            assert DataTypeValidator._types_compatible(naive, tz) is False
            assert DataTypeValidator._types_compatible(tz, naive) is False
        assert DataTypeValidator._types_compatible(tz, tz) is True

    @pytest.mark.parametrize(
        "declared,actual",
        [
            (_D.INT8, _D.FLOAT),
            (_D.FLOAT, _D.INT8),
            (_D.UINT8, _D.DOUBLE),
            (_D.DOUBLE, _D.UINT8),
            (_D.UINT64, _D.FLOAT),
            (_D.INT8, _D.UINT64),
            (_D.FLOAT, _D.UINT64),
            (_D.DOUBLE, _D.UINT64),
        ],
    )
    def test_lenient_numeric_family(self, declared: DataType, actual: DataType) -> None:
        assert DataTypeValidator._types_loosely_compatible(declared, actual) is True

    def test_lenient_timestamp_family_includes_tz(self) -> None:
        members = [_D.TIMESTAMP_MILLIS, _D.TIMESTAMP_MICROS, _D.TIMESTAMP_MICROS_TZ]
        for declared in members:
            for actual in members:
                assert DataTypeValidator._types_loosely_compatible(declared, actual) is True

    def test_lenient_cross_family_rejected(self) -> None:
        assert DataTypeValidator._types_loosely_compatible(_D.INT64, _D.TIMESTAMP_MICROS_TZ) is False
        assert DataTypeValidator._types_loosely_compatible(_D.TIMESTAMP_MICROS_TZ, _D.UINT8) is False
        assert DataTypeValidator._types_loosely_compatible(_D.STRING, _D.UINT8) is False

    @pytest.mark.parametrize("other", [_D.STRING, _D.INT64, _D.DOUBLE, _D.BINARY, _D.TIMESTAMP_MICROS_TZ])
    def test_list_exact_only(self, other: DataType) -> None:
        assert DataTypeValidator._types_compatible(_D.LIST, _D.LIST) is True
        assert DataTypeValidator._types_loosely_compatible(_D.LIST, _D.LIST) is True
        for declared, actual in ((_D.LIST, other), (other, _D.LIST)):
            assert DataTypeValidator._types_compatible(declared, actual) is False
            assert DataTypeValidator._types_loosely_compatible(declared, actual) is False


class TestValidateExtendedTypes:
    """End to end validate() on Arrow tables with the extended members."""

    @staticmethod
    def _strict(feature: Feature) -> Feature:
        feature.options.add_to_group(DefaultOptionKeys.strict_type_enforcement, True)
        return feature

    def test_strict_int8_column_declared_int32_passes(self) -> None:
        table = pa.table({"c": pa.array([1, 2], type=pa.int8())})
        feature_set = FeatureSet([self._strict(Feature.int32_of("c"))])
        DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_strict_tz_column_declared_naive_raises(self) -> None:
        table = pa.table({"c": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us", tz="UTC"))})
        feature_set = FeatureSet([self._strict(Feature.timestamp_micros_of("c"))])
        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, _arrow_resolver(table))

    def test_strict_naive_column_declared_tz_raises(self) -> None:
        table = pa.table({"c": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us"))})
        feature = self._strict(Feature("c", data_type=DataType.TIMESTAMP_MICROS_TZ))
        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(FeatureSet([feature]), _arrow_resolver(table))

    def test_lenient_tz_column_declared_naive_passes(self) -> None:
        table = pa.table({"c": pa.array([datetime(2024, 1, 1)], type=pa.timestamp("us", tz="UTC"))})
        DataTypeValidator.validate(FeatureSet([Feature.timestamp_micros_of("c")]), _arrow_resolver(table))

    def test_list_column_declared_string_raises_in_lenient(self) -> None:
        table = pa.table({"c": pa.array([["a"], ["b"]], type=pa.list_(pa.string()))})
        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(FeatureSet([Feature.str_of("c")]), _arrow_resolver(table))


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

        # Use pd.StringDtype() so the column resolves as STRING; with the per-framework
        # native-dispatch path, plain object dtype returns None (ambiguous) and the
        # validator skips it, which would defeat the mismatch assertion.
        df = pd.DataFrame({"name": pd.array(["alice", "bob"], dtype=pd.StringDtype())})
        fw = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        feature_set = FeatureSet([Feature.int64_of("name")])

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(feature_set, lambda col: fw._extract_column_data_type(df, col))

    def test_typed_feature_group_runs_through_run_all_on_pandas_matching_passes(self) -> None:
        from typing import Any

        import pandas as pd

        from mloda.provider import ComputeFramework, FeatureGroup, DataCreatorFeatureGroup
        from mloda.user import Feature as UFeature
        from mloda.user import PluginCollector, mloda
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        class _Src(DataCreatorFeatureGroup):
            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"price"}

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
            compute_frameworks=[PandasDataFrame],
            plugin_collector=pc,
        )
        df = next(d for d in results if "price_typed_match" in d.columns)
        assert df["price_typed_match"].tolist() == [2.0, 4.0, 6.0]

    def test_typed_feature_group_runs_through_run_all_on_pandas_mismatch_raises(self) -> None:
        from typing import Any

        import pandas as pd
        import pytest as _pytest

        from mloda.provider import ComputeFramework, FeatureGroup, DataCreatorFeatureGroup
        from mloda.user import Feature as UFeature
        from mloda.user import PluginCollector, mloda
        from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import (
            PandasDataFrame,
        )

        class _Src(DataCreatorFeatureGroup):
            @classmethod
            def feature_names_supported(cls) -> set[str]:
                return {"price"}

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
        # mlodaAPI wraps the DataTypeMismatchError in a generic Exception; match on the
        # mismatch substring so unrelated exception bugs don't satisfy the assertion.
        with _pytest.raises(Exception, match="STRING") as exc_info:
            mloda.run_all(
                [UFeature("price_typed_mismatch")],
                compute_frameworks=[PandasDataFrame],
                plugin_collector=pc,
            )
        assert "STRING" in str(exc_info.value)
        assert "DOUBLE" in str(exc_info.value)
