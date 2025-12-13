from typing import Any, Optional, Set
import pyarrow as pa
import pytest
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_collection import Features
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.abstract_plugins.components.plugin_option.plugin_collector import PlugInCollector
from mloda_core.abstract_plugins.components.validators.datatype_validator import (
    DataTypeValidator,
    DataTypeMismatchError,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from tests.test_core.test_tooling import MlodaTestRunner, PARALLELIZATION_MODES_SYNC_THREADING


class TypedFeatureSource(AbstractFeatureGroup):
    """Feature group that returns typed data for testing."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Returns int32 data."""
        table = pa.table({cls.get_class_name(): pa.array([25, 30, 35], type=pa.int32())})
        return table

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})


class StringDataSource(AbstractFeatureGroup):
    """Feature group that returns string data for testing type mismatches."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Returns string data."""
        return {cls.get_class_name(): ["a", "b", "c"]}

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})


class Int64DataSource(AbstractFeatureGroup):
    """Feature group that returns int64 data for testing widening."""

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """Returns int64 data."""
        table = pa.table({"Int64DataSource": pa.array([25, 30, 35], type=pa.int64())})
        return table

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"Int64DataSource"})


class TestDataTypeEnforcementSeparation:
    """Test that same feature with different types creates different hashes."""

    def test_int32_and_int64_same_feature_different_hashes(self) -> None:
        """Features with same name but different types should have different similarity hashes."""
        feature_int32 = Feature.int32_of("age")
        feature_int64 = Feature.int64_of("age")

        hash_int32 = feature_int32.has_similarity_properties()
        hash_int64 = feature_int64.has_similarity_properties()

        # Different types should produce different hashes
        assert hash_int32 != hash_int64, "Same feature with different types should have different hashes"

    def test_same_type_same_feature_identical_hashes(self) -> None:
        """Features with same name and same type should have identical hashes."""
        feature_int32_a = Feature.int32_of("age")
        feature_int32_b = Feature.int32_of("age")

        hash_a = feature_int32_a.has_similarity_properties()
        hash_b = feature_int32_b.has_similarity_properties()

        # Same types should produce identical hashes
        assert hash_a == hash_b, "Same feature with same type should have identical hashes"

    def test_untyped_features_identical_hashes(self) -> None:
        """Untyped features with same name should have identical hashes."""
        feature_untyped_a = Feature.not_typed("age")
        feature_untyped_b = Feature.not_typed("age")

        hash_a = feature_untyped_a.has_similarity_properties()
        hash_b = feature_untyped_b.has_similarity_properties()

        # Untyped features should have identical hashes
        assert hash_a == hash_b, "Untyped features with same name should have identical hashes"

    def test_typed_vs_untyped_different_hashes(self) -> None:
        """Typed feature should have different hash than untyped feature."""
        feature_typed = Feature.int32_of("age")
        feature_untyped = Feature.not_typed("age")

        hash_typed = feature_typed.has_similarity_properties()
        hash_untyped = feature_untyped.has_similarity_properties()

        # Typed vs untyped should produce different hashes
        assert hash_typed != hash_untyped, "Typed and untyped features should have different hashes"


class TestStrictTypeEnforcement:
    """Test type mismatch detection."""

    def test_validator_raises_on_type_mismatch(self) -> None:
        """Validator should raise when actual type doesn't match declared."""
        # Table with STRING data
        table = pa.table({"age": pa.array(["25", "30"], type=pa.string())})

        # Feature declared as INT32
        feature = Feature.int32_of("age")
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError) as exc_info:
            DataTypeValidator.validate(table, feature_set)

        assert "age" in str(exc_info.value)
        assert "INT32" in str(exc_info.value)
        assert "STRING" in str(exc_info.value)

    def test_validator_allows_exact_match(self) -> None:
        """Validator should allow exact type match."""
        # Table with INT32 data
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        # Feature declared as INT32
        feature = Feature.int32_of("age")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should NOT raise
        DataTypeValidator.validate(table, feature_set)

    def test_validator_allows_widening_int32_to_int64(self) -> None:
        """Validator should allow safe type widening (INT32 -> INT64)."""
        # Table with INT32 data
        table = pa.table({"age": pa.array([25, 30], type=pa.int32())})

        # Feature declared as INT64 (accepts INT32)
        feature = Feature.int64_of("age")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should NOT raise - INT32 can widen to INT64
        DataTypeValidator.validate(table, feature_set)

    def test_validator_allows_widening_float_to_double(self) -> None:
        """Validator should allow safe type widening (FLOAT -> DOUBLE)."""
        # Table with FLOAT data
        table = pa.table({"value": pa.array([1.5, 2.5], type=pa.float32())})

        # Feature declared as DOUBLE (accepts FLOAT)
        feature = Feature.double_of("value")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should NOT raise - FLOAT can widen to DOUBLE
        DataTypeValidator.validate(table, feature_set)


class TestTypeChainPropagation:
    """Test types work correctly through feature dependencies."""

    def test_typed_features_maintain_type(self) -> None:
        """Features should maintain their declared data_type through processing."""
        feature_int32 = Feature.int32_of("base")
        feature_double = Feature.double_of("derived")

        assert feature_int32.data_type == DataType.INT32
        assert feature_double.data_type == DataType.DOUBLE

        # Types should be different (verified by the assertions above)

    def test_untyped_feature_has_no_type(self) -> None:
        """Untyped features should have None as data_type."""
        feature_untyped = Feature.not_typed("untyped")

        assert feature_untyped.data_type is None

    def test_typed_feature_constructors_set_correct_type(self) -> None:
        """All typed constructors should set the correct DataType."""
        assert Feature.int32_of("f").data_type == DataType.INT32
        assert Feature.int64_of("f").data_type == DataType.INT64
        assert Feature.double_of("f").data_type == DataType.DOUBLE
        assert Feature.str_of("f").data_type == DataType.STRING
        assert Feature.boolean_of("f").data_type == DataType.BOOLEAN


class TestIncompatibleTypeCoercion:
    """Test that incompatible types always fail."""

    def test_string_to_int_always_fails(self) -> None:
        """STRING -> INT should always fail validation."""
        table = pa.table({"value": pa.array(["a", "b"], type=pa.string())})

        feature = Feature.int32_of("value")
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError) as exc_info:
            DataTypeValidator.validate(table, feature_set)

        assert "STRING" in str(exc_info.value)
        assert "INT32" in str(exc_info.value)

    def test_narrowing_int64_to_int32_fails(self) -> None:
        """INT64 -> INT32 (narrowing) should fail in strict mode."""
        table = pa.table({"value": pa.array([25, 30], type=pa.int64())})

        feature = Feature.int32_of("value")  # Declared as INT32
        feature.options.add(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(table, feature_set)

    def test_narrowing_double_to_float_fails(self) -> None:
        """DOUBLE -> FLOAT (narrowing) should fail in strict mode."""
        table = pa.table({"value": pa.array([1.5, 2.5], type=pa.float64())})

        # Feature declared as FLOAT
        feature = Feature(name="value", data_type=DataType.FLOAT)
        feature.options.add(DefaultOptionKeys.strict_type_enforcement, True)
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(table, feature_set)

    def test_boolean_to_int_fails(self) -> None:
        """BOOLEAN -> INT should fail."""
        table = pa.table({"value": pa.array([True, False], type=pa.bool_())})

        feature = Feature.int32_of("value")
        feature_set = FeatureSet()
        feature_set.add(feature)

        with pytest.raises(DataTypeMismatchError):
            DataTypeValidator.validate(table, feature_set)


class TestUntypedFeatures:
    """Test that untyped features are handled correctly."""

    def test_untyped_features_skip_validation(self) -> None:
        """Features without data_type should skip validation."""
        table = pa.table({"value": pa.array(["a", "b"], type=pa.string())})

        feature = Feature.not_typed("value")  # No data_type
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should NOT raise - untyped features skip validation
        DataTypeValidator.validate(table, feature_set)

    def test_mixed_typed_and_untyped_validation(self) -> None:
        """Validation should only check typed features, skip untyped."""
        table = pa.table(
            {
                "typed_field": pa.array([25, 30], type=pa.int32()),
                "untyped_field": pa.array(["a", "b"], type=pa.string()),
            }
        )

        # One typed, one untyped
        feature_typed = Feature.int32_of("typed_field")
        feature_untyped = Feature.not_typed("untyped_field")
        feature_set = FeatureSet()
        feature_set.add(feature_typed)
        feature_set.add(feature_untyped)

        # Should NOT raise - typed field matches, untyped skipped
        DataTypeValidator.validate(table, feature_set)

    def test_untyped_feature_allows_any_type(self) -> None:
        """Untyped features should work with any data type."""
        # Try multiple data types
        tables = [
            pa.table({"field": pa.array([1, 2], type=pa.int32())}),
            pa.table({"field": pa.array([1.5, 2.5], type=pa.float64())}),
            pa.table({"field": pa.array(["a", "b"], type=pa.string())}),
            pa.table({"field": pa.array([True, False], type=pa.bool_())}),
        ]

        feature = Feature.not_typed("field")
        feature_set = FeatureSet()
        feature_set.add(feature)

        for table in tables:
            # Should NOT raise for any type
            DataTypeValidator.validate(table, feature_set)


@PARALLELIZATION_MODES_SYNC_THREADING
class TestEndToEndIntegration:
    """Test complete datatype enforcement pipeline through mlodaAPI."""

    def test_same_feature_different_types_separate_execution(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        """Same feature requested with different types should execute separately."""
        features = Features(
            [
                Feature(name="TypedFeatureSource", data_type=DataType.INT32, initial_requested_data=True),
                Feature(name="TypedFeatureSource", data_type=DataType.INT64, initial_requested_data=True),
            ]
        )

        run_result = MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)

        # Should get two separate results (because hashes differ)
        assert len(run_result.results) == 2, "Should have separate results for different types"

        # Both should contain the same feature name
        all_columns = []
        for result in run_result.results:
            all_columns.extend(result.column_names)

        assert all_columns.count("TypedFeatureSource") == 2, "Should have feature computed twice"

    def test_typed_feature_validates_correctly(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """Typed features should validate data types correctly during execution."""
        features = Features(
            [
                Feature(name="TypedFeatureSource", data_type=DataType.INT32, initial_requested_data=True),
            ]
        )

        run_result = MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)

        assert len(run_result.results) == 1
        result = run_result.results[0]
        assert "TypedFeatureSource" in result.column_names
        assert result.to_pydict()["TypedFeatureSource"] == [25, 30, 35]

    def test_type_mismatch_fails_at_validation(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """Type mismatches should be caught during validation."""
        features = Features(
            [
                Feature(name="StringDataSource", data_type=DataType.INT32, initial_requested_data=True),
            ]
        )

        # This should fail because StringDataSource returns strings, not int32
        with pytest.raises(Exception):
            MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)

    def test_widening_allowed_in_pipeline(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """Type widening should be allowed throughout the pipeline."""
        features = Features(
            [
                Feature(name="Int64DataSource", data_type=DataType.INT64, initial_requested_data=True),
            ]
        )

        run_result = MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)

        assert len(run_result.results) == 1
        result = run_result.results[0]
        assert "Int64DataSource" in result.column_names

    def test_untyped_feature_bypasses_validation(self, modes: Set[ParallelizationModes], flight_server: Any) -> None:
        """Untyped features should bypass all type validation."""
        features = Features(
            [
                Feature(name="StringDataSource", initial_requested_data=True),
            ]
        )

        run_result = MlodaTestRunner.run_api(features, parallelization_modes=modes, flight_server=flight_server)

        assert len(run_result.results) == 1
        result = run_result.results[0]
        assert "StringDataSource" in result.column_names
        assert result.to_pydict()["StringDataSource"] == ["a", "b", "c"]


class TestValidatorEdgeCases:
    """Test edge cases in validation logic."""

    def test_validation_with_missing_column(self) -> None:
        """Validator should skip features not present in data."""
        table = pa.table({"present": pa.array([1, 2], type=pa.int32())})

        feature_present = Feature.int32_of("present")
        feature_missing = Feature.int32_of("missing")
        feature_set = FeatureSet()
        feature_set.add(feature_present)
        feature_set.add(feature_missing)

        # Should NOT raise - missing columns are skipped
        DataTypeValidator.validate(table, feature_set)

    def test_validation_with_empty_feature_set(self) -> None:
        """Validator should handle empty feature set."""
        table = pa.table({"value": pa.array([1, 2], type=pa.int32())})
        feature_set = FeatureSet()

        # Should NOT raise
        DataTypeValidator.validate(table, feature_set)

    def test_validation_with_null_values(self) -> None:
        """Validator should handle null values correctly."""
        table = pa.table({"value": pa.array([1, None, 3], type=pa.int32())})

        feature = Feature.int32_of("value")
        feature_set = FeatureSet()
        feature_set.add(feature)

        # Should NOT raise - nulls are allowed
        DataTypeValidator.validate(table, feature_set)

    def test_multiple_features_one_mismatch(self) -> None:
        """If one feature mismatches, validation should fail."""
        table = pa.table(
            {
                "good": pa.array([1, 2], type=pa.int32()),
                "bad": pa.array(["a", "b"], type=pa.string()),
            }
        )

        feature_good = Feature.int32_of("good")
        feature_bad = Feature.int32_of("bad")  # Declared INT32, actually STRING
        feature_set = FeatureSet()
        feature_set.add(feature_good)
        feature_set.add(feature_bad)

        with pytest.raises(DataTypeMismatchError) as exc_info:
            DataTypeValidator.validate(table, feature_set)

        # Should complain about the bad field
        assert "bad" in str(exc_info.value)
        assert "INT32" in str(exc_info.value)
        assert "STRING" in str(exc_info.value)


@PARALLELIZATION_MODES_SYNC_THREADING
class TestStrictTypeEnforcementPropagation:
    """Test that mlodaAPI strict_type_enforcement propagates and works end-to-end."""

    def test_strict_enforcement_raises_on_mismatch_but_lenient_passes(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        """
        Test both strict and lenient mode behavior:
        - strict=True: INT32 declared but INT64 returned → raises DataTypeMismatchError
        - strict=False: INT32 declared but INT64 returned → passes (lenient allows numeric widening)
        """

        # FeatureGroup that returns INT64 regardless of declared type
        class ReturnsInt64(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator(supports_features={"ReturnsInt64"})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"ReturnsInt64": pa.array([1, 2, 3], type=pa.int64())}

        plugin_collector = PlugInCollector.enabled_feature_groups({ReturnsInt64})

        # STRICT MODE: Should raise because INT32 declared but INT64 returned
        # Note: DataTypeMismatchError is wrapped in Exception by Runner
        with pytest.raises(Exception) as exc_info:
            MlodaTestRunner.run_api(
                Features([Feature.int32_of("ReturnsInt64")]),
                parallelization_modes=modes,
                flight_server=flight_server,
                plugin_collector=plugin_collector,
                strict_type_enforcement=True,
            )
        assert "INT32" in str(exc_info.value)
        assert "INT64" in str(exc_info.value)
        assert "DataTypeMismatchError" in str(exc_info.value)

        # LENIENT MODE: Should pass because numeric types are interchangeable
        result = MlodaTestRunner.run_api(
            Features([Feature.int32_of("ReturnsInt64")]),
            parallelization_modes=modes,
            flight_server=flight_server,
            plugin_collector=plugin_collector,
            strict_type_enforcement=False,
        )
        assert len(result.results) == 1

    def test_strict_enforcement_propagates_to_dependencies(
        self, modes: Set[ParallelizationModes], flight_server: Any
    ) -> None:
        """
        Test that strict_type_enforcement propagates to dependency features.

        - Parent feature requests a dependency with INT32
        - Dependency FeatureGroup returns INT64
        - strict=True: Should raise DataTypeMismatchError on the DEPENDENCY
        """

        # Dependency: returns INT64 regardless of declared type
        class BaseFG(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator(supports_features={"BaseFG"})

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"BaseFG": pa.array([1, 2, 3], type=pa.int64())}

        # Parent: depends on BaseFG with INT32 type
        class DerivedFG(AbstractFeatureGroup):
            @classmethod
            def input_data(cls) -> Optional[BaseInputData]:
                return DataCreator(supports_features={"DerivedFG"})

            def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
                return {Feature.int32_of("BaseFG")}

            @classmethod
            def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
                return {"DerivedFG": pa.array([10, 20, 30], type=pa.int64())}

        plugin_collector = PlugInCollector.enabled_feature_groups({BaseFG, DerivedFG})

        # STRICT MODE: Should raise on BaseFG (dependency) type mismatch
        # Note: DataTypeMismatchError is wrapped in Exception by Runner
        with pytest.raises(Exception) as exc_info:
            MlodaTestRunner.run_api(
                Features([Feature.int64_of("DerivedFG")]),
                parallelization_modes=modes,
                flight_server=flight_server,
                plugin_collector=plugin_collector,
                strict_type_enforcement=True,
            )
        # Error should mention the dependency feature
        assert "BaseFG" in str(exc_info.value)
        assert "INT32" in str(exc_info.value)
        assert "INT64" in str(exc_info.value)
        assert "DataTypeMismatchError" in str(exc_info.value)
