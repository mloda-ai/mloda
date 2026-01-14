import inspect
import pyarrow as pa

from mloda.user import Feature
from mloda.provider import FeatureSet
from mloda_plugins.feature_group.input_data.read_dbs.sqlite import SQLITEReader


class TestSQLiteTypeAwareness:
    """Tests for SQLITEReader type awareness - using declared data_type instead of inference."""

    def test_read_as_pa_data_signature_accepts_features(self) -> None:
        """Test that read_as_pa_data accepts features parameter."""
        sig = inspect.signature(SQLITEReader.read_as_pa_data)
        params = list(sig.parameters.keys())
        assert "features" in params, "read_as_pa_data should accept 'features' parameter"

    def test_read_as_pa_data_uses_declared_type(self) -> None:
        """Test that read_as_pa_data uses declared data_type instead of inferring."""
        # SQLite returns all integers the same way, but we want INT64 specifically
        result = [(1, 25), (2, 30)]  # Mock DB result
        column_names = ["id", "age"]

        # Create features with declared types
        feature_id = Feature.int32_of("id")
        feature_age = Feature.int64_of("age")  # Explicitly INT64

        feature_set = FeatureSet()
        feature_set.add(feature_id)
        feature_set.add(feature_age)

        # Call read_as_pa_data with features
        table = SQLITEReader.read_as_pa_data(result, column_names, feature_set)

        # Assert age column is INT64 as declared, not inferred
        assert pa.types.is_int64(table.schema.field("age").type), "age should be INT64 as declared"
        assert pa.types.is_int32(table.schema.field("id").type), "id should be INT32 as declared"

    def test_read_as_pa_data_falls_back_to_inference(self) -> None:
        """Test that read_as_pa_data infers type when feature has no data_type."""
        result = [(1, "hello"), (2, "world")]
        column_names = ["id", "name"]

        # Features without explicit data_type
        feature_id = Feature.not_typed("id")
        feature_name = Feature.not_typed("name")

        feature_set = FeatureSet()
        feature_set.add(feature_id)
        feature_set.add(feature_name)

        table = SQLITEReader.read_as_pa_data(result, column_names, feature_set)

        # Should infer types from data
        assert pa.types.is_integer(table.schema.field("id").type), "id should be inferred as integer"
        assert pa.types.is_string(table.schema.field("name").type), "name should be inferred as string"

    def test_read_as_pa_data_mixed_declared_and_inferred(self) -> None:
        """Test that read_as_pa_data handles mix of declared and inferred types."""
        result = [(1, 42, "test"), (2, 84, "data")]
        column_names = ["id", "value", "label"]

        # Mix of typed and untyped features
        feature_id = Feature.int64_of("id")  # Declared
        feature_value = Feature.not_typed("value")  # Will be inferred
        feature_label = Feature.str_of("label")  # Declared

        feature_set = FeatureSet()
        feature_set.add(feature_id)
        feature_set.add(feature_value)
        feature_set.add(feature_label)

        table = SQLITEReader.read_as_pa_data(result, column_names, feature_set)

        # Verify mixed behavior
        assert pa.types.is_int64(table.schema.field("id").type), "id should be INT64 as declared"
        assert pa.types.is_integer(table.schema.field("value").type), "value should be inferred as integer"
        assert pa.types.is_string(table.schema.field("label").type), "label should be STRING as declared"
