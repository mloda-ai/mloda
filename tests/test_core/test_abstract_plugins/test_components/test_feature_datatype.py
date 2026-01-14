"""Tests for Feature data_type handling and similarity properties.

These tests validate:
1. has_similarity_properties() includes data_type when explicitly set
2. base_similarity_properties() excludes data_type for lenient grouping
3. is_different_data_type() correctly identifies features with different data types
"""

from mloda.user import Feature


def test_has_similarity_properties_includes_data_type() -> None:
    """Test that has_similarity_properties() includes data_type when explicitly set.

    Two features with the same name and options but different explicit data types
    should have DIFFERENT similarity property hashes.

    This enables separating features by type at plan time, so INT32 and INT64
    features can be processed in different execution groups.
    """
    feature_int32 = Feature.int32_of("age")
    feature_int64 = Feature.int64_of("age")

    hash_int32 = feature_int32.has_similarity_properties()
    hash_int64 = feature_int64.has_similarity_properties()

    assert hash_int32 != hash_int64, (
        "Features with different explicit data types should have different hashes. "
        f"Expected different but got: int32={hash_int32}, int64={hash_int64}"
    )


def test_has_similarity_properties_none_type_uses_base_hash() -> None:
    """Test that features with data_type=None use the base hash (no data_type).

    None-typed features (like index columns) should have the same hash as
    their base properties, enabling lenient grouping via the execution plan.
    """
    feature_typed = Feature.int32_of("amount")
    feature_untyped = Feature.not_typed("id")

    # Untyped feature should use base hash
    assert feature_untyped.has_similarity_properties() == feature_untyped.base_similarity_properties()

    # Typed feature should have different hash than its base (includes data_type)
    assert feature_typed.has_similarity_properties() != feature_typed.base_similarity_properties()


def test_base_similarity_properties_excludes_data_type() -> None:
    """Test that base_similarity_properties() excludes data_type.

    This allows None-typed features to match typed features during lenient grouping.
    """
    feature_int32 = Feature.int32_of("age")
    feature_int64 = Feature.int64_of("age")
    feature_untyped = Feature.not_typed("age")

    # All should have the same base hash (excludes data_type)
    base_int32 = feature_int32.base_similarity_properties()
    base_int64 = feature_int64.base_similarity_properties()
    base_untyped = feature_untyped.base_similarity_properties()

    assert base_int32 == base_int64 == base_untyped, (
        "All features with same options should have same base hash. "
        f"Got: int32={base_int32}, int64={base_int64}, untyped={base_untyped}"
    )


def test_is_different_data_type_correct_behavior() -> None:
    """Test that is_different_data_type() correctly identifies data type differences.

    The method should return True when two features have:
    - Same name
    - Different data types

    This test currently FAILS due to bug at line 167:
    - Incorrectly compares self.name with other.options
    - Incorrectly uses == instead of != for data_type comparison
    """
    # Case 1: Same name, different data types -> should return True
    feature_int32 = Feature.int32_of("age")
    feature_int64 = Feature.int64_of("age")

    assert feature_int32.is_different_data_type(feature_int64) is True, (
        "Features with same name but different data types should be identified as different. "
        f"feature_int32.data_type={feature_int32.data_type}, "
        f"feature_int64.data_type={feature_int64.data_type}"
    )

    # Case 2: Same name, same data type -> should return False
    feature_int32_a = Feature.int32_of("age")
    feature_int32_b = Feature.int32_of("age")

    assert feature_int32_a.is_different_data_type(feature_int32_b) is False, (
        "Features with same name and same data type should NOT be identified as different. "
        f"Both have data_type={feature_int32_a.data_type}"
    )

    # Case 3: Different names, different data types -> should return False
    # (not "different data type" in the sense this method intends to capture)
    feature_age = Feature.int32_of("age")
    feature_height = Feature.int64_of("height")

    assert feature_age.is_different_data_type(feature_height) is False, (
        "Features with different names should return False regardless of data types"
    )
