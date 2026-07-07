"""Tests for Feature data_type handling and similarity properties.

These tests validate:
1. similarity_hash() includes data_type when explicitly set
2. base_similarity_hash() excludes data_type for lenient grouping
3. is_different_data_type() correctly identifies features with different data types
4. similarity_hash() and base_similarity_hash() incorporate ONLY the context entries
   whose key is in options.inherited_context_keys (context DELIVERED via
   Options.inherit_from). Features differing in plain author-set context keep equal
   hashes (the documented "context does not affect splitting" contract); features
   whose INHERITED context values differ never share an execution-plan FeatureSet.
"""

from mloda.user import DataType, Feature, Options


def test_similarity_hash_includes_data_type() -> None:
    """Test that similarity_hash() includes data_type when explicitly set.

    Two features with the same name and options but different explicit data types
    should have DIFFERENT similarity property hashes.

    This enables separating features by type at plan time, so INT32 and INT64
    features can be processed in different execution groups.
    """
    feature_int32 = Feature.int32_of("age")
    feature_int64 = Feature.int64_of("age")

    hash_int32 = feature_int32.similarity_hash()
    hash_int64 = feature_int64.similarity_hash()

    assert hash_int32 != hash_int64, (
        "Features with different explicit data types should have different hashes. "
        f"Expected different but got: int32={hash_int32}, int64={hash_int64}"
    )


def test_similarity_hash_none_type_uses_base_hash() -> None:
    """Test that features with data_type=None use the base hash (no data_type).

    None-typed features (like index columns) should have the same hash as
    their base properties, enabling lenient grouping via the execution plan.
    """
    feature_typed = Feature.int32_of("amount")
    feature_untyped = Feature.not_typed("id")

    # Untyped feature should use base hash
    assert feature_untyped.similarity_hash() == feature_untyped.base_similarity_hash()

    # Typed feature should have different hash than its base (includes data_type)
    assert feature_typed.similarity_hash() != feature_typed.base_similarity_hash()


def test_base_similarity_hash_excludes_data_type() -> None:
    """Test that base_similarity_hash() excludes data_type.

    This allows None-typed features to match typed features during lenient grouping.
    """
    feature_int32 = Feature.int32_of("age")
    feature_int64 = Feature.int64_of("age")
    feature_untyped = Feature.not_typed("age")

    # All should have the same base hash (excludes data_type)
    base_int32 = feature_int32.base_similarity_hash()
    base_int64 = feature_int64.base_similarity_hash()
    base_untyped = feature_untyped.base_similarity_hash()

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


def test_similarity_hash_returns_int() -> None:
    """Verify similarity_hash() and base_similarity_hash() exist and return int."""
    feature = Feature.int32_of("age")

    assert isinstance(feature.similarity_hash(), int)
    assert isinstance(feature.base_similarity_hash(), int)


def test_old_method_names_removed() -> None:
    """Verify the old method names no longer exist on Feature."""
    feature = Feature.int32_of("age")

    assert not hasattr(feature, "has_similarity_properties")
    assert not hasattr(feature, "base_similarity_properties")


def _options_with_inherited_tenant(tenant: str) -> Options:
    """Build child Options whose "tenant" context entry was DELIVERED via inherit_from.

    The consumer carries the tenant in its context; the child pulls it with
    inherit_context_keys, so options.inherited_context_keys provenance is real,
    exactly as the engine merge produces it for declared input features.
    """
    consumer = Options(context={"tenant": tenant})
    child = Options(group={"data_source": "prod"})
    child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))
    return child


def _typed_feature_with_inherited_tenant(tenant: str) -> Feature:
    return Feature("ctx_hash_feature", options=_options_with_inherited_tenant(tenant), data_type=DataType.INT32)


def test_similarity_hash_differs_when_inherited_context_differs() -> None:
    """Two features identical except for an INHERITED context value must have DIFFERENT similarity hashes.

    Options.__hash__ is group-only by design, but the execution plan groups features
    into FeatureSets by similarity_hash(). If inherited context is excluded there, two
    same-named features that pulled different context values (tenant via
    inherit_context_keys) collapse into one FeatureSet and one consumer silently
    receives the other's data.
    """
    feature_acme = _typed_feature_with_inherited_tenant("acme")
    feature_beta = _typed_feature_with_inherited_tenant("beta")

    hash_acme = feature_acme.similarity_hash()
    hash_beta = feature_beta.similarity_hash()

    assert hash_acme != hash_beta, (
        "similarity_hash() must incorporate INHERITED context entries: features whose "
        "inherited context values differ must not share a FeatureSet. "
        f"Got equal hashes: acme={hash_acme}, beta={hash_beta}"
    )


def test_base_similarity_hash_differs_when_inherited_context_differs() -> None:
    """base_similarity_hash() (lenient grouping of None-typed features) must also see inherited context.

    Otherwise a None-typed feature that inherited {"tenant": "beta"} joins the typed
    group of a feature that inherited {"tenant": "acme"} during the second grouping pass.
    """
    feature_acme = Feature("ctx_hash_feature", options=_options_with_inherited_tenant("acme"))
    feature_beta = Feature("ctx_hash_feature", options=_options_with_inherited_tenant("beta"))

    base_acme = feature_acme.base_similarity_hash()
    base_beta = feature_beta.base_similarity_hash()

    assert base_acme != base_beta, (
        "base_similarity_hash() must incorporate INHERITED context entries. "
        f"Got equal hashes: acme={base_acme}, beta={base_beta}"
    )


def test_similarity_hashes_equal_for_plain_context_difference() -> None:
    """Regression pin (OLD behavior): plain author-set context does NOT affect the hashes.

    Context without inherit_from provenance is metadata by contract ("context does not
    affect splitting", pinned by the chainer context suites). Only INHERITED context
    participates in grouping.
    """
    feature_acme = Feature(
        "ctx_hash_feature",
        options=Options(group={"data_source": "prod"}, context={"tenant": "acme"}),
        data_type=DataType.INT32,
    )
    feature_beta = Feature(
        "ctx_hash_feature",
        options=Options(group={"data_source": "prod"}, context={"tenant": "beta"}),
        data_type=DataType.INT32,
    )

    assert feature_acme.similarity_hash() == feature_beta.similarity_hash(), (
        "Plain (non-inherited) context differences must keep EQUAL similarity hashes; "
        "only inherited context participates in grouping."
    )
    assert feature_acme.base_similarity_hash() == feature_beta.base_similarity_hash(), (
        "Plain (non-inherited) context differences must keep EQUAL base similarity hashes."
    )


def test_similarity_hashes_equal_for_equal_inherited_context() -> None:
    """Regression pin: inherited context with EQUAL values still hashes equal.

    Incorporating inherited context must not break grouping of genuinely identical features.
    """
    feature_a = _typed_feature_with_inherited_tenant("acme")
    feature_b = _typed_feature_with_inherited_tenant("acme")

    assert feature_a.similarity_hash() == feature_b.similarity_hash(), (
        "Features with identical group options, inherited context, compute framework, and "
        "data_type must keep equal similarity hashes so they still share one FeatureSet."
    )
    assert feature_a.base_similarity_hash() == feature_b.base_similarity_hash(), (
        "Features with identical group options and inherited context must keep equal base similarity hashes."
    )


def test_plain_context_difference_alongside_equal_inherited_context_hashes_equal() -> None:
    """Pin the ONLY-inherited scope: with equal inherited context, a differing plain
    author-set context key must not change the hashes."""
    options_a = _options_with_inherited_tenant("acme")
    options_a.add_to_context("annotation", "left")
    options_b = _options_with_inherited_tenant("acme")
    options_b.add_to_context("annotation", "right")

    feature_a = Feature("ctx_hash_feature", options=options_a, data_type=DataType.INT32)
    feature_b = Feature("ctx_hash_feature", options=options_b, data_type=DataType.INT32)

    assert feature_a.similarity_hash() == feature_b.similarity_hash(), (
        "Only INHERITED context entries participate in similarity_hash(); a plain "
        "author-set context key alongside equal inherited context must not split."
    )
    assert feature_a.base_similarity_hash() == feature_b.base_similarity_hash()


def test_similarity_hashes_ignore_inherited_context_insertion_order() -> None:
    """Delivery order of inherited context keys must not affect the hashes."""
    consumer = Options(context={"tenant": "acme", "region": "eu"})

    options_ab = Options(group={"data_source": "prod"})
    options_ab.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))
    options_ab.inherit_from(consumer, inherit_context_keys=frozenset({"region"}))

    options_ba = Options(group={"data_source": "prod"})
    options_ba.inherit_from(consumer, inherit_context_keys=frozenset({"region"}))
    options_ba.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))

    feature_ab = Feature("ctx_hash_feature", options=options_ab, data_type=DataType.INT32)
    feature_ba = Feature("ctx_hash_feature", options=options_ba, data_type=DataType.INT32)

    assert feature_ab.similarity_hash() == feature_ba.similarity_hash(), (
        "similarity_hash() must be independent of inherited-context delivery order."
    )
    assert feature_ab.base_similarity_hash() == feature_ba.base_similarity_hash(), (
        "base_similarity_hash() must be independent of inherited-context delivery order."
    )


def test_options_hash_and_eq_stay_group_only() -> None:
    """Regression pin: Options.__hash__/__eq__ keep group-only semantics.

    Feature resolution semantics are unchanged; only the Feature-level similarity
    hashes incorporate inherited context.
    """
    consumer_acme = Options(context={"tenant": "acme"})
    consumer_beta = Options(context={"tenant": "beta"})
    options_acme = Options(group={"data_source": "prod"})
    options_acme.inherit_from(consumer_acme, inherit_context_keys=frozenset({"tenant"}))
    options_beta = Options(group={"data_source": "prod"})
    options_beta.inherit_from(consumer_beta, inherit_context_keys=frozenset({"tenant"}))

    assert hash(options_acme) == hash(options_beta)
    assert options_acme == options_beta
