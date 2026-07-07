"""Tests for inherited-context-aware FeatureSet grouping in the execution plan.

ExecutionPlan.group_features_by_compute_framework_and_options buckets features
into FeatureSets via Feature.similarity_hash() / base_similarity_hash(). Features
whose INHERITED context differs (each pulled a different tenant value via
Options.inherit_from with inherit_context_keys, or received it via the consumer
push) must NOT share a FeatureSet, otherwise calculate_feature runs once and one
consumer silently receives the other's data.

Plain author-set context stays metadata: it must NOT affect grouping (the
documented "context does not affect splitting" contract, pinned by the chainer
context suites). Only context entries recorded in options.inherited_context_keys
participate.
"""

from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.user import DataType, Feature, Options


def _group(features: set[Feature]) -> dict[int, set[Feature]]:
    return ExecutionPlan().group_features_by_compute_framework_and_options(features)


def _options_with_inherited_tenant(tenant: str) -> Options:
    """Child Options whose "tenant" context entry was DELIVERED via inherit_from,
    so options.inherited_context_keys provenance is real."""
    consumer = Options(context={"tenant": tenant})
    child = Options(group={"data_source": "prod"})
    child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))
    return child


def test_features_differing_in_inherited_context_land_in_different_groups() -> None:
    """Same name, group options, compute framework, and data_type but different
    INHERITED context must produce two separate groups."""
    feature_acme = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    feature_beta = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("beta"),
        data_type=DataType.INT32,
    )

    groups = _group({feature_acme, feature_beta})

    assert len(groups) == 2, (
        "Features differing in INHERITED context must not share a FeatureSet. "
        f"Expected 2 groups, got {len(groups)}: {groups}"
    )
    for group in groups.values():
        assert len(group) == 1


def test_features_differing_only_in_plain_context_share_one_group() -> None:
    """Regression pin (OLD behavior): plain author-set context does NOT split.

    Context without inherit_from provenance is metadata by contract; features
    differing only in such context keep sharing one FeatureSet."""
    feature_acme = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "acme"}),
        data_type=DataType.INT32,
    )
    feature_beta = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "beta"}),
        data_type=DataType.INT32,
    )

    groups = _group({feature_acme, feature_beta})

    assert len(groups) == 1, (
        "Plain (non-inherited) context differences must keep features in ONE FeatureSet; "
        f"only inherited context splits. Got {len(groups)} groups: {groups}"
    )


def test_lenient_none_typed_pass_respects_inherited_context() -> None:
    """The second (lenient) pass assigns None-typed features to typed groups by
    base_similarity_hash. A None-typed feature must only join a typed group whose
    INHERITED context matches; an inherited-context mismatch must open a new group."""
    typed_acme = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    none_typed_acme = Feature(
        "tenant_scoped_index",
        options=_options_with_inherited_tenant("acme"),
    )
    none_typed_beta = Feature(
        "tenant_scoped_index",
        options=_options_with_inherited_tenant("beta"),
    )

    groups = _group({typed_acme, none_typed_acme, none_typed_beta})

    assert len(groups) == 2, (
        "The lenient None-typed pass must respect inherited context: the beta feature "
        f"must not join the acme group. Expected 2 groups, got {len(groups)}: {groups}"
    )

    acme_group = next(group for group in groups.values() if typed_acme in group)
    assert none_typed_acme in acme_group, (
        "Regression pin: a None-typed feature with MATCHING inherited context must still "
        "join the typed group via the lenient pass."
    )
    assert none_typed_beta not in acme_group, (
        "A None-typed feature whose inherited context differs must not join the typed group."
    )


def test_same_value_via_inherited_and_plain_resolves_to_one_group() -> None:
    """Same effective value delivered via inheritance vs directly must share one FeatureSet."""
    inherited_acme = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    plain_acme = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "acme"}),
        data_type=DataType.INT32,
    )

    groups = _group({inherited_acme, plain_acme})

    assert len(groups) == 1, (
        "Identical effective config delivered via inheritance vs directly must share one "
        f"FeatureSet (one computation), not split on provenance. Got {len(groups)}: {groups}"
    )


def test_plain_value_splits_from_inherited_value_when_key_is_a_split_key() -> None:
    """A key inherited by one feature splits every feature in scope by value: a plain
    tenant=beta feature stays isolated from the tenant=acme features. Without the
    split-key union this outcome depends on which duplicate survives dedup."""
    inherited_acme = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    plain_acme = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "acme"}),
        data_type=DataType.INT32,
    )
    plain_beta = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "beta"}),
        data_type=DataType.INT32,
    )

    groups = _group({inherited_acme, plain_acme, plain_beta})

    assert len(groups) == 2, (
        "A split key must isolate by value: the acme features share one FeatureSet and the "
        f"beta feature stays separate. Expected 2 groups, got {len(groups)}: {groups}"
    )
    beta_group = next(group for group in groups.values() if plain_beta in group)
    assert inherited_acme not in beta_group and plain_acme not in beta_group, (
        "The plain tenant=beta feature must not collapse into the tenant=acme FeatureSet."
    )


def test_features_with_identical_inherited_context_share_one_group() -> None:
    """Regression pin: identical group options and identical inherited context still group together."""
    feature_a = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    feature_b = Feature(
        "tenant_scoped_other",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )

    groups = _group({feature_a, feature_b})

    assert len(groups) == 1, (
        "Features with identical group options, inherited context, compute framework, and "
        f"data_type must keep sharing one FeatureSet. Got {len(groups)} groups: {groups}"
    )
