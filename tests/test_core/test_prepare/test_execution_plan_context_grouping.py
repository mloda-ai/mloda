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


def _options_with_propagated_env(env: str) -> Options:
    """Child Options whose "env" context entry was DELIVERED via the caller-side PUSH
    (consumer.propagate_context_keys), so options.inherited_context_keys provenance is real."""
    consumer = Options(context={"env": env}, propagate_context_keys=frozenset({"env"}))
    child = Options(group={"data_source": "prod"})
    child.inherit_from(consumer)
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


def test_none_typed_joins_first_of_two_data_type_only_groups() -> None:
    """Characterization pin for the O(1) reverse-map optimization (issue #622).

    Two typed features sharing everything (name, options, compute framework, inherited
    context) but differing ONLY in data_type form two distinct groups (data_type is part
    of similarity_hash). A None-typed feature with the same base properties shares the
    SAME base_similarity_hash as both typed groups, so the current code joins the FIRST
    matching group in insertion order and breaks. The optimization must preserve this
    first-match semantics: exactly 2 groups survive, one with 2 members, the other with 1.
    """
    typed_int = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.INT32,
    )
    typed_float = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
        data_type=DataType.FLOAT,
    )
    none_typed = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
    )

    groups = _group({typed_int, typed_float, none_typed})

    assert len(groups) == 2, (
        "Two typed features differing only in data_type must stay in distinct groups, and the "
        f"None-typed feature must join one existing group rather than form its own. Got {len(groups)}: {groups}"
    )

    sizes = sorted(len(group) for group in groups.values())
    assert sizes == [1, 2], (
        "The None-typed feature must join exactly one of the two typed groups (first-match "
        f"semantics), yielding group sizes [1, 2]. Got {sizes}: {groups}"
    )

    joined_group = next(group for group in groups.values() if none_typed in group)
    assert len(joined_group) == 2 and (typed_int in joined_group) ^ (typed_float in joined_group), (
        "The None-typed feature must join exactly one typed group, keeping the two typed features in separate groups."
    )


def test_two_none_typed_features_without_typed_group_share_one_group() -> None:
    """Characterization pin (issue #622): a newly created None-typed group is reused.

    When two None-typed features have identical base properties and there is NO matching
    typed group, the first opens a new group keyed by its base hash and the second must
    join that same group. Expect a single group holding both features.
    """
    none_typed_a = Feature(
        "tenant_scoped_source",
        options=_options_with_inherited_tenant("acme"),
    )
    none_typed_b = Feature(
        "tenant_scoped_index",
        options=_options_with_inherited_tenant("acme"),
    )

    groups = _group({none_typed_a, none_typed_b})

    assert len(groups) == 1, (
        "Two None-typed features with identical base properties and no typed group must share one "
        f"group (the newly created None-typed group is reused). Got {len(groups)}: {groups}"
    )
    only_group = next(iter(groups.values()))
    assert none_typed_a in only_group and none_typed_b in only_group


def test_split_context_hashable_empty_cases_return_empty_tuple() -> None:
    """Characterization pin (issue #622): the empty split-context canonicalizes to ().

    The optimization introduces an empty-case short-circuit; this pins its expected value.
    With EMPTY split_keys the result is the empty-tuple canonical value (), and for a feature
    carrying context that contains NONE of the split keys the result is also ()."""
    feature = Feature(
        "tenant_scoped_source",
        options=Options(group={"data_source": "prod"}, context={"tenant": "acme"}),
    )

    assert feature._split_context_hashable(frozenset()) == (), (
        "Empty split_keys must canonicalize to the empty tuple ()."
    )
    assert feature._split_context_hashable(frozenset({"region"})) == (), (
        "A feature carrying none of the split keys in its context must canonicalize to ()."
    )


def test_features_differing_in_propagated_context_land_in_different_groups() -> None:
    """Same name, group options, compute framework, and data_type but different PROPAGATED
    (pushed via consumer.propagate_context_keys) context must produce two separate groups."""
    feature_prod = Feature(
        "env_scoped_source",
        options=_options_with_propagated_env("prod"),
        data_type=DataType.INT32,
    )
    feature_staging = Feature(
        "env_scoped_source",
        options=_options_with_propagated_env("staging"),
        data_type=DataType.INT32,
    )

    groups = _group({feature_prod, feature_staging})

    assert len(groups) == 2, (
        "Features differing in PROPAGATED (pushed) context must not share a FeatureSet. "
        f"Expected 2 groups, got {len(groups)}: {groups}"
    )
    for group in groups.values():
        assert len(group) == 1


def test_features_with_identical_propagated_context_share_one_group() -> None:
    """Regression pin: identical group options and identical PROPAGATED context still group
    together (equal propagated values share one computation)."""
    feature_a = Feature(
        "env_scoped_source",
        options=_options_with_propagated_env("prod"),
        data_type=DataType.INT32,
    )
    feature_b = Feature(
        "env_scoped_other",
        options=_options_with_propagated_env("prod"),
        data_type=DataType.INT32,
    )

    groups = _group({feature_a, feature_b})

    assert len(groups) == 1, (
        "Features with identical group options and identical propagated context must share one "
        f"FeatureSet. Got {len(groups)} groups: {groups}"
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
