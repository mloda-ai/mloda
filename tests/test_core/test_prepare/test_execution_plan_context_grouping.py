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

import itertools
from collections.abc import Callable
from typing import Any

import pytest

from mloda.core.filter.global_filter import GlobalFilter
from mloda.core.prepare.execution_plan import ExecutionPlan
from mloda.provider import BaseArtifact, ComputeFramework, FeatureGroup
from mloda.user import DataType, Feature, FeatureName, FilterType, Options, SingleFilter
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


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


def test_grouping_is_callable_on_the_class() -> None:
    features = {Feature("static_grouping_a", data_type=DataType.INT32), Feature("static_grouping_b")}

    groups = ExecutionPlan.group_features_by_compute_framework_and_options(features)

    assert sum(len(group) for group in groups.values()) == 2


class MergeProbeFG(FeatureGroup):
    """Plain feature group (no artifact) for the same-framework merge helper tests."""


class MergeArtifactProbeFG(FeatureGroup):
    """Feature group with an artifact, which must never merge."""

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        return BaseArtifact


def _chosen(feature: Feature, framework: type[ComputeFramework] = PyArrowTable) -> Feature:
    feature.chosen_compute_framework = framework
    return feature


def _merge(
    groups: dict[int, set[Feature]],
    feature_group: type[FeatureGroup] = MergeProbeFG,
    global_filter: GlobalFilter | None = None,
    can_merge: Callable[[set[Feature]], bool] | None = None,
) -> dict[int, set[Feature]]:
    return ExecutionPlan._merge_same_framework_groups(feature_group, groups, global_filter, can_merge)


def _names_per_group(groups: dict[int, set[Feature]]) -> list[list[str]]:
    return sorted(sorted(str(f.name) for f in group) for group in groups.values())


MergeCase = tuple[dict[int, set[Feature]], dict[str, Any]]


def _case(*features: Feature, **merge_kwargs: Any) -> MergeCase:
    return _group(set(features)), merge_kwargs


def _filter_for(*names: str, same: bool = True) -> GlobalFilter:
    global_filter = GlobalFilter()
    for name in names:
        column = "merge_col" if same else f"merge_col_{name}"
        single_filter = SingleFilter(Feature(column), FilterType.EQUAL, {"value": 1})
        global_filter.add_filter_to_collection(MergeProbeFG, FeatureName(name), single_filter)
    return global_filter


def _merges_data_types() -> MergeCase:
    return _case(_chosen(Feature("m_a", data_type=DataType.INT32)), _chosen(Feature("m_b", data_type=DataType.INT64)))


def _merges_allowed_sets() -> MergeCase:
    return _case(_chosen(Feature("m_a", compute_framework="PyArrowTable")), _chosen(Feature("m_b")))


def _merges_typed_and_untyped() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)), _chosen(Feature("m_b", compute_framework="PyArrowTable"))
    )


def _merges_identical_filters() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)),
        _chosen(Feature("m_b", data_type=DataType.INT64)),
        global_filter=_filter_for("m_a", "m_b"),
    )


def _merges_when_can_merge_accepts() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)),
        _chosen(Feature("m_b", data_type=DataType.INT64)),
        can_merge=lambda members: True,
    )


def _keeps_same_name_variants() -> MergeCase:
    return _case(_chosen(Feature("m_a", data_type=DataType.INT32)), _chosen(Feature("m_a", data_type=DataType.INT64)))


def _keeps_different_frameworks() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32), PyArrowTable),
        _chosen(Feature("m_b", data_type=DataType.INT64), PandasDataFrame),
    )


def _keeps_different_options() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", options={"unit": "x"}, data_type=DataType.INT32)),
        _chosen(Feature("m_b", options={"unit": "y"}, data_type=DataType.INT64)),
    )


def _keeps_different_split_context() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", options=_options_with_inherited_tenant("acme"), data_type=DataType.INT32)),
        _chosen(Feature("m_b", options=_options_with_inherited_tenant("beta"), data_type=DataType.INT64)),
    )


def _keeps_differing_filters() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)),
        _chosen(Feature("m_b", data_type=DataType.INT64)),
        global_filter=_filter_for("m_a"),
    )


def _keeps_artifact_group() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)),
        _chosen(Feature("m_b", data_type=DataType.INT64)),
        feature_group=MergeArtifactProbeFG,
    )


def _keeps_when_can_merge_rejects() -> MergeCase:
    return _case(
        _chosen(Feature("m_a", data_type=DataType.INT32)),
        _chosen(Feature("m_b", data_type=DataType.INT64)),
        can_merge=lambda members: False,
    )


@pytest.mark.parametrize(
    "build",
    [
        _merges_data_types,
        _merges_allowed_sets,
        _merges_typed_and_untyped,
        _merges_identical_filters,
        _merges_when_can_merge_accepts,
    ],
)
def test_merge_same_framework_groups_merges(build: Callable[[], MergeCase]) -> None:
    groups, kwargs = build()
    assert len(groups) == 2

    merged = _merge(groups, **kwargs)

    assert _names_per_group(merged) == [["m_a", "m_b"]]


@pytest.mark.parametrize(
    "build, expected",
    [
        (_keeps_same_name_variants, [["m_a"], ["m_a"]]),
        (_keeps_different_frameworks, [["m_a"], ["m_b"]]),
        (_keeps_different_options, [["m_a"], ["m_b"]]),
        (_keeps_different_split_context, [["m_a"], ["m_b"]]),
        (_keeps_differing_filters, [["m_a"], ["m_b"]]),
        (_keeps_artifact_group, [["m_a"], ["m_b"]]),
        (_keeps_when_can_merge_rejects, [["m_a"], ["m_b"]]),
    ],
)
def test_merge_same_framework_groups_keeps_separate(build: Callable[[], MergeCase], expected: list[list[str]]) -> None:
    groups, kwargs = build()

    merged = _merge(groups, **kwargs)

    assert _names_per_group(merged) == expected


def test_merge_three_groups_is_deterministic_across_input_orders() -> None:
    x32 = _chosen(Feature("merge_x", data_type=DataType.INT32))
    x64 = _chosen(Feature("merge_x", data_type=DataType.INT64))
    y = _chosen(Feature("merge_y", compute_framework="PyArrowTable"))
    items = list(_group({x32, x64, y}).items())
    assert len(items) == 3

    outcomes = set()
    for order in itertools.permutations(items):
        merged = _merge(dict(order))
        assert len(merged) == 2
        with_y = next(group for group in merged.values() if y in group)
        outcomes.add(next(f.data_type for f in with_y if f.name == FeatureName("merge_x")))

    assert len(outcomes) == 1


def test_merge_tie_on_allowed_framework_sets_is_deterministic() -> None:
    """Same-name groups differing only in allowed frameworks: y merges with the first sorted set."""
    for _ in range(20):
        x_pandas = _chosen(Feature("merge_x", compute_framework="PandasDataFrame"))
        x_arrow = _chosen(Feature("merge_x", compute_framework="PyArrowTable"))
        y = _chosen(Feature("merge_y"))
        groups = _group({x_pandas, x_arrow, y})
        assert len(groups) == 3

        merged = _merge(groups)

        assert len(merged) == 2
        with_y = next(group for group in merged.values() if y in group)
        assert x_pandas in with_y
        assert x_arrow not in with_y
