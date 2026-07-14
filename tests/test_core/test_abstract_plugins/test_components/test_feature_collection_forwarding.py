"""
Tests for the forward-by-default collection merge of issue #579.

Encodes the target state where:
- Features.merge_options delegates to Options.inherit_from: input features inherit
  ALL consumer group options by default (the unspecified None sentinel and True are
  identical), except DefaultOptionKeys.in_features which never flows.
- Feature.forward_group=False isolates the input feature, a frozenset allowlist
  restricts the copy to the listed keys, and Feature.forward_group_exclude carves
  keys out of whatever forward_group would forward.
- Context never flows implicitly: only the child-side inherit_context_keys pull and
  the consumer-side propagate_context_keys push move context keys.
- FeatureChainParserMixin no longer stamps forward_group=True on its children; they
  stay at the None sentinel and the engine default does the forwarding.
- The old shield machinery stays deleted: DefaultOptionKeys.feature_chainer_parser_key,
  Options.update_with_protected_keys, and Options.forward_for_input_feature.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import (
    FeatureChainParserMixin,
)
from mloda.provider import DefaultOptionKeys, PropertySpec
from mloda.user import Feature
from mloda.user import FeatureName
from mloda.user import Features
from mloda.user import Options


def _merge(input_feature: Feature, consumer: Options) -> Feature:
    """Drive the collection-level merge exactly like the engine does (engine.py:348)."""
    features = Features([input_feature], child_options=consumer, child_uuid=uuid4())
    return features.collection[0]


class TestCollectionLevelMerge:
    """Features.merge_options must apply Options.inherit_from forward-by-default semantics."""

    def test_default_input_feature_inherits_all_consumer_group_options(self) -> None:
        """Forward by default: with forward_group left unspecified (None), every consumer
        group key except in_features flows into the child."""
        consumer = Options(
            group={
                "backend": "neo4j",
                "top_k": 5,
                DefaultOptionKeys.in_features: "child",
            }
        )

        child = _merge(Feature("child", options={"own_key": "own_value"}), consumer)

        assert child.options.group == {"own_key": "own_value", "backend": "neo4j", "top_k": 5}
        assert DefaultOptionKeys.in_features not in child.options.group
        assert DefaultOptionKeys.in_features not in child.options.context
        assert child.options.inherited_group_keys == frozenset({"backend", "top_k"})

    def test_explicit_none_input_feature_inherits_all_consumer_group_options(self) -> None:
        """forward_group=None (explicitly unspecified) is identical to the default: all flows."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        child = _merge(Feature("child", forward_group=None), consumer)

        assert child.options.group["backend"] == "neo4j"
        assert child.options.group["top_k"] == 5

    def test_forward_group_true_is_identical_to_the_default(self) -> None:
        """forward_group=True forwards all consumer group keys except in_features, like None."""
        consumer = Options(
            group={
                "backend": "neo4j",
                "top_k": 5,
                DefaultOptionKeys.in_features: "child",
            }
        )

        child = _merge(Feature("child", forward_group=True), consumer)

        assert child.options.group["backend"] == "neo4j"
        assert child.options.group["top_k"] == 5
        assert DefaultOptionKeys.in_features not in child.options.group

    def test_allowlisted_keys_are_the_only_forwarded_group_options(self) -> None:
        """forward_group as an allowlist forwards exactly the listed consumer group keys."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5, "query_text": "hello"})

        child = _merge(Feature("child", forward_group={"backend"}), consumer)

        assert child.options.group == {"backend": "neo4j"}

    def test_forward_group_false_forwards_nothing(self) -> None:
        """forward_group=False is the explicit opt-out: no consumer group key flows."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        child = _merge(Feature("child", options={"own_key": "own_value"}, forward_group=False), consumer)

        assert child.options.group == {"own_key": "own_value"}
        assert child.options.inherited_group_keys == frozenset()

    def test_forward_group_exclude_carves_keys_out_of_the_default(self) -> None:
        """forward_group_exclude subtracts keys from the forward-by-default copy."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        child = _merge(Feature("child", forward_group_exclude={"top_k"}), consumer)

        assert child.options.group == {"backend": "neo4j"}

    def test_forward_group_exclude_carves_keys_out_of_an_allowlist(self) -> None:
        """forward_group_exclude also subtracts keys from an explicit allowlist."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5, "query_text": "hello"})

        child = _merge(
            Feature("child", forward_group={"backend", "top_k"}, forward_group_exclude={"top_k"}),
            consumer,
        )

        assert child.options.group == {"backend": "neo4j"}

    def test_forward_group_false_with_exclude_raises(self) -> None:
        """False plus a non-empty exclude is contradictory and rejected at construction."""
        with pytest.raises(ValueError, match="forward_group=False"):
            Feature("child", forward_group=False, forward_group_exclude={"top_k"})

    def test_in_features_never_flows_under_any_directive(self) -> None:
        """in_features is excluded from every flow, even when explicitly allowlisted."""
        consumer = Options(group={"backend": "neo4j", DefaultOptionKeys.in_features: "child"})

        default_child = _merge(Feature("child"), consumer)
        true_child = _merge(Feature("child", forward_group=True), consumer)
        allowlist_child = _merge(
            Feature("child", forward_group={"backend", str(DefaultOptionKeys.in_features)}), consumer
        )

        for child in (default_child, true_child, allowlist_child):
            assert DefaultOptionKeys.in_features not in child.options.group
            assert DefaultOptionKeys.in_features not in child.options.context
            assert child.options.group["backend"] == "neo4j"

    def test_inherit_context_keys_pulls_listed_consumer_context_key(self) -> None:
        """inherit_context_keys pulls only the listed consumer context keys into the child context."""
        consumer = Options(group={"backend": "neo4j"}, context={"trace": "abc", "other_ctx": "x"})

        child = _merge(Feature("child", inherit_context_keys={"trace"}), consumer)

        assert child.options.context["trace"] == "abc"
        assert "other_ctx" not in child.options.context

    def test_consumer_context_never_flows_implicitly(self) -> None:
        """Forward by default is group-only: consumer context keys stay behind without a pull/push."""
        consumer = Options(group={"backend": "neo4j"}, context={"trace": "abc"})

        child = _merge(Feature("child"), consumer)

        assert child.options.group["backend"] == "neo4j"
        assert "trace" not in child.options.context
        assert "trace" not in child.options.group

    def test_propagate_context_keys_push_still_reaches_child_context(self) -> None:
        """The consumer-side push: propagate_context_keys keeps landing in the child's context."""
        consumer = Options(
            context={"session_id": "abc", "quiet": True},
            propagate_context_keys=frozenset({"session_id"}),
        )

        child = _merge(Feature("child"), consumer)

        assert child.options.context["session_id"] == "abc"
        assert "quiet" not in child.options.context

    def test_conflicting_child_group_value_raises(self) -> None:
        """A forwarded consumer key colliding with a different child value raises ValueError."""
        consumer = Options(group={"backend": "neo4j"})

        with pytest.raises(ValueError, match="backend"):
            _merge(Feature("child", options={"backend": "postgres"}), consumer)

    def test_feature_chainer_parser_key_frozenset_has_no_shield_semantics(self) -> None:
        """
        No shield semantics: a feature_chainer_parser_key-style frozenset carried in group
        options gets no special treatment under default forwarding. The literal string is
        used on purpose so this test survives the deletion of the enum member; it is an
        ordinary group key on both sides and forwards like any other key.
        """
        shield = frozenset({"shielded_key"})
        consumer = Options(
            group={
                "feature_chainer_parser_key": shield,
                "shielded_key": "consumer_value",
                "backend": "neo4j",
            }
        )
        input_feature = Feature("child", options={"feature_chainer_parser_key": shield})

        child = _merge(input_feature, consumer)

        assert child.options.group["shielded_key"] == "consumer_value"
        assert child.options.group["backend"] == "neo4j"
        assert child.options.group["feature_chainer_parser_key"] == shield

    def test_child_options_still_set_to_consumer_options(self) -> None:
        """Pins engine wiring: merged features keep the consumer options as child_options."""
        consumer = Options(group={"backend": "neo4j"})

        child = _merge(Feature("child"), consumer)

        assert child.child_options is not None
        assert child.child_options.group == {"backend": "neo4j"}


class _ForwardingChainedGroup(FeatureChainParserMixin):
    """Minimal chainer subclass following the existing mixin test fixture style."""

    PREFIX_PATTERN = r".*__([\w]+)_fwdchain$"
    PROPERTY_MAPPING = {
        "operation": PropertySpec(
            "Operation to apply",
            allowed_values={"op1": "Operation 1"},
            context=True,
            strict_validation=True,
        )
    }


class TestMixinLeavesChildrenAtDefault:
    """FeatureChainParserMixin.input_features must NOT stamp forward_group on its children."""

    def test_string_parsed_children_have_forward_group_none(self) -> None:
        """Children parsed from a chained feature name stay at the default None sentinel."""
        group = _ForwardingChainedGroup()

        result = group.input_features(Options(), FeatureName("left&right__op1_fwdchain"))

        assert result is not None
        assert {str(child.name) for child in result} == {"left", "right"}
        for child in result:
            assert child.forward_group is None

    def test_config_declared_children_keep_author_directives_untouched(self) -> None:
        """
        Config path: in_features children left at the unspecified default (None) stay at
        None; an explicit user allowlist is kept untouched.
        """
        plain_child = Feature("plain_child")
        assert plain_child.forward_group is None  # precondition: the default is the None sentinel
        explicit_child = Feature("explicit_child", forward_group={"x"})
        options = Options(group={DefaultOptionKeys.in_features: frozenset({plain_child, explicit_child})})
        group = _ForwardingChainedGroup()

        result = group.input_features(options, FeatureName("config_based_feature"))

        assert result is not None
        by_name = {str(child.name): child for child in result}
        assert by_name["plain_child"].forward_group is None
        assert by_name["explicit_child"].forward_group == frozenset({"x"})

    def test_config_declared_child_with_explicit_false_is_not_flipped(self) -> None:
        """
        Config path: a child with an EXPLICIT forward_group=False opt-out is preserved by
        the mixin, and after the collection-level merge it receives nothing from the consumer.
        """
        optout_child = Feature("optout_child", forward_group=False)
        options = Options(group={DefaultOptionKeys.in_features: frozenset({optout_child})})
        group = _ForwardingChainedGroup()

        result = group.input_features(options, FeatureName("config_based_feature"))

        assert result is not None
        by_name = {str(child.name): child for child in result}
        assert by_name["optout_child"].forward_group is False

        consumer = Options(group={"backend": "neo4j", "top_k": 5})
        merged = _merge(by_name["optout_child"], consumer)

        assert "backend" not in merged.options.group
        assert "top_k" not in merged.options.group


class TestShieldMachineryRetirement:
    """The old shield API must be deleted once inherit_from drives the merge."""

    def test_default_option_keys_has_no_feature_chainer_parser_key_member(self) -> None:
        assert not hasattr(DefaultOptionKeys, "feature_chainer_parser_key")
        assert "feature_chainer_parser_key" not in {member.value for member in DefaultOptionKeys}

    def test_options_has_no_update_with_protected_keys(self) -> None:
        assert not hasattr(Options, "update_with_protected_keys")

    def test_options_has_no_forward_for_input_feature(self) -> None:
        assert not hasattr(Options, "forward_for_input_feature")
