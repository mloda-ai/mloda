"""
Tests for issue #602: copy Options for string-declared input features instead of
sharing the consumer's instance.

Today a bare-string input feature keeps the consumer's Options object (Feature stores
the passed Options instance as-is, and Features.build_feature_collection reuses one
child_options across the loop). The self-merge in Options.inherit_from then no-ops, so
every string child aliases the consumer's group and context dicts. These tests pin the
desired post-fix behavior: each string child owns a deep, independent copy of the
consumer's options, so mutations never leak, while the forwarded values are preserved.
"""

from __future__ import annotations

from uuid import uuid4

from mloda.user import Feature
from mloda.user import Features
from mloda.user import Options


def _children(consumer: Options) -> dict[str, Feature]:
    """Build a collection from two bare-string input features, keyed by name."""
    features = Features(["alpha", "beta"], child_options=consumer, child_uuid=uuid4())
    return {str(feature.name): feature for feature in features.collection}


class TestStringChildOptionsIsolation:
    """String-declared input features must own copies of the consumer's Options."""

    def test_string_children_get_distinct_option_instances(self) -> None:
        """Each string child's options is a fresh object, not the shared consumer instance."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        assert alpha.options is not consumer
        assert beta.options is not consumer
        assert alpha.options is not beta.options

    def test_top_level_group_mutation_does_not_leak(self) -> None:
        """Mutating one child's top-level group value leaves consumer and sibling untouched."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        alpha.options.group["backend"] = "MUTATED"

        assert consumer.group["backend"] == "neo4j"
        assert beta.options.group["backend"] == "neo4j"

    def test_nested_group_mutation_does_not_leak(self) -> None:
        """The sharper aliasing check: mutating a nested container in one child stays local."""
        consumer = Options(group={"cfg": {"a": 1}})

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        alpha.options.group["cfg"]["a"] = 999

        assert consumer.group["cfg"]["a"] == 1
        assert beta.options.group["cfg"]["a"] == 1

    def test_top_level_context_mutation_does_not_leak(self) -> None:
        """Mutating one child's top-level context value leaves consumer and sibling untouched."""
        consumer = Options(group={"backend": "neo4j"}, context={"trace": "abc"})

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        alpha.options.context["trace"] = "MUTATED"

        assert consumer.context["trace"] == "abc"
        assert beta.options.context["trace"] == "abc"

    def test_nested_context_mutation_does_not_leak(self) -> None:
        """The sharper aliasing check for context: nested mutation in one child stays local."""
        consumer = Options(group={"backend": "neo4j"}, context={"cfg": {"a": 1}})

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        alpha.options.context["cfg"]["a"] = 999

        assert consumer.context["cfg"]["a"] == 1
        assert beta.options.context["cfg"]["a"] == 1

    def test_forwarded_group_and_context_values_are_preserved(self) -> None:
        """The copy must not drop data: forwarded group and context values still reach the child."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5}, context={"trace": "abc"})

        children = _children(consumer)
        alpha = children["alpha"]

        assert alpha.options.group["backend"] == "neo4j"
        assert alpha.options.group["top_k"] == 5
        assert alpha.options.context["trace"] == "abc"


class TestStringChildBehaviorPreserved:
    """String-declared input features must keep pre-fix self-merge behavior, not real forwarding.

    Copying a string child's Options must not route it through the real forwarding merge in
    Options.inherit_from. Pre-fix a string child was a self-merge no-op: no provenance was
    recorded and consumer option values were never compared. These tests pin that behavior so
    the forwarded-name-mismatch guard and dual-consumption warning do not newly fire on string
    children, and so an identity-equality option value does not make the merge raise.
    """

    def test_string_child_provenance_stays_empty(self) -> None:
        """A string child records no forwarding provenance (self-merge preserved)."""
        consumer = Options(group={"backend": "neo4j", "top_k": 5})

        alpha = _children(consumer)["alpha"]

        assert alpha.options.inherited_group_keys == frozenset()
        assert alpha.options.last_forwarded_group_keys == frozenset()

    def test_string_child_does_not_raise_on_identity_equality_value(self) -> None:
        """A consumer option value with identity-based __eq__ must not make the copy raise."""
        consumer = Options(group={"payload": object()})

        features = Features(["alpha"], child_options=consumer, child_uuid=uuid4())
        children = {str(feature.name): feature for feature in features.collection}
        alpha = children["alpha"]

        assert "payload" in alpha.options.group

    def test_forwarded_nested_context_is_isolated(self) -> None:
        """Nested context forwarded via propagate_context_keys is preserved and copied per child."""
        consumer = Options(context={"cfg": {"a": 1}}, propagate_context_keys=frozenset({"cfg"}))

        children = _children(consumer)
        alpha, beta = children["alpha"], children["beta"]

        assert alpha.options.context["cfg"]["a"] == 1
        assert beta.options.context["cfg"]["a"] == 1

        alpha.options.context["cfg"]["a"] = 999

        assert consumer.context["cfg"]["a"] == 1
        assert beta.options.context["cfg"]["a"] == 1
