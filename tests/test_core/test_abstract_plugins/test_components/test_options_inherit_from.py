from copy import deepcopy
from uuid import uuid4

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.user import Feature
from mloda.user import Features
from mloda.user import Options


class TestInheritFromDefaultForwarding:
    """
    Test suite for Options.inherit_from default behavior: forward by default.

    WHY THIS EXISTS:
    inherit_from is the engine merge primitive for declared input features.
    `self` is the options of the input (upstream/child) feature; `consumer` is
    the options of the feature that declared it. By default ALL consumer group
    keys flow to the child (except in_features). Opting out is explicit via
    forward_group=False or forward_group_exclude.
    """

    def test_default_copies_all_group_keys(self) -> None:
        """Default call copies every consumer group key onto the child."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer)

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["top_k"] == 5
        assert child.group["own_key"] == "own_value"

    def test_default_copies_no_context_keys(self) -> None:
        """Default call must not copy any consumer context keys onto the child."""
        consumer = Options(context={"debug_mode": True, "trace_id": "abc"})
        child = Options()

        child.inherit_from(consumer)

        assert child.context == {}

    def test_default_does_not_copy_in_features(self) -> None:
        """in_features stays off the child even though the default forwards all group keys."""
        consumer = Options(
            group={
                DefaultOptionKeys.in_features: "consumer_source",
                "kg_backend": "neo4j",
            }
        )
        child = Options()

        child.inherit_from(consumer)

        assert DefaultOptionKeys.in_features not in child.group
        assert DefaultOptionKeys.in_features not in child.context
        assert child.group["kg_backend"] == "neo4j"

    def test_default_conflicting_child_group_value_raises_naming_key(self) -> None:
        """A forwarded key already on the child with a different value raises ValueError naming the key."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError, match="kg_backend"):
            child.inherit_from(consumer)

    def test_default_equal_child_group_value_is_noop(self) -> None:
        """A forwarded key already on the child with the same value passes and keeps its value."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer)

        assert child.group["kg_backend"] == "neo4j"

    def test_default_forwarded_key_existing_in_child_context_raises(self) -> None:
        """A forwarded group key that exists in child.context is a cross-conflict."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(context={"kg_backend": "neo4j"})

        with pytest.raises(ValueError, match="kg_backend"):
            child.inherit_from(consumer)

    def test_default_does_not_mutate_consumer(self) -> None:
        """The consumer object must not be mutated by inherit_from."""
        consumer = Options(
            group={"kg_backend": "neo4j"},
            context={"trace_id": "abc"},
        )
        group_before = deepcopy(consumer.group)
        context_before = deepcopy(consumer.context)
        propagate_before = consumer.propagate_context_keys

        child = Options(group={"own_key": "own_value"})
        child.inherit_from(consumer)

        assert consumer.group == group_before
        assert consumer.context == context_before
        assert consumer.propagate_context_keys == propagate_before


class TestInheritFromForwardGroupNone:
    """
    Test suite for forward_group=None: the unspecified sentinel.

    None is the default of the parameter (mirroring Feature.forward_group).
    It is equivalent to True: all consumer group keys are copied (except
    in_features). Only the group flow is affected: the context pull and the
    consumer-side push keep working alongside None.
    """

    def test_none_copies_all_group_keys(self) -> None:
        """forward_group=None copies every consumer group key onto the child."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer, forward_group=None)

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["top_k"] == 5
        assert child.group["own_key"] == "own_value"

    def test_none_and_true_are_equivalent(self) -> None:
        """forward_group=None and forward_group=True produce the same child group."""
        consumer = Options(
            group={
                "kg_backend": "neo4j",
                "top_k": 5,
                DefaultOptionKeys.in_features: "consumer_source",
            }
        )
        child_none = Options()
        child_true = Options()

        child_none.inherit_from(consumer, forward_group=None)
        child_true.inherit_from(consumer, forward_group=True)

        assert child_none.group == child_true.group

    def test_none_copies_no_context_keys(self) -> None:
        """forward_group=None must not leak anything into the child context."""
        consumer = Options(group={"kg_backend": "neo4j"}, context={"trace_id": "abc"})
        child = Options()

        child.inherit_from(consumer, forward_group=None)

        assert child.group == {"kg_backend": "neo4j"}
        assert child.context == {}

    def test_signature_default_is_none(self) -> None:
        """The forward_group parameter default is the None sentinel, not a bool."""
        from inspect import signature

        default = signature(Options.inherit_from).parameters["forward_group"].default
        assert default is None

    def test_none_still_allows_context_pull(self) -> None:
        """inherit_context_keys keeps working when forward_group is passed as None."""
        consumer = Options(group={"kg_backend": "neo4j"}, context={"trace_id": "abc"})
        child = Options()

        child.inherit_from(consumer, forward_group=None, inherit_context_keys=frozenset({"trace_id"}))

        assert child.context["trace_id"] == "abc"
        assert child.group["kg_backend"] == "neo4j"


class TestInheritFromForwardGroupTrue:
    """
    Test suite for forward_group=True: inherit all consumer group keys.

    Copies all of consumer.group except in_features. Context never flows via
    forward_group.
    """

    def test_all_group_keys_copied_except_in_features(self) -> None:
        """True copies every consumer group key except in_features."""
        consumer = Options(
            group={
                "kg_backend": "neo4j",
                "top_k": 5,
                DefaultOptionKeys.in_features: "consumer_source",
            }
        )
        child = Options()

        child.inherit_from(consumer, forward_group=True)

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["top_k"] == 5
        assert DefaultOptionKeys.in_features not in child.group

    def test_conflicting_child_group_value_raises_naming_key(self) -> None:
        """With True, an existing differing child group value raises ValueError naming the key."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"top_k": 10})

        with pytest.raises(ValueError, match="top_k"):
            child.inherit_from(consumer, forward_group=True)

    def test_equal_child_group_values_are_fine(self) -> None:
        """With True, equal pre-existing values do not raise."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"top_k": 5})

        child.inherit_from(consumer, forward_group=True)

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["top_k"] == 5

    def test_consumer_context_not_copied_by_true(self) -> None:
        """forward_group=True must not copy consumer.context; context never flows via forward_group."""
        consumer = Options(
            group={"kg_backend": "neo4j"},
            context={"debug_mode": True, "trace_id": "abc"},
        )
        child = Options()

        child.inherit_from(consumer, forward_group=True)

        assert child.context == {}
        assert "debug_mode" not in child.group
        assert "trace_id" not in child.group


class TestInheritFromForwardGroupFalse:
    """
    Test suite for forward_group=False: the explicit opt-out.

    False copies no group keys. Combining False with a non-empty
    forward_group_exclude is contradictory and raises ValueError. The
    child-side context pull (inherit_context_keys) keeps working alongside
    False because it is an explicit child request; the consumer-side push
    (propagate_context_keys) is BLOCKED by False, see
    TestInheritFromForwardGroupFalseBlocksPush.
    """

    def test_false_copies_no_group_keys(self) -> None:
        """forward_group=False copies no consumer group keys onto the child."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer, forward_group=False)

        assert child.group == {"own_key": "own_value"}
        assert "kg_backend" not in child.group
        assert "top_k" not in child.group

    def test_false_still_allows_context_pull(self) -> None:
        """inherit_context_keys keeps working when forward_group is False."""
        consumer = Options(group={"kg_backend": "neo4j"}, context={"trace_id": "abc"})
        child = Options()

        child.inherit_from(consumer, forward_group=False, inherit_context_keys=frozenset({"trace_id"}))

        assert child.context["trace_id"] == "abc"
        assert "kg_backend" not in child.group

    def test_false_with_nonempty_exclude_raises_valueerror(self) -> None:
        """forward_group=False plus a non-empty forward_group_exclude is contradictory."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        with pytest.raises(ValueError, match="forward_group"):
            child.inherit_from(consumer, forward_group=False, forward_group_exclude=frozenset({"kg_backend"}))

    def test_false_with_empty_exclude_is_fine(self) -> None:
        """forward_group=False with the default empty exclude copies nothing and does not raise."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, forward_group=False, forward_group_exclude=frozenset())

        assert child.group == {}


class TestInheritFromForwardGroupFalseBlocksPush:
    """
    Test suite for forward_group=False blocking the consumer-side push.

    forward_group=False means "the child is fully isolated": besides the group
    flow, it also blocks the consumer-side propagate_context_keys PUSH. This
    restores the retired shield's escape hatch: a child with its own differing
    value for a pushed context key can opt out instead of hitting the
    "Context key ... conflict" ValueError. Only the literal False blocks the
    push (an empty frozenset allowlist does not). The child-side pull via
    inherit_context_keys stays available alongside False: it is an explicit
    child request, not a consumer-side push.
    """

    def test_false_blocks_push_and_child_keeps_own_differing_value(self) -> None:
        """A child with forward_group=False and its own differing value for a pushed
        context key does not raise and keeps its own value (the escape hatch)."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options(context={"env": "staging"})

        child.inherit_from(consumer, forward_group=False)

        assert child.context["env"] == "staging"

    def test_without_false_the_push_conflict_still_raises(self) -> None:
        """The same differing child without forward_group=False still hits the conflict."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options(context={"env": "staging"})

        with pytest.raises(ValueError, match="Context key.*conflict"):
            child.inherit_from(consumer)

    def test_false_blocks_push_delivery_to_child_without_own_value(self) -> None:
        """With forward_group=False, a pushed key does NOT land on a child that has
        no own value for it: nothing flows."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options()

        child.inherit_from(consumer, forward_group=False)

        assert "env" not in child.context
        assert "env" not in child.group

    def test_pull_with_false_still_delivers_pushed_key(self) -> None:
        """An explicit child-side pull (inherit_context_keys) wins over the isolation:
        combining it with forward_group=False still delivers the value."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options()

        child.inherit_from(consumer, forward_group=False, inherit_context_keys=frozenset({"env"}))

        assert child.context["env"] == "prod"

    def test_empty_allowlist_does_not_block_push(self) -> None:
        """Only the literal False blocks the push; an empty frozenset allowlist
        (no group keys flow) still lets the push deliver."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset())

        assert child.context["env"] == "prod"


class TestInheritFromForwardGroupAllowlist:
    """
    Test suite for forward_group as an explicit frozenset allowlist.

    Only listed keys are copied, and only group-to-group. Listed keys missing
    from consumer.group are optional selectors: silent no-op, never read from
    consumer.context. in_features is never forwarded.
    """

    def test_listed_keys_present_in_consumer_group_are_copied(self) -> None:
        """Keys named in the allowlist and present in consumer.group land in child.group."""
        consumer = Options(group={"kg_backend": "neo4j", "region": "eu-west"})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend", "region"}))

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["region"] == "eu-west"

    def test_unlisted_keys_are_not_copied(self) -> None:
        """Consumer group keys not in the allowlist must not flow to the child."""
        consumer = Options(group={"kg_backend": "neo4j", "query_text": "hello", "top_k": 5})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))

        assert child.group["kg_backend"] == "neo4j"
        assert "query_text" not in child.group
        assert "top_k" not in child.group

    def test_listed_key_absent_from_consumer_group_is_silent_noop(self) -> None:
        """An allowlisted key the consumer does not carry is skipped without error."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend", "optional_selector"}))

        assert child.group["kg_backend"] == "neo4j"
        assert "optional_selector" not in child.group
        assert "optional_selector" not in child.context

    def test_listed_key_is_not_read_from_consumer_context(self) -> None:
        """forward_group is group-to-group only: consumer.context is never a source."""
        consumer = Options(context={"optional_selector": "ctx_value"})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"optional_selector"}))

        assert "optional_selector" not in child.group
        assert "optional_selector" not in child.context

    def test_in_features_never_forwarded_even_if_listed(self) -> None:
        """in_features stays off the child even when explicitly allowlisted."""
        consumer = Options(
            group={
                DefaultOptionKeys.in_features: "consumer_source",
                "kg_backend": "neo4j",
            }
        )
        child = Options()

        child.inherit_from(
            consumer,
            forward_group=frozenset({DefaultOptionKeys.in_features, "kg_backend"}),
        )

        assert DefaultOptionKeys.in_features not in child.group
        assert child.group["kg_backend"] == "neo4j"

    def test_conflicting_existing_child_group_value_raises_naming_key(self) -> None:
        """A listed key already on the child with a different value raises ValueError naming the key."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError, match="kg_backend"):
            child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))

    def test_equal_existing_child_group_value_is_fine(self) -> None:
        """A listed key already on the child with the same value passes and keeps its value."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))

        assert child.group["kg_backend"] == "neo4j"

    def test_forwarded_key_existing_in_child_context_raises(self) -> None:
        """A forwarded group key that exists in child.context is a cross-conflict."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(context={"kg_backend": "neo4j"})

        with pytest.raises(ValueError, match="kg_backend"):
            child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))


class TestInheritFromForwardGroupExclude:
    """
    Test suite for forward_group_exclude: typed opt-out of single keys.

    The exclude set is subtracted from whatever key set forward_group
    produced (all keys for None/True, the listed keys for an allowlist).
    Excluding a key the consumer does not carry is a silent no-op.
    """

    def test_exclude_subtracts_under_default(self) -> None:
        """With the default forward_group, excluded keys are not copied; the rest are."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5, "region": "eu-west"})
        child = Options()

        child.inherit_from(consumer, forward_group_exclude=frozenset({"top_k"}))

        assert child.group["kg_backend"] == "neo4j"
        assert child.group["region"] == "eu-west"
        assert "top_k" not in child.group

    def test_exclude_subtracts_under_true(self) -> None:
        """With forward_group=True, excluded keys are not copied; the rest are."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        child.inherit_from(consumer, forward_group=True, forward_group_exclude=frozenset({"kg_backend"}))

        assert child.group["top_k"] == 5
        assert "kg_backend" not in child.group

    def test_exclude_subtracts_from_allowlist(self) -> None:
        """An exclude combined with an allowlist removes keys from the allowlisted set."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5, "region": "eu-west"})
        child = Options()

        child.inherit_from(
            consumer,
            forward_group=frozenset({"kg_backend", "top_k"}),
            forward_group_exclude=frozenset({"top_k"}),
        )

        assert child.group["kg_backend"] == "neo4j"
        assert "top_k" not in child.group
        assert "region" not in child.group

    def test_excluded_key_avoids_group_conflict(self) -> None:
        """Excluding a key that would conflict with the child's own value prevents the ValueError."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "memgraph"})

        child.inherit_from(consumer, forward_group_exclude=frozenset({"kg_backend"}))

        assert child.group["kg_backend"] == "memgraph"
        assert child.group["top_k"] == 5

    def test_exclude_key_absent_from_consumer_is_silent_noop(self) -> None:
        """Excluding a key the consumer does not carry is skipped without error."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, forward_group_exclude=frozenset({"missing_key"}))

        assert child.group["kg_backend"] == "neo4j"

    def test_signature_exclude_default_is_empty_frozenset(self) -> None:
        """The forward_group_exclude parameter default is an empty frozenset."""
        from inspect import signature

        default = signature(Options.inherit_from).parameters["forward_group_exclude"].default
        assert default == frozenset()

    def test_exclude_does_not_affect_context_pull(self) -> None:
        """forward_group_exclude only affects the group flow, not inherit_context_keys."""
        consumer = Options(group={"kg_backend": "neo4j"}, context={"trace_id": "abc"})
        child = Options()

        child.inherit_from(
            consumer,
            forward_group_exclude=frozenset({"trace_id"}),
            inherit_context_keys=frozenset({"trace_id"}),
        )

        assert child.context["trace_id"] == "abc"
        assert child.group["kg_backend"] == "neo4j"


class TestInheritedGroupKeys:
    """
    Test suite for Options.inherited_group_keys.

    After inherit_from, the attribute holds every group key that FLOWED from
    the consumer: keys newly copied onto the child AND forwarded keys the child
    already held with an equal value (the value flowed either way). Keys the
    consumer does not forward (missing, allowlist carve-out, exclude) never
    count. It lets later engine stages distinguish inherited options from the
    child's own declarations.
    """

    def test_fresh_options_has_empty_inherited_group_keys(self) -> None:
        """A freshly constructed Options carries an empty frozenset."""
        options = Options(group={"own_key": "own_value"})

        assert isinstance(options.inherited_group_keys, frozenset)
        assert options.inherited_group_keys == frozenset()

    def test_default_inherit_records_copied_keys(self) -> None:
        """With the default forward, all newly-copied group keys are recorded."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer)

        assert child.inherited_group_keys == frozenset({"kg_backend", "top_k"})

    def test_child_own_keys_are_not_recorded(self) -> None:
        """Keys the child carries that the consumer does not forward are never recorded."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer)

        assert "own_key" not in child.inherited_group_keys

    def test_pre_existing_equal_value_key_is_recorded(self) -> None:
        """A forwarded key the child already carried with an EQUAL value COUNTS as inherited:
        the value flowed from the consumer even though the copy itself was a no-op."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer)

        assert child.inherited_group_keys == frozenset({"kg_backend", "top_k"})

    def test_allowlist_records_only_listed_copied_keys(self) -> None:
        """With an allowlist, only the keys actually copied are recorded."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend", "missing_key"}))

        assert child.inherited_group_keys == frozenset({"kg_backend"})

    def test_false_records_no_keys(self) -> None:
        """forward_group=False copies nothing, so nothing is recorded."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, forward_group=False)

        assert child.inherited_group_keys == frozenset()

    def test_in_features_never_recorded(self) -> None:
        """in_features is never copied, so it is never recorded."""
        consumer = Options(group={DefaultOptionKeys.in_features: "consumer_source", "kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer)

        assert child.inherited_group_keys == frozenset({"kg_backend"})

    def test_deepcopy_preserves_inherited_group_keys(self) -> None:
        """inherited_group_keys survives Options.__deepcopy__."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()
        child.inherit_from(consumer)

        copied = deepcopy(child)

        assert copied.inherited_group_keys == frozenset({"kg_backend"})


class TestInheritedGroupKeysAccumulation:
    """
    Test suite for inherited_group_keys accumulation across inherit_from calls.

    One child Feature/Options instance can be declared by SEVERAL consumers
    (fan-in on a shared input feature). Each inherit_from call must UNION its
    newly forwarded keys into inherited_group_keys instead of overwriting, so
    the provenance of the first merge survives the second. Together with the
    equal-value rule this keeps the dual-consumption warning alive for every
    consumer of a shared child.
    """

    def test_two_sequential_inherits_union_inherited_keys(self) -> None:
        """A second inherit_from from a different consumer ADDS its keys to the recorded set."""
        consumer_a = Options(group={"kg_backend": "neo4j"})
        consumer_b = Options(group={"top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer_a)
        assert child.inherited_group_keys == frozenset({"kg_backend"})

        child.inherit_from(consumer_b)

        assert child.inherited_group_keys == frozenset({"kg_backend", "top_k"})

    def test_second_inherit_with_equal_shared_key_preserves_first_provenance(self) -> None:
        """Two consumers forwarding the SAME key with EQUAL values: the key stays recorded
        after the second merge (the shared-child dual-consumption scenario)."""
        consumer_a = Options(group={"mode": "fast"})
        consumer_b = Options(group={"mode": "fast", "salt": "b"})
        child = Options()

        child.inherit_from(consumer_a)
        child.inherit_from(consumer_b)

        assert child.inherited_group_keys == frozenset({"mode", "salt"})

    def test_equal_value_key_counts_under_allowlist(self) -> None:
        """An allowlisted key the child already holds with an equal value is recorded."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))

        assert child.inherited_group_keys == frozenset({"kg_backend"})

    def test_author_key_not_forwarded_is_not_recorded(self) -> None:
        """A key the child author set that the consumer does NOT forward (excluded) never
        counts, even though consumer and child hold equal values."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer, forward_group_exclude=frozenset({"kg_backend"}))

        assert child.inherited_group_keys == frozenset({"top_k"})

    def test_self_merge_after_two_inherits_preserves_union(self) -> None:
        """inherit_from(self) after two real merges keeps the accumulated union intact."""
        consumer_a = Options(group={"kg_backend": "neo4j"})
        consumer_b = Options(group={"top_k": 5})
        child = Options()

        child.inherit_from(consumer_a)
        child.inherit_from(consumer_b)

        child.inherit_from(child)

        assert child.inherited_group_keys == frozenset({"kg_backend", "top_k"})


class TestLastForwardedGroupKeys:
    """
    Test suite for Options.last_forwarded_group_keys (NEW SPEC).

    Distinct from inherited_group_keys: while inherited_group_keys is the
    ACCUMULATING UNION of every group key ever forwarded onto self across all
    inherit_from calls, last_forwarded_group_keys holds ONLY the keys the LAST
    inherit_from call actually forwarded. Each real merge REPLACES it with that
    call's forwarded set (same membership rule as inherited: keys copied AND
    equal-value forwards count; in_features, allowlist carve-outs, and excludes
    never count). A self-merge PRESERVES it (whereas inherited is also
    preserved). __deepcopy__ preserves it. Default frozenset() at construction.

    The engine reads this per-consumer set (intersected with the consumer's
    PROPERTY_MAPPING keys) to attribute the dual-consumption warning to the
    consumer that actually forwarded the key, not to every consumer that merely
    declares it.
    """

    def test_fresh_options_has_empty_last_forwarded_group_keys(self) -> None:
        """A freshly constructed Options carries an empty frozenset."""
        options = Options(group={"own_key": "own_value"})

        assert isinstance(options.last_forwarded_group_keys, frozenset)
        assert options.last_forwarded_group_keys == frozenset()

    def test_default_inherit_records_this_call_forwarded_keys(self) -> None:
        """A default merge records exactly the keys forwarded in that call."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        child.inherit_from(consumer)

        assert child.last_forwarded_group_keys == frozenset({"kg_backend", "top_k"})

    def test_equal_value_forward_is_recorded(self) -> None:
        """A forwarded key the child already held with an EQUAL value counts, mirroring
        inherited_group_keys: the value flowed even though the write was a no-op."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer)

        assert child.last_forwarded_group_keys == frozenset({"kg_backend", "top_k"})

    def test_second_merge_replaces_rather_than_unions(self) -> None:
        """Unlike inherited_group_keys (which UNIONS), a second merge REPLACES
        last_forwarded_group_keys with only that call's forwarded set."""
        consumer_a = Options(group={"kg_backend": "neo4j"})
        consumer_b = Options(group={"top_k": 5})
        child = Options()

        child.inherit_from(consumer_a)
        assert child.last_forwarded_group_keys == frozenset({"kg_backend"})

        child.inherit_from(consumer_b)

        # last_forwarded reflects ONLY the second call ...
        assert child.last_forwarded_group_keys == frozenset({"top_k"})
        # ... while inherited_group_keys still accumulates the union.
        assert child.inherited_group_keys == frozenset({"kg_backend", "top_k"})

    def test_non_forwarding_second_merge_resets_last_forwarded_to_empty(self) -> None:
        """A second consumer that forwards nothing (empty group) empties last_forwarded,
        even though the first consumer's key stays in the accumulating inherited union."""
        consumer_a = Options(group={"mode": "fast"})
        consumer_b = Options(group={})
        child = Options()

        child.inherit_from(consumer_a)
        assert child.last_forwarded_group_keys == frozenset({"mode"})

        child.inherit_from(consumer_b)

        assert child.last_forwarded_group_keys == frozenset()
        assert child.inherited_group_keys == frozenset({"mode"})

    def test_false_records_empty_last_forwarded(self) -> None:
        """forward_group=False forwards nothing, so last_forwarded is empty."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, forward_group=False)

        assert child.last_forwarded_group_keys == frozenset()

    def test_exclude_absent_from_last_forwarded(self) -> None:
        """An excluded key never appears in last_forwarded."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        child.inherit_from(consumer, forward_group_exclude=frozenset({"top_k"}))

        assert child.last_forwarded_group_keys == frozenset({"kg_backend"})

    def test_allowlist_records_only_forwarded_keys(self) -> None:
        """With an allowlist, only the keys actually forwarded are recorded (missing keys skipped)."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        child.inherit_from(consumer, forward_group=frozenset({"kg_backend", "missing_key"}))

        assert child.last_forwarded_group_keys == frozenset({"kg_backend"})

    def test_in_features_never_in_last_forwarded(self) -> None:
        """in_features is never forwarded, so it never appears in last_forwarded."""
        consumer = Options(group={DefaultOptionKeys.in_features: "consumer_source", "kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer)

        assert child.last_forwarded_group_keys == frozenset({"kg_backend"})

    def test_deepcopy_preserves_last_forwarded_group_keys(self) -> None:
        """last_forwarded_group_keys survives Options.__deepcopy__."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()
        child.inherit_from(consumer)

        copied = deepcopy(child)

        assert copied.last_forwarded_group_keys == frozenset({"kg_backend"})


class TestInheritFromSelfMergeGuard:
    """
    Test suite for the self-merge defensive guard.

    ``opts.inherit_from(opts)`` (the consumer IS self) must be a no-op: it must
    NOT deep-copy self's own values into itself, must NOT stamp self-referential
    provenance, and must PRESERVE last_forwarded_group_keys (the aliasing hazard:
    a shared Options reused as an input feature keeps its recorded forwarded set)
    while PRESERVING the accumulated inherited_group_keys. It returns frozenset().
    """

    def test_self_merge_is_noop_preserving_provenance_and_last_forwarded(self) -> None:
        """A self-merge preserves inherited_group_keys, leaves values untouched, adds no
        new keys, PRESERVES last_forwarded_group_keys, and returns an empty frozenset."""
        opts = Options(group={"a": 1})
        consumer = Options(group={"a": 1})
        real_forwarded = opts.inherit_from(consumer)

        assert opts.inherited_group_keys == frozenset({"a"})
        assert real_forwarded == frozenset({"a"})
        assert opts.last_forwarded_group_keys == frozenset({"a"})

        self_forwarded = opts.inherit_from(opts)

        # Provenance preserved, no self-referential churn.
        assert opts.inherited_group_keys == frozenset({"a"})
        # Values untouched, no new keys introduced.
        assert opts.group == {"a": 1}
        # A self-merge forwards nothing this call, so it RETURNS an empty frozenset ...
        assert self_forwarded == frozenset()
        # ... but PRESERVES the previously recorded forwarded set (aliasing regression).
        assert opts.last_forwarded_group_keys == frozenset({"a"})


class TestInheritFromReturnValue:
    """
    Test suite for the frozenset[str] RETURN VALUE of Options.inherit_from (NEW SPEC).

    inherit_from now returns the set of consumer group keys it forwarded in THIS call,
    with the same membership rule as last_forwarded_group_keys: keys copied AND keys the
    child already held with an equal value both count; in_features, allowlist carve-outs,
    and excluded keys never count. A self-merge (consumer IS self) returns frozenset()
    without touching the recorded set. The engine consumes this return value (stored on
    Feature.forwarded_group_keys) for dual-consumption attribution instead of reading the
    possibly-clobbered last_forwarded_group_keys off the shared Options instance.
    """

    def test_normal_merge_returns_forwarded_frozenset(self) -> None:
        """A default merge returns exactly the keys forwarded in that call."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        returned = child.inherit_from(consumer)

        assert returned == frozenset({"kg_backend", "top_k"})

    def test_return_matches_last_forwarded_membership(self) -> None:
        """The return value has the same membership as last_forwarded_group_keys."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"own_key": "own_value"})

        returned = child.inherit_from(consumer)

        assert returned == child.last_forwarded_group_keys

    def test_equal_value_forward_is_in_return(self) -> None:
        """A forwarded key the child already held with an EQUAL value counts in the return."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options(group={"kg_backend": "neo4j"})

        returned = child.inherit_from(consumer)

        assert returned == frozenset({"kg_backend", "top_k"})

    def test_forward_group_false_returns_empty(self) -> None:
        """forward_group=False forwards nothing, so the return is empty."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        returned = child.inherit_from(consumer, forward_group=False)

        assert returned == frozenset()

    def test_empty_group_consumer_returns_empty(self) -> None:
        """A consumer with an empty group forwards nothing, so the return is empty."""
        consumer = Options(group={})
        child = Options(group={"own_key": "own_value"})

        returned = child.inherit_from(consumer)

        assert returned == frozenset()

    def test_allowlist_carve_out_honored_in_return(self) -> None:
        """With an allowlist, only the keys actually forwarded appear in the return."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        returned = child.inherit_from(consumer, forward_group=frozenset({"kg_backend", "missing_key"}))

        assert returned == frozenset({"kg_backend"})
        assert returned == child.last_forwarded_group_keys

    def test_exclude_carve_out_honored_in_return(self) -> None:
        """An excluded key never appears in the return."""
        consumer = Options(group={"kg_backend": "neo4j", "top_k": 5})
        child = Options()

        returned = child.inherit_from(consumer, forward_group_exclude=frozenset({"top_k"}))

        assert returned == frozenset({"kg_backend"})
        assert returned == child.last_forwarded_group_keys

    def test_in_features_carve_out_honored_in_return(self) -> None:
        """in_features is never forwarded, so it never appears in the return."""
        consumer = Options(group={DefaultOptionKeys.in_features: "consumer_source", "kg_backend": "neo4j"})
        child = Options()

        returned = child.inherit_from(consumer)

        assert returned == frozenset({"kg_backend"})

    def test_self_merge_returns_empty_and_preserves_last_forwarded(self) -> None:
        """A self-merge on an Options that already recorded a forwarded set (the aliasing
        hazard: the same instance reused as an input feature) returns frozenset() while
        the recorded last_forwarded_group_keys survives intact."""
        consumer = Options(group={"kg_backend": "neo4j"})
        shared = Options()
        shared.inherit_from(consumer)
        assert shared.last_forwarded_group_keys == frozenset({"kg_backend"})

        returned = shared.inherit_from(shared)

        # The self-merge forwards nothing this call ...
        assert returned == frozenset()
        # ... and PRESERVES the true forwarded set the engine later attributes provenance from.
        assert shared.last_forwarded_group_keys == frozenset({"kg_backend"})

    def test_second_merge_return_replaces_not_unions(self) -> None:
        """Each call's return reflects ONLY that call's forwarded set, not the accumulated union."""
        consumer_a = Options(group={"kg_backend": "neo4j"})
        consumer_b = Options(group={"top_k": 5})
        child = Options()

        first = child.inherit_from(consumer_a)
        second = child.inherit_from(consumer_b)

        assert first == frozenset({"kg_backend"})
        assert second == frozenset({"top_k"})


class TestFeatureForwardedGroupKeys:
    """
    Test suite for Feature.forwarded_group_keys (NEW SPEC).

    Features.merge_options stores the frozenset returned by Options.inherit_from onto
    feature.forwarded_group_keys, so the engine attributes the dual-consumption warning
    from this per-feature value instead of reading the (possibly clobbered) shared
    Options.last_forwarded_group_keys. Default frozenset() on a fresh Feature.
    """

    def test_fresh_feature_has_empty_forwarded_group_keys(self) -> None:
        """A freshly constructed Feature carries an empty forwarded_group_keys frozenset."""
        feature = Feature(name="child")

        assert isinstance(feature.forwarded_group_keys, frozenset)
        assert feature.forwarded_group_keys == frozenset()

    def test_merge_options_stores_inherit_from_return(self) -> None:
        """Features.merge_options stores the inherit_from return value onto the feature."""
        consumer_opts = Options(group={"kg_backend": "neo4j", "top_k": 5})
        features = Features(
            [Feature(name="child")],
            child_options=consumer_opts,
            child_uuid=uuid4(),
        )

        (child,) = features.collection
        assert child.forwarded_group_keys == frozenset({"kg_backend", "top_k"})
        assert child.forwarded_group_keys == child.options.last_forwarded_group_keys

    def test_merge_options_stored_value_honors_exclude(self) -> None:
        """The stored forwarded set reflects a feature-level forward_group_exclude carve-out."""
        consumer_opts = Options(group={"kg_backend": "neo4j", "top_k": 5})
        features = Features(
            [Feature(name="child", forward_group_exclude=frozenset({"top_k"}))],
            child_options=consumer_opts,
            child_uuid=uuid4(),
        )

        (child,) = features.collection
        assert child.forwarded_group_keys == frozenset({"kg_backend"})

    def test_merge_options_stored_value_empty_on_forward_group_false(self) -> None:
        """An isolated feature (forward_group=False) stores an empty forwarded set."""
        consumer_opts = Options(group={"kg_backend": "neo4j"})
        features = Features(
            [Feature(name="child", forward_group=False)],
            child_options=consumer_opts,
            child_uuid=uuid4(),
        )

        (child,) = features.collection
        assert child.forwarded_group_keys == frozenset()

    def test_string_child_forwarded_group_keys_recorded(self) -> None:
        """A bare-string input feature also gets its forwarded set stored."""
        consumer_opts = Options(group={"kg_backend": "neo4j"})
        features = Features(
            ["child"],
            child_options=consumer_opts,
            child_uuid=uuid4(),
        )

        (child,) = features.collection
        assert child.forwarded_group_keys == frozenset({"kg_backend"})


class TestInheritContextKeys:
    """
    Test suite for inherit_context_keys: child-side context pull.

    The child explicitly pulls listed context keys from the consumer,
    context-to-context only. Missing keys are a silent no-op; consumer.group
    is never a source; in_features is never pulled.
    """

    def test_listed_context_keys_are_pulled_into_child_context(self) -> None:
        """Listed keys present in consumer.context land in child.context."""
        consumer = Options(context={"trace_id": "abc", "log_level": "INFO"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"trace_id", "log_level"}))

        assert child.context["trace_id"] == "abc"
        assert child.context["log_level"] == "INFO"

    def test_unlisted_context_keys_are_not_pulled(self) -> None:
        """Consumer context keys not listed must stay off the child."""
        consumer = Options(context={"trace_id": "abc", "log_level": "INFO"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"trace_id"}))

        assert child.context["trace_id"] == "abc"
        assert "log_level" not in child.context

    def test_listed_key_absent_from_consumer_context_is_silent_noop(self) -> None:
        """A listed key the consumer context does not carry is skipped without error."""
        consumer = Options(context={"trace_id": "abc"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"trace_id", "missing_key"}))

        assert child.context["trace_id"] == "abc"
        assert "missing_key" not in child.context
        assert "missing_key" not in child.group

    def test_listed_key_is_not_read_from_consumer_group(self) -> None:
        """inherit_context_keys is context-to-context only: consumer.group is never a source."""
        consumer = Options(group={"cfg": "group_value"})
        child = Options()

        child.inherit_from(consumer, forward_group=False, inherit_context_keys=frozenset({"cfg"}))

        assert "cfg" not in child.context

    def test_in_features_never_pulled(self) -> None:
        """in_features stays off the child even when explicitly listed for context pull."""
        consumer = Options(context={DefaultOptionKeys.in_features: "consumer_source"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({DefaultOptionKeys.in_features}))

        assert DefaultOptionKeys.in_features not in child.context
        assert DefaultOptionKeys.in_features not in child.group

    def test_conflicting_existing_child_context_value_raises_naming_key(self) -> None:
        """A pulled key already on the child context with a different value raises ValueError naming the key."""
        consumer = Options(context={"trace_id": "abc"})
        child = Options(context={"trace_id": "xyz"})

        with pytest.raises(ValueError, match="trace_id"):
            child.inherit_from(consumer, inherit_context_keys=frozenset({"trace_id"}))

    def test_equal_existing_child_context_value_is_fine(self) -> None:
        """A pulled key already on the child context with the same value passes."""
        consumer = Options(context={"trace_id": "abc"})
        child = Options(context={"trace_id": "abc"})

        child.inherit_from(consumer, inherit_context_keys=frozenset({"trace_id"}))

        assert child.context["trace_id"] == "abc"

    def test_pulled_key_existing_in_child_group_raises(self) -> None:
        """A pulled context key that exists in child.group is a cross-conflict."""
        consumer = Options(context={"env": "prod"})
        child = Options(group={"env": "prod"})

        with pytest.raises(ValueError, match="env"):
            child.inherit_from(consumer, inherit_context_keys=frozenset({"env"}))


class TestInheritedContextKeys:
    """
    Test suite for Options.inherited_context_keys (NEW SPEC).

    Mirrors inherited_group_keys but for CONTEXT. After inherit_from, the
    attribute holds every context key actually DELIVERED to self: via the
    child-side inherit_context_keys PULL or the consumer-side
    propagate_context_keys PUSH. Keys delivered with an equal pre-existing
    value still count (the value flowed either way). Keys skipped because they
    are absent from consumer.context, blocked by forward_group=False (push
    only), or in NON_FORWARDED_KEYS never count. It UNIONS across calls; a
    self-merge PRESERVES it; __deepcopy__ PRESERVES it.

    Only these DELIVERED context keys participate in Feature.similarity_hash
    grouping, so this bookkeeping is the provenance the execution plan reads.
    """

    def test_fresh_options_has_empty_inherited_context_keys(self) -> None:
        """A freshly constructed Options carries an empty frozenset."""
        options = Options(context={"trace_id": "abc"})

        assert isinstance(options.inherited_context_keys, frozenset)
        assert options.inherited_context_keys == frozenset()

    def test_pull_records_delivered_key(self) -> None:
        """A key delivered via inherit_context_keys is recorded."""
        consumer = Options(context={"tenant": "acme"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))

        assert child.inherited_context_keys == frozenset({"tenant"})

    def test_push_records_delivered_key(self) -> None:
        """A key delivered via the consumer-side propagate_context_keys push is recorded."""
        consumer = Options(
            context={"session_id": "abc"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options()

        child.inherit_from(consumer)

        assert child.inherited_context_keys == frozenset({"session_id"})

    def test_pull_with_equal_existing_value_records_key(self) -> None:
        """A pulled key the child already held with an EQUAL value still counts:
        the value flowed from the consumer even though the write was a no-op."""
        consumer = Options(context={"tenant": "acme"})
        child = Options(context={"tenant": "acme"})

        child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))

        assert child.inherited_context_keys == frozenset({"tenant"})

    def test_push_with_equal_existing_value_records_key(self) -> None:
        """A pushed key the child already held with an EQUAL value still counts."""
        consumer = Options(
            context={"session_id": "abc"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options(context={"session_id": "abc"})

        child.inherit_from(consumer)

        assert child.inherited_context_keys == frozenset({"session_id"})

    def test_pull_key_absent_from_consumer_context_is_not_recorded(self) -> None:
        """A pulled key the consumer does NOT carry is skipped and never recorded."""
        consumer = Options(context={"tenant": "acme"})
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant", "missing_key"}))

        assert child.inherited_context_keys == frozenset({"tenant"})

    def test_forward_group_false_blocks_push_and_does_not_record(self) -> None:
        """A push blocked by forward_group=False delivers nothing, so nothing is recorded."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options()

        child.inherit_from(consumer, forward_group=False)

        assert child.inherited_context_keys == frozenset()

    def test_pull_with_forward_group_false_still_records(self) -> None:
        """The child-side PULL wins over isolation: combined with forward_group=False
        it still delivers the value, so the key IS recorded."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options()

        child.inherit_from(consumer, forward_group=False, inherit_context_keys=frozenset({"env"}))

        assert child.inherited_context_keys == frozenset({"env"})

    def test_in_features_is_never_recorded_in_context_provenance(self) -> None:
        """in_features is never delivered through any flow, so it never appears in the provenance."""
        consumer = Options(
            context={DefaultOptionKeys.in_features: "consumer_source", "tenant": "acme"},
            propagate_context_keys=frozenset({DefaultOptionKeys.in_features, "tenant"}),
        )
        child = Options()

        child.inherit_from(consumer)

        assert DefaultOptionKeys.in_features not in child.inherited_context_keys
        assert child.inherited_context_keys == frozenset({"tenant"})

    def test_unions_across_two_merges(self) -> None:
        """Two sequential merges from different consumers UNION their delivered keys."""
        consumer_a = Options(context={"tenant": "acme"})
        consumer_b = Options(context={"region": "eu"})
        child = Options()

        child.inherit_from(consumer_a, inherit_context_keys=frozenset({"tenant"}))
        assert child.inherited_context_keys == frozenset({"tenant"})

        child.inherit_from(consumer_b, inherit_context_keys=frozenset({"region"}))

        assert child.inherited_context_keys == frozenset({"tenant", "region"})

    def test_self_merge_preserves_inherited_context_keys(self) -> None:
        """A self-merge (shared Options instance) preserves the recorded set instead of resetting it."""
        consumer = Options(context={"tenant": "acme"})
        child = Options()
        child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))
        assert child.inherited_context_keys == frozenset({"tenant"})

        child.inherit_from(child)

        assert child.inherited_context_keys == frozenset({"tenant"})

    def test_deepcopy_preserves_inherited_context_keys(self) -> None:
        """inherited_context_keys survives Options.__deepcopy__."""
        consumer = Options(context={"tenant": "acme"})
        child = Options()
        child.inherit_from(consumer, inherit_context_keys=frozenset({"tenant"}))

        copied = deepcopy(child)

        assert copied.inherited_context_keys == frozenset({"tenant"})


class TestInheritFromPropagateContextKeysPush:
    """
    Test suite for the consumer-side push via propagate_context_keys.

    A consumer that marks context keys with propagate_context_keys pushes them
    into the child's context through inherit_from; the excluded set is just
    {in_features}. The push is blocked when the child opted out with
    forward_group=False (see TestInheritFromForwardGroupFalseBlocksPush).
    """

    def test_consumer_propagate_context_keys_are_pushed(self) -> None:
        """Keys listed in consumer.propagate_context_keys land in child.context; others do not."""
        consumer = Options(
            context={"session_id": "abc", "algo": "sum"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options()

        child.inherit_from(consumer)

        assert child.context["session_id"] == "abc"
        assert "algo" not in child.context

    def test_push_excluded_set_is_only_in_features(self) -> None:
        """Only in_features is excluded from the push; shield-style key protection is retired."""
        consumer = Options(
            group={"legacy_shield": frozenset({"session_id"})},
            context={
                "session_id": "abc",
                DefaultOptionKeys.in_features: "consumer_source",
            },
            propagate_context_keys=frozenset({"session_id", DefaultOptionKeys.in_features}),
        )
        child = Options()

        child.inherit_from(consumer)

        # session_id is pushed despite being listed under a shield-style frozenset key
        assert child.context["session_id"] == "abc"
        # in_features is the only exclusion
        assert DefaultOptionKeys.in_features not in child.context

    def test_push_and_pull_of_same_key_is_fine(self) -> None:
        """A key both pushed (propagate_context_keys) and pulled (inherit_context_keys) yields one value, no error."""
        consumer = Options(
            context={"session_id": "abc"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options()

        child.inherit_from(consumer, inherit_context_keys=frozenset({"session_id"}))

        assert child.context["session_id"] == "abc"

    def test_push_conflict_with_differing_child_context_value_raises(self) -> None:
        """A pushed key conflicting with an existing differing child context value raises ValueError."""
        consumer = Options(
            context={"session_id": "abc"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options(context={"session_id": "xyz"})

        with pytest.raises(ValueError, match="Context key.*conflict"):
            child.inherit_from(consumer)

    def test_push_same_value_no_error(self) -> None:
        """A pushed key matching the existing child context value passes."""
        consumer = Options(
            context={"session_id": "abc"},
            propagate_context_keys=frozenset({"session_id"}),
        )
        child = Options(context={"session_id": "abc"})

        child.inherit_from(consumer)

        assert child.context["session_id"] == "abc"

    def test_push_key_existing_in_child_group_raises(self) -> None:
        """A pushed context key that exists in child.group is a cross-conflict."""
        consumer = Options(
            context={"env": "prod"},
            propagate_context_keys=frozenset({"env"}),
        )
        child = Options(group={"env": "staging"})

        with pytest.raises(ValueError, match="env"):
            child.inherit_from(consumer)


class TestInheritFromInteraction:
    """
    Test suite for combined usage and for the retirement of the
    shield-style key protection.
    """

    def test_forward_group_and_inherit_context_keys_combine_in_one_call(self) -> None:
        """A frozenset forward_group and inherit_context_keys work together in a single call."""
        consumer = Options(
            group={"kg_backend": "neo4j", "query_text": "hello"},
            context={"trace_id": "abc", "log_level": "INFO"},
        )
        child = Options()

        child.inherit_from(
            consumer,
            forward_group=frozenset({"kg_backend"}),
            inherit_context_keys=frozenset({"trace_id"}),
        )

        assert child.group["kg_backend"] == "neo4j"
        assert "query_text" not in child.group
        assert child.context["trace_id"] == "abc"
        assert "log_level" not in child.context

    def test_exclude_and_inherit_context_keys_combine_in_one_call(self) -> None:
        """A forward_group_exclude and inherit_context_keys work together under the default forward."""
        consumer = Options(
            group={"kg_backend": "neo4j", "query_text": "hello"},
            context={"trace_id": "abc"},
        )
        child = Options()

        child.inherit_from(
            consumer,
            forward_group_exclude=frozenset({"query_text"}),
            inherit_context_keys=frozenset({"trace_id"}),
        )

        assert child.group["kg_backend"] == "neo4j"
        assert "query_text" not in child.group
        assert child.context["trace_id"] == "abc"

    def test_shield_style_frozenset_key_gets_no_special_treatment(self) -> None:
        """
        Pins the shield retirement: a consumer carrying a shield-style frozenset key
        gets no special treatment. With forward_group=True, that key and the keys
        it names are copied like any other group keys (only in_features is skipped).
        """
        shield_key = "legacy_shield"
        consumer = Options(
            group={
                shield_key: frozenset({"shielded_key"}),
                "shielded_key": "shielded_value",
                "kg_backend": "neo4j",
                DefaultOptionKeys.in_features: "consumer_source",
            }
        )
        child = Options()

        child.inherit_from(consumer, forward_group=True)

        # the shield-style key is copied like any other group key
        assert child.group[shield_key] == frozenset({"shielded_key"})
        # keys named by it are NOT shielded from the copy
        assert child.group["shielded_key"] == "shielded_value"
        assert child.group["kg_backend"] == "neo4j"
        # in_features remains the only exclusion
        assert DefaultOptionKeys.in_features not in child.group


class TestInheritFromRichConflictError:
    """
    Test suite for the rich forwarding-conflict error (NEW SPEC).

    When inherit_from would forward key K but self.group[K] already holds a
    DIFFERENT value, the raised ValueError must be actionable: it names the
    key, the consumer's value, the child's existing value, attributes the
    conflict to option forwarding (the word "forwarded"), and names both
    remedies ("forward_group_exclude" and "forward_group=False"). It is raised
    BEFORE add_to_group, replacing the generic validator message for this
    path. Equal values stay a no-op; the group/context cross-conflict path
    keeps its existing validator behavior.
    """

    def test_conflict_message_names_key_and_both_values(self) -> None:
        """The message contains the key, the consumer's value, AND the child's value."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError) as excinfo:
            child.inherit_from(consumer)

        message = str(excinfo.value)
        assert "kg_backend" in message
        assert "neo4j" in message, f"consumer value missing from: {message}"
        assert "memgraph" in message, f"child value missing from: {message}"

    def test_conflict_message_attributes_forwarding(self) -> None:
        """The message says the conflicting key was 'forwarded' (attribution to option forwarding)."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError) as excinfo:
            child.inherit_from(consumer)

        assert "forwarded" in str(excinfo.value)

    def test_conflict_message_names_both_remedies(self) -> None:
        """The message names the remedies: forward_group_exclude and forward_group=False."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError) as excinfo:
            child.inherit_from(consumer)

        message = str(excinfo.value)
        assert "forward_group_exclude" in message
        assert "forward_group=False" in message

    def test_allowlist_conflict_raises_same_rich_message(self) -> None:
        """The rich error also fires on the allowlist path, not only under the default forward."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError) as excinfo:
            child.inherit_from(consumer, forward_group=frozenset({"kg_backend"}))

        message = str(excinfo.value)
        assert "kg_backend" in message
        assert "neo4j" in message
        assert "memgraph" in message
        assert "forwarded" in message
        assert "forward_group_exclude" in message
        assert "forward_group=False" in message

    def test_conflict_leaves_child_group_value_unchanged(self) -> None:
        """Regression pin: the error is raised before any mutation; the child keeps its value."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError):
            child.inherit_from(consumer)

        assert child.group["kg_backend"] == "memgraph"

    def test_equal_value_stays_noop_and_counted_as_inherited(self) -> None:
        """Regression pin: equal values do not raise and still count in inherited_group_keys."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "neo4j"})

        child.inherit_from(consumer)

        assert child.group["kg_backend"] == "neo4j"
        assert child.inherited_group_keys == frozenset({"kg_backend"})

    def test_cross_conflict_keeps_existing_validator_message(self) -> None:
        """Regression pin: a forwarded group key living in child.context keeps the
        validator's cross-conflict behavior, not the rich forwarding message."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(context={"kg_backend": "neo4j"})

        with pytest.raises(ValueError, match="Cannot add to group"):
            child.inherit_from(consumer)


class TestInheritFromOwnerParameter:
    """
    Test suite for the optional owner parameter of inherit_from (NEW SPEC).

    inherit_from gains `owner: str | None = None`, used purely to make the
    rich conflict error identify the child feature (Features.merge_options
    passes str(feature.name)). It changes nothing but the error message.
    """

    def test_signature_owner_default_is_none(self) -> None:
        """The owner parameter exists and defaults to None."""
        from inspect import signature

        default = signature(Options.inherit_from).parameters["owner"].default
        assert default is None

    def test_owner_name_appears_in_conflict_message(self) -> None:
        """A passed owner string shows up in the rich conflict message."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options(group={"kg_backend": "memgraph"})

        with pytest.raises(ValueError) as excinfo:
            child.inherit_from(consumer, owner="my_child_feature")

        assert "my_child_feature" in str(excinfo.value)

    def test_owner_does_not_change_success_behavior(self) -> None:
        """Passing owner on a conflict-free merge changes nothing."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        child.inherit_from(consumer, owner="my_child_feature")

        assert child.group["kg_backend"] == "neo4j"
        assert child.inherited_group_keys == frozenset({"kg_backend"})


class TestMergeOptionsConflictNamesChildFeature:
    """
    Test suite for child-feature attribution through Features.merge_options (NEW SPEC).

    When the collection-level merge (Features.merge_options -> inherit_from)
    hits a forwarding conflict, the raised message must contain the child
    feature's NAME so users can locate the conflicting input feature.
    Design: merge_options passes owner=str(feature.name).
    """

    def test_merge_conflict_message_contains_child_feature_name(self) -> None:
        """The rich conflict error raised through the collection merge names the child feature."""
        consumer = Options(group={"backend": "neo4j"})
        input_feature = Feature("conflicting_child_feature", options={"backend": "postgres"})

        with pytest.raises(ValueError) as excinfo:
            Features([input_feature], child_options=consumer, child_uuid=uuid4())

        assert "conflicting_child_feature" in str(excinfo.value)

    def test_merge_conflict_message_is_the_full_rich_error(self) -> None:
        """The collection-level conflict carries key, both values, attribution, and remedies."""
        consumer = Options(group={"backend": "neo4j"})
        input_feature = Feature("conflicting_child_feature", options={"backend": "postgres"})

        with pytest.raises(ValueError) as excinfo:
            Features([input_feature], child_options=consumer, child_uuid=uuid4())

        message = str(excinfo.value)
        assert "backend" in message
        assert "neo4j" in message
        assert "postgres" in message
        assert "forwarded" in message
        assert "forward_group_exclude" in message
        assert "forward_group=False" in message

    def test_merge_without_conflict_stays_fine(self) -> None:
        """Regression pin: a conflict-free collection merge keeps working with the owner wiring."""
        consumer = Options(group={"backend": "neo4j"})
        input_feature = Feature("clean_child_feature", options={"own_key": "own_value"})

        features = Features([input_feature], child_options=consumer, child_uuid=uuid4())

        merged = features.collection[0]
        assert merged.options.group["backend"] == "neo4j"
        assert merged.options.group["own_key"] == "own_value"


class TestForwardingContradictionGuardBothEntryPoints:
    """
    Regression pins for the forward_group=False + non-empty forward_group_exclude
    contradiction guard (NEW SPEC consolidates it into one shared helper).

    Both live entry points must KEEP raising ValueError with the current message:
    the Feature constructor AND a direct Options.inherit_from call (the latter is
    reached via post-construction attribute assignment). These tests pin behavior
    through both paths so the guard can move into a single shared helper without
    losing either raise.
    """

    def test_feature_constructor_raises_contradiction(self) -> None:
        """Regression pin: Feature(...) rejects the contradictory combination."""
        with pytest.raises(ValueError, match="forward_group=False cannot be combined"):
            Feature("child", forward_group=False, forward_group_exclude={"top_k"})

    def test_inherit_from_direct_call_raises_contradiction(self) -> None:
        """Regression pin: Options.inherit_from rejects the contradictory combination."""
        consumer = Options(group={"kg_backend": "neo4j"})
        child = Options()

        with pytest.raises(ValueError, match="forward_group=False cannot be combined"):
            child.inherit_from(consumer, forward_group=False, forward_group_exclude=frozenset({"kg_backend"}))

    def test_post_construction_assignment_is_caught_by_inherit_from(self) -> None:
        """Regression pin: attributes set after construction bypass the constructor guard
        but are still caught when the collection merge reaches inherit_from."""
        consumer = Options(group={"kg_backend": "neo4j"})
        input_feature = Feature("child")
        input_feature.forward_group = False
        input_feature.forward_group_exclude = frozenset({"kg_backend"})

        with pytest.raises(ValueError, match="forward_group=False cannot be combined"):
            Features([input_feature], child_options=consumer, child_uuid=uuid4())


class TestInheritFromCrossCategoryForwardConflictHint:
    """
    Test suite for the cross-category forwarded-key conflict message (NEW SPEC).

    When a forwarded group key already lives on the CHILD in CONTEXT (not group),
    the group-forward pre-check (which only tests `key in self.group`) misses it,
    and add_to_group raises the terse validator message "already exists in context
    options. Cannot add to group." That message does not tell the user how to opt
    out. The intended behavior: raise a ValueError whose message guides the user to
    the forwarding opt-outs (it mentions `forward_group_exclude`).
    """

    def test_cross_category_conflict_message_hints_forward_group_exclude(self) -> None:
        """A forwarded group key that collides with a child CONTEXT key raises a ValueError
        whose message names the `forward_group_exclude` remedy."""
        child = Options(context={"top_k": 5})
        consumer = Options(group={"top_k": 10})

        with pytest.raises(ValueError, match="forward_group_exclude"):
            child.inherit_from(consumer)


class TestNonForwardedKeysConstant:
    """
    Test suite for the shared never-forwarded key set (NEW SPEC).

    The hardcoded {DefaultOptionKeys.in_features} exclusion set (duplicated in
    Options.inherit_from and identify_feature_group._input_feature_forwarding_hint)
    becomes a public module-level constant NON_FORWARDED_KEYS in options.py.
    """

    def test_constant_is_importable_frozenset_containing_in_features(self) -> None:
        """NON_FORWARDED_KEYS is importable from options.py and contains in_features."""
        from mloda.core.abstract_plugins.components.options import NON_FORWARDED_KEYS

        assert isinstance(NON_FORWARDED_KEYS, frozenset)
        assert DefaultOptionKeys.in_features in NON_FORWARDED_KEYS

    def test_in_features_never_forwarded_regression_pin(self) -> None:
        """Regression pin: in_features never flows through any inherit_from flow."""
        consumer = Options(
            group={DefaultOptionKeys.in_features: "consumer_source", "kg_backend": "neo4j"},
            context={},
        )
        child = Options()

        child.inherit_from(consumer)
        child.inherit_from(consumer, forward_group=frozenset({DefaultOptionKeys.in_features}))

        assert DefaultOptionKeys.in_features not in child.group
        assert DefaultOptionKeys.in_features not in child.context
        assert child.group["kg_backend"] == "neo4j"


class TestInheritFromCrossNamespaceIndependence:
    """
    Regression pins for issue #621: consumer CONTEXT vs child GROUP is independent.

    WHY THIS EXISTS:
    inherit_from forwards group->group by default and only compares forwarded
    consumer GROUP keys against the child's group/context. It NEVER compares a
    consumer CONTEXT key against a child GROUP key in that path (context can still
    flow context->context via inherit_context_keys / propagate_context_keys, but
    that is a different flow). This is deliberate under the
    group/context namespace separation: group drives resolution/splitting, context
    is metadata. So a consumer context and a child group that happen to share a key
    name resolve SILENTLY, the child keeps its own group value, and no ValueError is
    raised. Only this one direction is dropped; the reverse (consumer group vs child
    context) still raises. These pins lock in the asymmetry so a future regression
    that starts cross-comparing namespaces would break them.
    """

    def test_consumer_context_and_child_group_same_name_resolve_silently(self) -> None:
        """The DoD scenario: consumer context algo=sum and child group algo=mean do not clash;
        inherit_from does not raise and the child keeps group['algo'] == 'mean'."""
        consumer = Options(context={"algo": "sum"})
        child = Options(group={"algo": "mean"})

        child.inherit_from(consumer)

        assert child.group["algo"] == "mean"
        assert "algo" not in child.inherited_group_keys
        assert child.last_forwarded_group_keys == frozenset()

    def test_consumer_context_never_leaks_into_child_group(self) -> None:
        """consumer.context is not a source for the child's group: 'algo' stays out of any
        cross-namespace merge, child.context does not gain it, and child.group['algo'] is unchanged."""
        consumer = Options(context={"algo": "sum"})
        child = Options(group={"algo": "mean"})

        child.inherit_from(consumer)

        assert child.group["algo"] == "mean"
        assert "algo" not in child.context

    def test_reverse_direction_still_raises_consumer_group_vs_child_context(self) -> None:
        """The asymmetry guardrail: only context->group is silent. A consumer GROUP key
        forwarded onto a child that holds it in CONTEXT is still a cross-conflict and raises."""
        consumer = Options(group={"algo": "sum"})
        child = Options(context={"algo": "mean"})

        with pytest.raises(ValueError, match="child's context"):
            child.inherit_from(consumer)

    def test_equal_value_cross_namespace_also_silent(self) -> None:
        """Equal values across namespaces are likewise independent, not deduped: consumer
        context algo=mean and child group algo=mean do not raise and the child keeps its value."""
        consumer = Options(context={"algo": "mean"})
        child = Options(group={"algo": "mean"})

        child.inherit_from(consumer)

        assert child.group["algo"] == "mean"


class TestInheritFromAtomicOnConflict:
    """
    Regression pins for issue #623: inherit_from must be ATOMIC on conflict.

    WHY THIS EXISTS:
    The group forward iterates sorted(group_keys) and calls add_to_group per key,
    mutating self.group immediately, while the provenance fields
    (inherited_group_keys, last_forwarded_group_keys) are assigned only AFTER the
    loop. So a mid-loop conflict on a LATER sorted key raises ValueError after an
    EARLIER sorted key was already committed to self.group, leaving self partially
    mutated (polluted group) with stale provenance. The contract SHOULD be atomic:
    a raise leaves self completely unchanged. These pins arrange a multi-key forward
    where an earlier sorted key forwards cleanly and a later key conflicts, then
    assert self is untouched after the raise.
    """

    def test_group_conflict_is_atomic_no_partial_commit(self) -> None:
        """The canonical reproduction: consumer group {'aaa': 1, 'zzz': 9} forwarded onto a
        child holding group {'zzz': 8}. 'aaa' sorts first and forwards cleanly, then 'zzz'
        conflicts and raises. After the raise the earlier 'aaa' must NOT be committed and
        provenance must be untouched (empty, as this is the child's first merge)."""
        consumer = Options(group={"aaa": 1, "zzz": 9})
        child = Options(group={"zzz": 8})

        with pytest.raises(ValueError):
            child.inherit_from(consumer)

        assert child.group == {"zzz": 8}
        assert child.inherited_group_keys == frozenset()
        assert child.last_forwarded_group_keys == frozenset()

    def test_group_conflict_preserves_pre_existing_provenance(self) -> None:
        """A child that already inherited a key from a benign consumer must keep its group
        AND its recorded provenance exactly when a SECOND, conflicting multi-key forward
        raises. The failed call must not touch self at all."""
        benign = Options(group={"benign": 7})
        child = Options(group={"zzz": 8})
        child.inherit_from(benign)

        group_before = deepcopy(child.group)
        inherited_before = child.inherited_group_keys
        last_forwarded_before = child.last_forwarded_group_keys
        assert inherited_before == frozenset({"benign"})
        assert last_forwarded_before == frozenset({"benign"})

        conflicting = Options(group={"aaa": 1, "zzz": 9})
        with pytest.raises(ValueError):
            child.inherit_from(conflicting)

        assert child.group == group_before
        assert child.inherited_group_keys == inherited_before
        assert child.last_forwarded_group_keys == last_forwarded_before

    def test_context_cross_conflict_is_atomic_no_partial_commit(self) -> None:
        """A forwarded group key that collides with an existing child CONTEXT key raises the
        cross-conflict branch. Arrange 'aaa' (sorts first) to forward cleanly into group and
        'zzz' (sorts later) to hit the context cross-conflict. After the raise the earlier
        'aaa' must NOT be committed to group and provenance must be untouched."""
        consumer = Options(group={"aaa": 1, "zzz": 9})
        child = Options(context={"zzz": 8})

        with pytest.raises(ValueError):
            child.inherit_from(consumer)

        assert "aaa" not in child.group
        assert child.group == {}
        assert child.inherited_group_keys == frozenset()
        assert child.last_forwarded_group_keys == frozenset()
