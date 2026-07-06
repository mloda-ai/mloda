"""Tests for allowlist-based input-feature option forwarding (issue #579).

These tests specify the REWORKED merge between a consumer feature (the feature
that declared ``input_features``) and each input feature it returns:

- Default is allowlist / no inheritance: the input feature receives NONE of the
  consumer's group options. ``in_features`` never flows down.
- ``Feature(..., forward_group=...)`` opts specific consumer GROUP keys in
  (a set/frozenset of keys, or True for all keys except ``in_features``).
- ``Feature(..., inherit_context_keys=...)`` is the child-side context pull,
  symmetric to the existing parent-side ``Options(propagate_context_keys=...)``
  push.
- Both new parameters are excluded from identity (``__eq__``/``__hash__``),
  like ``link``, ``index`` and ``feature_group_scope`` already are.

The merge is exercised through ``Features(..., child_options=..., child_uuid=...)``,
which is exactly how the engine wraps declared input features. The merge only
happens when ``child_uuid`` is passed.
"""

from uuid import uuid4

import pytest

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.options import Options


# ---------------------------------------------------------------------------
# Constructor contract: forward_group
# ---------------------------------------------------------------------------


def test_forward_group_defaults_to_none() -> None:
    """Without the new keyword, forward_group is None."""
    feature = Feature("upstream")
    assert feature.forward_group is None


def test_forward_group_stores_set() -> None:
    """A set of keys is accepted and stored on the feature."""
    feature = Feature("upstream", forward_group={"corpus_backend"})
    assert feature.forward_group == {"corpus_backend"}


def test_forward_group_stores_frozenset() -> None:
    """A frozenset of keys is accepted and stored on the feature."""
    feature = Feature("upstream", forward_group=frozenset({"corpus_backend"}))
    assert feature.forward_group == frozenset({"corpus_backend"})


def test_forward_group_stores_true() -> None:
    """forward_group=True (forward all) is accepted and stored."""
    feature = Feature("upstream", forward_group=True)
    assert feature.forward_group is True


def test_forward_group_bare_string_raises_typeerror() -> None:
    """A bare string is ambiguous with an iterable of characters and must raise TypeError.

    The current "unexpected keyword argument 'forward_group'" TypeError must NOT
    satisfy this test: the message has to name forward_group as an invalid value.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("upstream", forward_group="corpus_backend")  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# Constructor contract: inherit_context_keys
# ---------------------------------------------------------------------------


def test_inherit_context_keys_defaults_to_none() -> None:
    """Without the new keyword, inherit_context_keys is None."""
    feature = Feature("upstream")
    assert feature.inherit_context_keys is None


def test_inherit_context_keys_stores_set() -> None:
    """A set of keys is accepted and stored on the feature."""
    feature = Feature("upstream", inherit_context_keys={"trace_id"})
    assert feature.inherit_context_keys == {"trace_id"}


def test_inherit_context_keys_stores_frozenset() -> None:
    """A frozenset of keys is accepted and stored on the feature."""
    feature = Feature("upstream", inherit_context_keys=frozenset({"trace_id"}))
    assert feature.inherit_context_keys == frozenset({"trace_id"})


def test_inherit_context_keys_bare_string_raises_typeerror() -> None:
    """A bare string must raise a TypeError naming inherit_context_keys."""
    with pytest.raises(TypeError) as exc_info:
        Feature("upstream", inherit_context_keys="trace_id")  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "inherit_context_keys" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# Identity invariants: both parameters are excluded from equality / hash
# ---------------------------------------------------------------------------


def test_forward_group_excluded_from_eq_and_hash() -> None:
    """A feature with forward_group equals and hashes like one without it."""
    forwarding = Feature("upstream", forward_group={"corpus_backend"})
    plain = Feature("upstream")
    assert forwarding == plain
    assert hash(forwarding) == hash(plain)


def test_different_forward_groups_are_equal() -> None:
    """Two features with different forward_group values are still equal and hash equal."""
    feature_a = Feature("upstream", forward_group={"corpus_backend"})
    feature_b = Feature("upstream", forward_group=True)
    assert feature_a == feature_b
    assert hash(feature_a) == hash(feature_b)


def test_inherit_context_keys_excluded_from_eq_and_hash() -> None:
    """A feature with inherit_context_keys equals and hashes like one without it."""
    inheriting = Feature("upstream", inherit_context_keys={"trace_id"})
    plain = Feature("upstream")
    assert inheriting == plain
    assert hash(inheriting) == hash(plain)


def test_forwarding_and_plain_collapse_in_set() -> None:
    """Forwarding and plain features with the same name collapse to ONE set element."""
    collection = {
        Feature("upstream", forward_group={"corpus_backend"}, inherit_context_keys={"trace_id"}),
        Feature("upstream"),
    }
    assert len(collection) == 1


# ---------------------------------------------------------------------------
# Default merge behaviour: allowlist means NOTHING is inherited
# ---------------------------------------------------------------------------


def test_default_no_group_options_inherited() -> None:
    """By default an input feature receives none of the consumer's group options."""
    consumer = Options(group={"corpus_backend": "faiss", "top_k": 9})
    child = Feature("upstream")
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {}


def test_in_features_never_flows_down_by_default() -> None:
    """The consumer's in_features key must not land on the input feature."""
    consumer = Options(group={DefaultOptionKeys.in_features: frozenset({"upstream"})})
    child = Feature("upstream")
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert DefaultOptionKeys.in_features not in child.options.group
    assert DefaultOptionKeys.in_features not in child.options.context


def test_default_no_context_options_inherited() -> None:
    """Consumer context keys are not copied without propagate/inherit opt-in."""
    consumer = Options(context={"trace_id": "abc"})
    child = Feature("upstream")
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.context == {}


def test_unrelated_same_key_no_longer_conflicts() -> None:
    """Consumer and input feature may hold the SAME group key with DIFFERENT values.

    Since nothing is merged by default, no ValueError is raised and the input
    feature keeps its own value untouched.
    """
    consumer = Options(group={"top_k": 9})
    child = Feature("upstream", {"top_k": 5})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {"top_k": 5}


def test_forward_group_false_means_no_forwarding() -> None:
    """forward_group=False behaves like None: nothing is forwarded."""
    consumer = Options(group={"corpus_backend": "faiss"})
    child = Feature("upstream", forward_group=False)
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {}


# ---------------------------------------------------------------------------
# Selective forwarding: forward_group as a key allowlist
# ---------------------------------------------------------------------------


def test_forward_group_copies_only_listed_keys() -> None:
    """Only the listed consumer group keys are copied; others stay behind."""
    consumer = Options(group={"corpus_backend": "faiss", "top_k": 9})
    child = Feature("upstream", forward_group={"corpus_backend"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {"corpus_backend": "faiss"}


def test_forward_group_missing_key_skipped_silently() -> None:
    """Listed keys absent on the consumer are skipped without error."""
    consumer = Options(group={"top_k": 9})
    child = Feature("upstream", forward_group={"corpus_backend"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {}


def test_forward_group_reads_group_not_context() -> None:
    """forward_group only reads the consumer's GROUP options, never its context."""
    consumer = Options(context={"corpus_backend": "faiss"})
    child = Feature("upstream", forward_group={"corpus_backend"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {}
    assert child.options.context == {}


def test_forward_group_conflict_raises_valueerror() -> None:
    """A forwarded key already present with a DIFFERENT value raises ValueError."""
    consumer = Options(group={"corpus_backend": "faiss"})
    child = Feature("upstream", {"corpus_backend": "qdrant"}, forward_group={"corpus_backend"})
    with pytest.raises(ValueError):
        Features([child], child_options=consumer, child_uuid=uuid4())


def test_forward_group_same_value_no_error() -> None:
    """A forwarded key already present with the SAME value does not raise."""
    consumer = Options(group={"corpus_backend": "faiss"})
    child = Feature("upstream", {"corpus_backend": "faiss"}, forward_group={"corpus_backend"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {"corpus_backend": "faiss"}


# ---------------------------------------------------------------------------
# Forward all: forward_group=True
# ---------------------------------------------------------------------------


def test_forward_group_true_copies_all_keys_except_in_features() -> None:
    """forward_group=True copies every consumer group key EXCEPT in_features."""
    consumer = Options(
        group={
            "corpus_backend": "faiss",
            "top_k": 9,
            DefaultOptionKeys.in_features: frozenset({"upstream"}),
        }
    )
    child = Feature("upstream", forward_group=True)
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.group == {"corpus_backend": "faiss", "top_k": 9}
    assert DefaultOptionKeys.in_features not in child.options.group


def test_forward_group_true_conflict_raises_valueerror() -> None:
    """forward_group=True with a differing existing value still raises ValueError."""
    consumer = Options(group={"top_k": 9})
    child = Feature("upstream", {"top_k": 5}, forward_group=True)
    with pytest.raises(ValueError):
        Features([child], child_options=consumer, child_uuid=uuid4())


# ---------------------------------------------------------------------------
# Child-side context pull: inherit_context_keys
# ---------------------------------------------------------------------------


def test_inherit_context_keys_copies_only_listed_keys() -> None:
    """Only the listed consumer context keys are pulled into the input feature."""
    consumer = Options(context={"trace_id": "abc", "other": "x"})
    child = Feature("upstream", inherit_context_keys={"trace_id"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.context == {"trace_id": "abc"}


def test_inherit_context_keys_missing_key_skipped_silently() -> None:
    """Listed context keys absent on the consumer are skipped without error."""
    consumer = Options(context={"other": "x"})
    child = Feature("upstream", inherit_context_keys={"trace_id"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.context == {}


def test_inherit_context_keys_conflict_raises_valueerror() -> None:
    """An inherited key already present with a DIFFERENT value raises ValueError."""
    consumer = Options(context={"trace_id": "abc"})
    child = Feature("upstream", Options(context={"trace_id": "def"}), inherit_context_keys={"trace_id"})
    with pytest.raises(ValueError):
        Features([child], child_options=consumer, child_uuid=uuid4())


def test_inherit_context_keys_same_value_no_error() -> None:
    """An inherited key already present with the SAME value does not raise."""
    consumer = Options(context={"trace_id": "abc"})
    child = Feature("upstream", Options(context={"trace_id": "abc"}), inherit_context_keys={"trace_id"})
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.context == {"trace_id": "abc"}


# ---------------------------------------------------------------------------
# Existing consumer-side push (propagate_context_keys) still works
# ---------------------------------------------------------------------------


def test_propagate_context_keys_push_still_works() -> None:
    """The consumer-side propagate_context_keys push still lands keys on the child."""
    consumer = Options(context={"trace_id": "abc"}, propagate_context_keys=frozenset({"trace_id"}))
    child = Feature("upstream")
    Features([child], child_options=consumer, child_uuid=uuid4())
    assert child.options.context.get("trace_id") == "abc"


# ---------------------------------------------------------------------------
# Merge only happens when child_uuid is passed
# ---------------------------------------------------------------------------


def test_no_merge_without_child_uuid() -> None:
    """Without child_uuid the Features plumbing must not forward anything."""
    consumer = Options(group={"corpus_backend": "faiss"}, context={"trace_id": "abc"})
    child = Feature("upstream", forward_group=True, inherit_context_keys={"trace_id"})
    Features([child], child_options=consumer)
    assert child.options.group == {}
    assert child.options.context == {}
