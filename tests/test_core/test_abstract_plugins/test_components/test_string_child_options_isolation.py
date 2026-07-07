"""Unifying string-declared and object input-feature option forwarding.

These tests pin the INTENDED behavior after the string/object forwarding paths
are unified. Today a string-declared input feature is built with the consumer's
own Options instance, so ``Options.inherit_from`` takes its ``consumer is self``
self-merge no-op: string children alias the consumer's group/context dicts,
record no provenance, and share nested value objects with the consumer and each
other. Object children go through the real forwarding merge but forward nested
values BY REFERENCE (issue #607).

The unified contract these tests specify:

1. A string-declared input feature owns its OWN Options instance, never the
   consumer's and never shared with a sibling string child.
2. A string child goes through the SAME real forwarding merge as an object
   child, so it RECORDS provenance: ``inherited_group_keys`` equals the set of
   forwarded consumer group keys and ``last_forwarded_group_keys`` is populated.
   (Deliberate change: provenance is NO LONGER empty for string children.)
3. Forwarded values are DEEP-copied, so mutating a forwarded value on a child
   never leaks to the consumer or a sibling, including NESTED containers. This
   isolation holds for BOTH string children and object children (the #607 fix,
   applied uniformly).
4. Forwarded group and context values are still preserved on the child.
5. A consumer option whose value has identity-based ``__eq__`` (``object()``)
   must not make ``inherit_from`` raise while constructing a string child.
"""

from __future__ import annotations

from uuid import uuid4

from mloda.user import Feature
from mloda.user import Features
from mloda.user import Options


def _string_children(consumer: Options, names: list[str]) -> dict[str, Feature]:
    """Build a Features collection from string names and index children by name."""
    features = Features(list(names), child_options=consumer, child_uuid=uuid4())
    return {str(f.name): f for f in features.collection}


# ---------------------------------------------------------------------------
# 1. Distinct Options instances
# ---------------------------------------------------------------------------


def test_string_children_get_distinct_options_instances() -> None:
    """Each string child owns its own Options, not the consumer and not a sibling's."""
    consumer = Options(group={"gk": "gv"})

    children = _string_children(consumer, ["alpha", "beta"])
    alpha = children["alpha"]
    beta = children["beta"]

    assert alpha.options is not consumer
    assert beta.options is not consumer
    assert alpha.options is not beta.options


# ---------------------------------------------------------------------------
# 2. Top-level mutation isolation
# ---------------------------------------------------------------------------


def test_top_level_group_mutation_does_not_leak_to_consumer_or_sibling() -> None:
    """Rebinding a forwarded top-level group key on one child leaves others untouched."""
    consumer = Options(group={"gk": "gv"})

    children = _string_children(consumer, ["alpha", "beta"])
    alpha = children["alpha"]
    beta = children["beta"]

    alpha.options.group["gk"] = "mutated"

    assert consumer.group["gk"] == "gv"
    assert beta.options.group["gk"] == "gv"


# ---------------------------------------------------------------------------
# 3. Nested mutation isolation (the #607 fix on the string path)
# ---------------------------------------------------------------------------


def test_nested_group_mutation_does_not_leak_to_consumer_or_sibling() -> None:
    """Mutating a forwarded NESTED group value on one child leaves others untouched."""
    consumer = Options(group={"cfg": {"a": 1}})

    children = _string_children(consumer, ["alpha", "beta"])
    alpha = children["alpha"]
    beta = children["beta"]

    alpha.options.group["cfg"]["a"] = 999

    assert consumer.group["cfg"]["a"] == 1
    assert beta.options.group["cfg"]["a"] == 1


def test_nested_context_mutation_does_not_leak_to_consumer_or_sibling() -> None:
    """Mutating a forwarded NESTED context value on one child leaves others untouched."""
    consumer = Options(
        group={"gk": "gv"},
        context={"cfg": {"a": 1}},
        propagate_context_keys=frozenset({"cfg"}),
    )

    children = _string_children(consumer, ["alpha", "beta"])
    alpha = children["alpha"]
    beta = children["beta"]

    alpha.options.context["cfg"]["a"] = 999

    assert consumer.context["cfg"]["a"] == 1
    assert beta.options.context["cfg"]["a"] == 1


# ---------------------------------------------------------------------------
# 4. Forwarded values preserved (as deep copies, not aliases)
# ---------------------------------------------------------------------------


def test_forwarded_group_and_context_values_preserved_as_deep_copies() -> None:
    """Forwarded group/context values are kept on the child but are distinct objects."""
    consumer = Options(
        group={"gk": "gv", "cfg": {"a": 1}},
        context={"ctx": {"b": 2}},
        propagate_context_keys=frozenset({"ctx"}),
    )

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    assert alpha.options.group["gk"] == "gv"
    assert alpha.options.group["cfg"] == {"a": 1}
    assert alpha.options.context["ctx"] == {"b": 2}

    assert alpha.options.group["cfg"] is not consumer.group["cfg"]
    assert alpha.options.context["ctx"] is not consumer.context["ctx"]


# ---------------------------------------------------------------------------
# 5. Provenance populated for string children
# ---------------------------------------------------------------------------


def test_string_child_records_forwarding_provenance() -> None:
    """A string child records the forwarded group keys in its provenance sets."""
    consumer = Options(group={"gk": "gv", "cfg": {"a": 1}})

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    assert alpha.options.inherited_group_keys == frozenset({"gk", "cfg"})
    assert alpha.options.last_forwarded_group_keys == frozenset({"gk", "cfg"})


# ---------------------------------------------------------------------------
# 6. Identity-based __eq__ consumer value does not raise
# ---------------------------------------------------------------------------


def test_identity_eq_consumer_value_does_not_break_string_child() -> None:
    """A consumer value with identity __eq__ (object()) forwards without raising,
    and the TOP-LEVEL opaque value is SHARED by reference (identity preserved)."""
    sentinel = object()
    consumer = Options(group={"gk": "gv", "tok": sentinel})

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    assert alpha.options is not consumer
    assert "tok" in alpha.options.group
    assert alpha.options.inherited_group_keys == frozenset({"gk", "tok"})
    # A top-level opaque leaf must remain the SAME object on the child so identity
    # (for models/validators/handles and Options dedup) is preserved.
    assert alpha.options.group["tok"] is sentinel


# ---------------------------------------------------------------------------
# 8. Nested opaque-leaf identity preserved (container SPINE copied, leaves shared)
# ---------------------------------------------------------------------------


def test_nested_opaque_leaf_identity_preserved_while_spine_is_fresh() -> None:
    """An opaque leaf nested inside a forwarded list AND dict stays the SAME object on
    the child, while the mutable container spines around it are fresh copies."""
    sentinel = object()
    consumer = Options(group={"models": [sentinel], "wrap": {"m": sentinel}})

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    # Leaves shared by reference: identity preserved through nested containers.
    assert alpha.options.group["models"][0] is sentinel
    assert alpha.options.group["wrap"]["m"] is sentinel

    # Container spines are fresh, not aliases of the consumer's containers.
    assert alpha.options.group["models"] is not consumer.group["models"]
    assert alpha.options.group["wrap"] is not consumer.group["wrap"]


# ---------------------------------------------------------------------------
# 9. Nested container mutation still isolated (regression guard for the spine copy)
# ---------------------------------------------------------------------------


def test_nested_container_mutation_still_isolated_after_spine_copy() -> None:
    """Mutating a forwarded nested container on one child does not leak to the consumer
    or a sibling, even though nested leaves are shared by reference."""
    consumer = Options(group={"models": [1, 2], "wrap": {"m": 0}})

    children = _string_children(consumer, ["alpha", "beta"])
    alpha = children["alpha"]
    beta = children["beta"]

    alpha.options.group["wrap"]["m2"] = 1
    alpha.options.group["models"].append(999)

    assert consumer.group["wrap"] == {"m": 0}
    assert consumer.group["models"] == [1, 2]
    assert beta.options.group["wrap"] == {"m": 0}
    assert beta.options.group["models"] == [1, 2]


# ---------------------------------------------------------------------------
# 10. Dedup preserved: shared identity leaves keep siblings Options-equal
# ---------------------------------------------------------------------------


def test_dedup_preserved_for_siblings_sharing_identity_leaf() -> None:
    """Two string children forwarding the SAME container-of-identity-objects from the
    SAME consumer stay equal and hash-equal at the Options level (dedup preserved)."""
    consumer = Options(group={"models": [object()]})

    children = _string_children(consumer, ["alpha", "beta"])

    assert children["alpha"].options == children["beta"].options
    assert hash(children["alpha"].options) == hash(children["beta"].options)


# ---------------------------------------------------------------------------
# 11. Tuple-nested mutable container is isolated (spine copy recurses into tuples)
# ---------------------------------------------------------------------------


def test_tuple_nested_mutable_container_is_isolated() -> None:
    """A mutable container nested inside a forwarded tuple is spine-copied, so mutating
    it on the child does not leak back to the consumer."""
    consumer = Options(group={"tup": ("a", [1, 2])})

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    alpha.options.group["tup"][1].append(999)

    assert consumer.group["tup"][1] == [1, 2]


# ---------------------------------------------------------------------------
# 12. Unpicklable leaf inside a container does not reopen the leak
# ---------------------------------------------------------------------------


def test_unpicklable_leaf_does_not_reopen_container_leak() -> None:
    """A non-deepcopyable leaf (threading.Lock) inside a forwarded dict must not force
    the whole container to be shared: the container spine is still copied (mutation
    isolated) while the unpicklable leaf itself is shared by reference."""
    import threading

    lock = threading.Lock()
    consumer = Options(group={"cfg": {"lock": lock, "a": 1}})

    children = _string_children(consumer, ["alpha"])
    alpha = children["alpha"]

    alpha.options.group["cfg"]["a"] = 999

    # Container spine copied: the write does not leak to the consumer.
    assert consumer.group["cfg"]["a"] == 1
    # The unpicklable leaf is shared by reference (identity preserved).
    assert alpha.options.group["cfg"]["lock"] is consumer.group["cfg"]["lock"]


# ---------------------------------------------------------------------------
# 7. Object child nested isolation (#607 on the general path)
# ---------------------------------------------------------------------------


def test_object_child_nested_group_mutation_does_not_leak_to_consumer() -> None:
    """A Feature-object child also deep-copies forwarded nested group values."""
    consumer = Options(group={"cfg": {"a": 1}})

    features = Features([Feature("alpha")], child_options=consumer, child_uuid=uuid4())
    alpha = features.collection[0]

    alpha.options.group["cfg"]["a"] = 999

    assert consumer.group["cfg"]["a"] == 1
