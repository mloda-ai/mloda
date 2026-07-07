"""Tests for the input-feature option forwarding parameters (issue #579).

These tests specify three OPTIONAL keyword parameters on ``Feature``:

    Feature(
        "x",
        forward_group={"data_source"},
        forward_group_exclude={"tenant"},
        inherit_context_keys={"run_id"},
    )

The parameters only carry meaning on features returned from a feature
group's ``input_features``: the engine uses them to decide which of the
consuming feature's options are inherited by the input feature.
``forward_group`` allowlists group-option keys (with the sentinels
``None`` = unspecified and ``True`` = inherit ALL consumer group options,
``False`` = EXPLICIT opt-out, inherit nothing), ``forward_group_exclude``
subtracts keys from whatever set ``forward_group`` produced, and
``inherit_context_keys`` allowlists context-option keys.

Contract under test:

* stored as ``feature.forward_group`` / ``feature.forward_group_exclude``
  / ``feature.inherit_context_keys``
* defaults: ``forward_group is None``, ``forward_group_exclude ==
  frozenset()``, ``inherit_context_keys == frozenset()``
* set/frozenset/list/tuple of str normalise to ``frozenset[str]``;
  ``forward_group_exclude=None`` normalises to an empty frozenset
* bare ``str`` or non-iterable non-bool values raise ``TypeError`` naming
  the parameter (a str would otherwise silently iterate into characters)
* containers with non-str ELEMENTS raise ``TypeError`` naming the parameter
* ``forward_group=False`` combined with a non-empty
  ``forward_group_exclude`` raises ``ValueError`` (contradictory directive)
* all three attributes are EXCLUDED from ``__eq__`` and ``__hash__``,
  exactly like the existing ``link`` and ``index`` attributes

The file also pins the consumer attribution metadata that supports the
dual-consumption warning: ``feature.consumer_attributions`` is a list of
``(consumer_class_name, forwarded_declared_keys)`` tuples appended by the
engine, one entry PER consumer feature group that declares the feature.
``forwarded_declared_keys`` is the set of group keys THIS consumer actually
forwarded onto the child (its ``Options.last_forwarded_group_keys``)
INTERSECTED with the consumer's own PROPERTY_MAPPING keys, so a consumer that
merely DECLARES a key but never forwards it records an empty set and is not
warned. Entries are appended through the idempotent
``feature.add_consumer_attribution(name, keys)`` helper: an identical
``(name, keys)`` entry already present is not appended again (bounding growth
when one Feature instance is reused across mloda runs), while a distinct entry
appends. The list replaces the former scalar ``consumer_feature_group_name`` /
``consumer_property_keys`` pair, whose per-consumer overwrite lost the first
consumer's provenance when one child instance was shared by two consumers.
Excluded from ``__eq__`` and ``__hash__`` like link and index.
"""

import pytest

from mloda.user import Feature


# ---------------------------------------------------------------------------
# Constructor contract: defaults
# ---------------------------------------------------------------------------


def test_forward_group_defaults_to_none() -> None:
    """Without the new keyword, forward_group is the unspecified sentinel None."""
    feature = Feature("subject_token")
    assert feature.forward_group is None


def test_forward_group_exclude_defaults_to_empty_frozenset() -> None:
    """Without the new keyword, forward_group_exclude is an empty frozenset."""
    feature = Feature("subject_token")
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset()


def test_inherit_context_keys_defaults_to_empty_frozenset() -> None:
    """Without the new keyword, inherit_context_keys is an empty frozenset."""
    feature = Feature("subject_token")
    assert isinstance(feature.inherit_context_keys, frozenset)
    assert feature.inherit_context_keys == frozenset()


# ---------------------------------------------------------------------------
# Constructor contract: forward_group normalisation
# ---------------------------------------------------------------------------


def test_forward_group_set_normalises_to_frozenset() -> None:
    """A set of str is stored as a frozenset[str]."""
    feature = Feature("x", forward_group={"data_source", "tenant"})
    assert isinstance(feature.forward_group, frozenset)
    assert feature.forward_group == frozenset({"data_source", "tenant"})


def test_forward_group_frozenset_stays_frozenset() -> None:
    """A frozenset of str is stored as a frozenset[str]."""
    feature = Feature("x", forward_group=frozenset({"data_source"}))
    assert isinstance(feature.forward_group, frozenset)
    assert feature.forward_group == frozenset({"data_source"})


def test_forward_group_list_normalises_to_frozenset() -> None:
    """A list of str is normalised to a frozenset[str]."""
    feature = Feature("x", forward_group=["data_source", "tenant"])
    assert isinstance(feature.forward_group, frozenset)
    assert feature.forward_group == frozenset({"data_source", "tenant"})


def test_forward_group_tuple_normalises_to_frozenset() -> None:
    """A tuple of str is normalised to a frozenset[str]."""
    feature = Feature("x", forward_group=("data_source",))
    assert isinstance(feature.forward_group, frozenset)
    assert feature.forward_group == frozenset({"data_source"})


def test_forward_group_true_stays_true() -> None:
    """forward_group=True stays the bool sentinel True (inherit all group options)."""
    feature = Feature("x", forward_group=True)
    assert feature.forward_group is True


def test_forward_group_false_stays_false() -> None:
    """forward_group=False stays the bool sentinel False (EXPLICIT opt-out, inherit nothing)."""
    feature = Feature("x", forward_group=False)
    assert feature.forward_group is False


def test_forward_group_none_stays_none() -> None:
    """forward_group=None stays None (unspecified; the engine forwards all group options)."""
    feature = Feature("x", forward_group=None)
    assert feature.forward_group is None


# ---------------------------------------------------------------------------
# Constructor contract: forward_group_exclude normalisation
# ---------------------------------------------------------------------------


def test_forward_group_exclude_set_normalises_to_frozenset() -> None:
    """A set of str is stored as a frozenset[str]."""
    feature = Feature("x", forward_group_exclude={"tenant", "region"})
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset({"tenant", "region"})


def test_forward_group_exclude_frozenset_stays_frozenset() -> None:
    """A frozenset of str is stored as a frozenset[str]."""
    feature = Feature("x", forward_group_exclude=frozenset({"tenant"}))
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset({"tenant"})


def test_forward_group_exclude_list_normalises_to_frozenset() -> None:
    """A list of str is normalised to a frozenset[str]."""
    feature = Feature("x", forward_group_exclude=["tenant", "region"])
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset({"tenant", "region"})


def test_forward_group_exclude_tuple_normalises_to_frozenset() -> None:
    """A tuple of str is normalised to a frozenset[str]."""
    feature = Feature("x", forward_group_exclude=("tenant",))
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset({"tenant"})


def test_forward_group_exclude_none_normalises_to_empty_frozenset() -> None:
    """forward_group_exclude=None normalises to an empty frozenset, not None."""
    feature = Feature("x", forward_group_exclude=None)
    assert isinstance(feature.forward_group_exclude, frozenset)
    assert feature.forward_group_exclude == frozenset()


# ---------------------------------------------------------------------------
# Constructor guard: forward_group=False plus a non-empty exclude is
# contradictory (opt out of everything AND opt out of single keys)
# ---------------------------------------------------------------------------


def test_forward_group_false_with_nonempty_exclude_raises_valueerror() -> None:
    """forward_group=False together with a non-empty forward_group_exclude raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        Feature("x", forward_group=False, forward_group_exclude={"tenant"})
    message = str(exc_info.value)
    assert "forward_group" in message


def test_forward_group_false_with_empty_exclude_is_fine() -> None:
    """forward_group=False with an empty forward_group_exclude constructs without error."""
    feature = Feature("x", forward_group=False, forward_group_exclude=frozenset())
    assert feature.forward_group is False
    assert feature.forward_group_exclude == frozenset()


def test_forward_group_allowlist_with_exclude_is_allowed() -> None:
    """An allowlist combined with an exclude is a valid subtraction directive."""
    feature = Feature(
        "x",
        forward_group={"data_source", "tenant"},
        forward_group_exclude={"tenant"},
    )
    assert feature.forward_group == frozenset({"data_source", "tenant"})
    assert feature.forward_group_exclude == frozenset({"tenant"})


# ---------------------------------------------------------------------------
# Constructor contract: inherit_context_keys normalisation
# ---------------------------------------------------------------------------


def test_inherit_context_keys_set_normalises_to_frozenset() -> None:
    """A set of str is stored as a frozenset[str]."""
    feature = Feature("x", inherit_context_keys={"run_id", "debug"})
    assert isinstance(feature.inherit_context_keys, frozenset)
    assert feature.inherit_context_keys == frozenset({"run_id", "debug"})


def test_inherit_context_keys_frozenset_stays_frozenset() -> None:
    """A frozenset of str is stored as a frozenset[str]."""
    feature = Feature("x", inherit_context_keys=frozenset({"run_id"}))
    assert isinstance(feature.inherit_context_keys, frozenset)
    assert feature.inherit_context_keys == frozenset({"run_id"})


def test_inherit_context_keys_list_normalises_to_frozenset() -> None:
    """A list of str is normalised to a frozenset[str]."""
    feature = Feature("x", inherit_context_keys=["run_id", "debug"])
    assert isinstance(feature.inherit_context_keys, frozenset)
    assert feature.inherit_context_keys == frozenset({"run_id", "debug"})


def test_inherit_context_keys_tuple_normalises_to_frozenset() -> None:
    """A tuple of str is normalised to a frozenset[str]."""
    feature = Feature("x", inherit_context_keys=("run_id",))
    assert isinstance(feature.inherit_context_keys, frozenset)
    assert feature.inherit_context_keys == frozenset({"run_id"})


# ---------------------------------------------------------------------------
# Constructor guard: bare str and non-iterable non-bool values are rejected
# ---------------------------------------------------------------------------


def test_forward_group_bare_str_raises_typeerror() -> None:
    """A bare str forward_group raises TypeError naming the parameter.

    A str would silently iterate into single characters, turning
    forward_group='data_source' into an allowlist of letters.
    The guard also rejects the pre-implementation 'unexpected keyword'
    TypeError so this test cannot pass before the parameter exists.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group="data_source")  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group" in message
    assert "unexpected keyword" not in message


def test_inherit_context_keys_bare_str_raises_typeerror() -> None:
    """A bare str inherit_context_keys raises TypeError naming the parameter."""
    with pytest.raises(TypeError) as exc_info:
        Feature("x", inherit_context_keys="run_id")  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "inherit_context_keys" in message
    assert "unexpected keyword" not in message


def test_forward_group_exclude_bare_str_raises_typeerror() -> None:
    """A bare str forward_group_exclude raises TypeError naming the parameter."""
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group_exclude="tenant")  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group_exclude" in message
    assert "unexpected keyword" not in message


def test_forward_group_int_raises_typeerror() -> None:
    """A non-iterable non-bool forward_group (int) raises TypeError naming the parameter.

    bool is a subclass of int, so the implementation must accept True/False
    while still rejecting other ints like 123.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group=123)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group" in message
    assert "unexpected keyword" not in message


def test_inherit_context_keys_int_raises_typeerror() -> None:
    """A non-iterable non-bool inherit_context_keys (int) raises TypeError naming the parameter."""
    with pytest.raises(TypeError) as exc_info:
        Feature("x", inherit_context_keys=123)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "inherit_context_keys" in message
    assert "unexpected keyword" not in message


def test_forward_group_exclude_int_raises_typeerror() -> None:
    """A non-iterable forward_group_exclude (int) raises TypeError naming the parameter.

    Unlike forward_group, exclude has no bool sentinels: only containers of
    str (or None) are accepted.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group_exclude=123)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group_exclude" in message
    assert "unexpected keyword" not in message


def test_forward_group_non_str_elements_raise_typeerror() -> None:
    """A container with non-str elements raises TypeError naming forward_group.

    An accepted container type (set) must still be rejected when its ELEMENTS
    are not str; otherwise {1, 2} silently becomes a useless allowlist that
    never matches any option key.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group={1, 2})  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group" in message
    assert "unexpected keyword" not in message


def test_inherit_context_keys_non_str_elements_raise_typeerror() -> None:
    """A container with non-str elements raises TypeError naming inherit_context_keys."""
    with pytest.raises(TypeError) as exc_info:
        Feature("x", inherit_context_keys={1})  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "inherit_context_keys" in message
    assert "unexpected keyword" not in message


def test_forward_group_exclude_non_str_elements_raise_typeerror() -> None:
    """A container with non-str elements raises TypeError naming forward_group_exclude."""
    with pytest.raises(TypeError) as exc_info:
        Feature("x", forward_group_exclude={1, 2})  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "forward_group_exclude" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# Identity invariants: all three attributes are excluded from __eq__ and
# __hash__, exactly like the existing link and index attributes.
# ---------------------------------------------------------------------------


def test_forward_group_excluded_from_eq_and_hash() -> None:
    """Two same-name features differing only in forward_group are equal and hash equal."""
    forwarding = Feature("subject_token", forward_group={"data_source"})
    plain = Feature("subject_token")
    assert forwarding == plain
    assert hash(forwarding) == hash(plain)


def test_forward_group_true_excluded_from_eq_and_hash() -> None:
    """The bool sentinel True is also excluded from equality and hash."""
    forwarding = Feature("subject_token", forward_group=True)
    plain = Feature("subject_token")
    assert forwarding == plain
    assert hash(forwarding) == hash(plain)


def test_forward_group_exclude_excluded_from_eq_and_hash() -> None:
    """Two same-name features differing only in forward_group_exclude are equal and hash equal."""
    excluding = Feature("subject_token", forward_group_exclude={"tenant"})
    plain = Feature("subject_token")
    assert excluding == plain
    assert hash(excluding) == hash(plain)


def test_inherit_context_keys_excluded_from_eq_and_hash() -> None:
    """Two same-name features differing only in inherit_context_keys are equal and hash equal."""
    inheriting = Feature("subject_token", inherit_context_keys={"run_id"})
    plain = Feature("subject_token")
    assert inheriting == plain
    assert hash(inheriting) == hash(plain)


def test_different_allowlists_are_equal_with_same_options() -> None:
    """Same name/options but different allowlists: still equal with equal hashes."""
    feature_a = Feature(
        "subject_token",
        options={"data_source": "prod"},
        forward_group={"data_source"},
        forward_group_exclude={"tenant"},
        inherit_context_keys={"run_id"},
    )
    feature_b = Feature(
        "subject_token",
        options={"data_source": "prod"},
        forward_group=True,
        forward_group_exclude=frozenset(),
        inherit_context_keys=frozenset(),
    )
    assert feature_a == feature_b
    assert hash(feature_a) == hash(feature_b)


def test_allowlisted_and_plain_collapse_in_set() -> None:
    """An allowlisted and an otherwise-identical plain feature collapse to ONE set element."""
    collection = {
        Feature(
            "subject_token",
            forward_group={"data_source"},
            forward_group_exclude={"tenant"},
            inherit_context_keys={"run_id"},
        ),
        Feature("subject_token"),
    }
    assert len(collection) == 1


# ---------------------------------------------------------------------------
# Consumer attribution: resolution-only metadata stamped by the engine, one
# (consumer class name, forwarded_declared_keys) entry appended per consumer
# feature group that declares the feature as an input feature.
# forwarded_declared_keys = the keys THIS consumer actually forwarded onto the
# child INTERSECTED with the consumer's PROPERTY_MAPPING keys, so a consumer
# that declares but does not forward a key records an empty set (and is not
# warned). Entries are appended through the idempotent
# add_consumer_attribution helper; a LIST (instead of two scalar attributes)
# keeps every consumer's provenance when one child instance is shared by
# several consumers. Excluded from __eq__ and __hash__ like link and index.
# ---------------------------------------------------------------------------


def test_consumer_attributions_defaults_to_empty_list() -> None:
    """A freshly constructed feature carries an empty consumer_attributions list."""
    feature = Feature("subject_token")
    assert isinstance(feature.consumer_attributions, list)
    assert feature.consumer_attributions == []


def test_add_consumer_attribution_accumulates_entries_in_order() -> None:
    """Distinct entries added per consumer are all kept, in add order."""
    feature = Feature("subject_token")

    feature.add_consumer_attribution("ConsumerAGroup", frozenset({"mode"}))
    feature.add_consumer_attribution("ConsumerBGroup", frozenset({"mode", "salt"}))

    assert feature.consumer_attributions == [
        ("ConsumerAGroup", frozenset({"mode"})),
        ("ConsumerBGroup", frozenset({"mode", "salt"})),
    ]


def test_add_consumer_attribution_is_idempotent_for_identical_entry() -> None:
    """Adding an identical (name, keys) entry twice yields ONE entry.

    Bounds unbounded growth when the same Feature instance is reused across
    several mloda runs and the engine re-stamps the same attribution.
    """
    feature = Feature("subject_token")

    feature.add_consumer_attribution("ConsumerAGroup", frozenset({"mode"}))
    feature.add_consumer_attribution("ConsumerAGroup", frozenset({"mode"}))

    assert feature.consumer_attributions == [("ConsumerAGroup", frozenset({"mode"}))]


def test_add_consumer_attribution_appends_distinct_entries() -> None:
    """A different name OR a different key set is a distinct entry and appends."""
    feature = Feature("subject_token")

    feature.add_consumer_attribution("ConsumerAGroup", frozenset({"mode"}))
    feature.add_consumer_attribution("ConsumerAGroup", frozenset({"mode", "salt"}))  # different keys
    feature.add_consumer_attribution("ConsumerBGroup", frozenset({"mode"}))  # different name

    assert feature.consumer_attributions == [
        ("ConsumerAGroup", frozenset({"mode"})),
        ("ConsumerAGroup", frozenset({"mode", "salt"})),
        ("ConsumerBGroup", frozenset({"mode"})),
    ]


def test_consumer_attributions_excluded_from_eq_and_hash() -> None:
    """Two same-name features differing only in consumer_attributions are equal and hash equal."""
    attributed = Feature("subject_token")
    attributed.add_consumer_attribution("ConsumerAGroup", frozenset({"mode"}))
    plain = Feature("subject_token")

    assert attributed == plain
    assert hash(attributed) == hash(plain)


def test_legacy_scalar_consumer_attributes_are_gone() -> None:
    """The replaced scalar pair must no longer exist on Feature instances."""
    feature = Feature("subject_token")
    assert not hasattr(feature, "consumer_feature_group_name")
    assert not hasattr(feature, "consumer_property_keys")
