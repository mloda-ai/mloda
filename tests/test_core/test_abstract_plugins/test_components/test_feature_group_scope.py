"""Tests for the per-feature FeatureGroup resolution scope (issue #508).

These tests specify a new OPTIONAL, RESOLUTION-ONLY scope on ``Feature``:

    Feature("subject_token", feature_group=SomeFeatureGroup)
    Feature("subject_token", feature_group="SomeFeatureGroup")

The scope is stored on ``feature.feature_group_scope`` and is EXCLUDED from
identity (``__eq__``/``__hash__``/``similarity_hash``), exactly like
``Feature.link`` and ``Feature.index`` already are. It only influences which
feature group a feature resolves to, never how features are grouped/batched.

The resolution behaviour itself is covered in
tests/test_core/test_prepare/test_feature_group_scope_resolution.py.
"""

import pytest

from mloda.provider import FeatureGroup
from mloda.user import Feature


class _ScopeDummyFeatureGroup(FeatureGroup):
    """Concrete FeatureGroup used only as a scope target in these tests."""


class _OtherScopeDummyFeatureGroup(FeatureGroup):
    """A second concrete FeatureGroup, distinct from the first."""


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


def test_scope_defaults_to_none() -> None:
    """Without the new keyword, feature_group_scope is None."""
    feature = Feature("subject_token")
    assert feature.feature_group_scope is None


def test_scope_stores_string() -> None:
    """A string scope is stored verbatim on feature_group_scope."""
    feature = Feature("subject_token", feature_group="SomeSource")
    assert feature.feature_group_scope == "SomeSource"


def test_scope_stores_class_object_as_is() -> None:
    """A FeatureGroup class scope is stored AS-IS (not converted to a string)."""
    feature = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    assert feature.feature_group_scope is _ScopeDummyFeatureGroup


def test_scope_none_is_none() -> None:
    """Passing None explicitly keeps feature_group_scope as None."""
    feature = Feature("subject_token", feature_group=None)
    assert feature.feature_group_scope is None


def test_scope_empty_string_is_none() -> None:
    """An empty string scope normalises to None."""
    feature = Feature("subject_token", feature_group="")
    assert feature.feature_group_scope is None


def test_scope_invalid_int_raises_typeerror() -> None:
    """An int scope raises TypeError mentioning feature_group (not a kwarg error)."""
    with pytest.raises(TypeError) as exc_info:
        Feature("subject_token", feature_group=123)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "feature_group" in message
    # Guard against a false pass while the kwarg does not yet exist: the current
    # "unexpected keyword argument 'feature_group'" TypeError must NOT satisfy this.
    assert "unexpected keyword" not in message


def test_scope_invalid_object_raises_typeerror() -> None:
    """An arbitrary object instance scope raises TypeError mentioning feature_group."""
    with pytest.raises(TypeError) as exc_info:
        Feature("subject_token", feature_group=object())  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "feature_group" in message
    assert "unexpected keyword" not in message


def test_options_dict_does_not_set_scope() -> None:
    """There is NO options fallback: options={'feature_group': ...} must not set the scope."""
    feature = Feature("subject_token", options={"feature_group": "SomeSource"})
    assert feature.feature_group_scope is None


def test_constructor_scope_does_not_pollute_options() -> None:
    """The constructor scope value must not leak into options.group or options.context."""
    feature = Feature("subject_token", feature_group="SomeSource")
    assert "feature_group" not in feature.options.group
    assert "feature_group" not in feature.options.context


# ---------------------------------------------------------------------------
# Identity invariants: scope is excluded from equality / hash / similarity
# ---------------------------------------------------------------------------


def test_scoped_equals_unscoped() -> None:
    """A scoped feature equals an otherwise-identical unscoped feature."""
    scoped = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    unscoped = Feature("subject_token")
    assert scoped == unscoped


def test_scoped_hash_equals_unscoped() -> None:
    """A scoped feature hashes equal to an otherwise-identical unscoped feature."""
    scoped = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    unscoped = Feature("subject_token")
    assert hash(scoped) == hash(unscoped)


def test_different_scopes_are_equal() -> None:
    """Two features scoped to different feature groups are still equal and hash equal."""
    scoped_a = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    scoped_b = Feature("subject_token", feature_group=_OtherScopeDummyFeatureGroup)
    assert scoped_a == scoped_b
    assert hash(scoped_a) == hash(scoped_b)


def test_similarity_hash_identical_scoped_vs_unscoped() -> None:
    """similarity_hash must be identical so read-batching is unaffected by scope."""
    scoped = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    unscoped = Feature("subject_token")
    assert scoped.similarity_hash() == unscoped.similarity_hash()


def test_scoped_and_unscoped_collapse_in_set() -> None:
    """A scoped and an otherwise-identical unscoped feature collapse to ONE set element."""
    collection = {
        Feature("subject_token", feature_group=_ScopeDummyFeatureGroup),
        Feature("subject_token"),
    }
    assert len(collection) == 1
