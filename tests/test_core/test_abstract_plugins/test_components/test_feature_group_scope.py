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

import copy

import pytest

from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Features
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class _ScopeDummyFeatureGroup(FeatureGroup):
    """Concrete FeatureGroup used only as a scope target in these tests."""


class _OtherScopeDummyFeatureGroup(FeatureGroup):
    """A second concrete FeatureGroup, distinct from the first."""


class _NotAFeatureGroup:
    """A non-FeatureGroup class that nonetheless exposes get_class_name().

    ComputeFramework and several other mloda base classes also expose a
    ``get_class_name`` classmethod, so a mere ``hasattr(x, "get_class_name")``
    guard is not sufficient to prove ``x`` is a FeatureGroup subclass.
    """

    @classmethod
    def get_class_name(cls) -> str:
        return "X"


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


# ---------------------------------------------------------------------------
# Constructor guard: only FeatureGroup subclasses (or names) are valid scopes
# ---------------------------------------------------------------------------


def test_scope_compute_framework_class_raises_typeerror() -> None:
    """A ComputeFramework subclass must be rejected, not silently accepted.

    ComputeFramework exposes get_class_name(), so a hasattr-based guard wrongly
    treats it as a valid FeatureGroup scope. The scope must reject any class that
    is not a FeatureGroup subclass, with a TypeError naming feature_group.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", feature_group=PandasDataFrame)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "feature_group" in message
    assert "unexpected keyword" not in message


def test_scope_non_feature_group_class_raises_typeerror() -> None:
    """An unrelated class exposing get_class_name() must be rejected.

    Guards against the hasattr(x, "get_class_name") shortcut: presence of that
    method does not make a class a FeatureGroup subclass.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", feature_group=_NotAFeatureGroup)  # type: ignore[arg-type]
    message = str(exc_info.value)
    assert "feature_group" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# deepcopy characterization: class-object scope survives the engine copy path
# ---------------------------------------------------------------------------


def test_class_scope_survives_deepcopy() -> None:
    """A class-object scope survives copy.deepcopy and preserves identity.

    Characterization/guard test for the engine's deepcopy path: deepcopying a
    Feature must keep the same FeatureGroup class object as the scope (classes
    deepcopy to themselves) while equality/hash stay independent of the scope.
    """
    f = Feature("subject_token", feature_group=_ScopeDummyFeatureGroup)
    g = copy.deepcopy(f)

    assert g.feature_group_scope is f.feature_group_scope
    assert g.feature_group_scope is _ScopeDummyFeatureGroup
    assert g == f
    assert hash(g) == hash(f)


# ---------------------------------------------------------------------------
# Duplicate guard: scope is excluded from identity, so same-name features with
# different scopes collide inside a single Features request (issue #508).
# ---------------------------------------------------------------------------


def test_same_name_two_class_scopes_in_one_request_raise_duplicate() -> None:
    """Two same-name features with different class scopes are equal, so Features rejects them.

    The scope is excluded from identity, so both features compare equal and
    check_duplicate_feature raises "Duplicate feature setup" for the name.
    """
    with pytest.raises(ValueError, match="Duplicate feature setup"):
        Features(
            [
                Feature("subject_token", feature_group=_ScopeDummyFeatureGroup),
                Feature("subject_token", feature_group=_OtherScopeDummyFeatureGroup),
            ]
        )


def test_same_name_two_string_scopes_in_one_request_raise_duplicate() -> None:
    """Two same-name features with different string scopes still collide as duplicates."""
    with pytest.raises(ValueError, match="Duplicate feature setup"):
        Features(
            [
                Feature("subject_token", feature_group="A"),
                Feature("subject_token", feature_group="B"),
            ]
        )


def test_same_name_scoped_and_unscoped_in_one_request_raise_duplicate() -> None:
    """A scoped and an otherwise-identical unscoped feature collide as duplicates."""
    with pytest.raises(ValueError, match="Duplicate feature setup"):
        Features(
            [
                Feature("subject_token", feature_group=_ScopeDummyFeatureGroup),
                Feature("subject_token"),
            ]
        )


# ---------------------------------------------------------------------------
# Root-base rejection: the abstract FeatureGroup root class is not a scope
# ---------------------------------------------------------------------------


def test_scope_root_feature_group_class_raises_typeerror() -> None:
    """Scoping to the abstract root FeatureGroup class must be rejected.

    The root class can never be a concrete resolution target; accepting it
    silently produces a scope that matches everything (with issubclass
    semantics) or nothing (with identity semantics). It must raise a TypeError
    naming feature_group at construction time.
    """
    with pytest.raises(TypeError) as exc_info:
        Feature("x", feature_group=FeatureGroup)
    message = str(exc_info.value)
    assert "feature_group" in message
    assert "unexpected keyword" not in message


# ---------------------------------------------------------------------------
# String normalization: whitespace-only scopes collapse to None
# ---------------------------------------------------------------------------


def test_scope_whitespace_only_string_is_none() -> None:
    """A whitespace-only string scope normalises to None, like the empty string."""
    feature = Feature("x", feature_group="   ")
    assert feature.feature_group_scope is None


def test_scope_padded_string_is_stored_stripped() -> None:
    """A class-name string scope with surrounding whitespace is stored STRIPPED.

    Resolution compares the scope against get_class_name() results, which never
    carry padding, so ' SomeSource ' stored verbatim could never match anything.
    The constructor must normalise it to 'SomeSource'.
    """
    feature = Feature("subject_token", feature_group=" SomeSource ")
    assert feature.feature_group_scope == "SomeSource"


# ---------------------------------------------------------------------------
# Typed helpers forward the scope to the constructor
# ---------------------------------------------------------------------------


def test_int32_of_forwards_class_scope() -> None:
    """Feature.int32_of forwards a class-object scope to feature_group_scope."""
    feature = Feature.int32_of("subject_token", feature_group=_ScopeDummyFeatureGroup)
    assert feature.feature_group_scope is _ScopeDummyFeatureGroup


def test_str_of_forwards_string_scope() -> None:
    """Feature.str_of forwards a string scope to feature_group_scope."""
    feature = Feature.str_of("subject_token", feature_group="SomeSource")
    assert feature.feature_group_scope == "SomeSource"


def test_not_typed_forwards_class_scope() -> None:
    """Feature.not_typed forwards a class-object scope to feature_group_scope."""
    feature = Feature.not_typed("subject_token", feature_group=_ScopeDummyFeatureGroup)
    assert feature.feature_group_scope is _ScopeDummyFeatureGroup
