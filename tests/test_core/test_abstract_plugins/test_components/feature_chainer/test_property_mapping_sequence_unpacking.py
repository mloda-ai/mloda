"""Uniform sequence unpacking for PROPERTY_MAPPING validation (issue #600).

The SPEC declares the arity, not the caller's Python syntax. What a validator receives
must not depend on whether the user typed a list, a tuple, a set or a frozenset.

* ``element_validator`` and the membership check see ELEMENTS. Every sequence container
  (list, tuple, set, frozenset) unpacks element-wise and identically.
* Elements arrive as their real values with their real types. The old
  ``str(tuple)`` hop (a hashability workaround, for values that were already hashable)
  is gone: an int element arrives as ``1``, never as ``"1"`` or ``"(1, 2)"``.
* A scalar is exactly one element. A ``str`` is a SCALAR, not a sequence of characters.
* A ``dict`` is a COMPOSITE value, not a sequence to iterate over its keys.
* ``match_guard`` is unaffected: it still receives the raw, whole, un-unpacked value with
  its original container type. It is the escape hatch for genuinely composite values
  (shape, length, ordering, cross-element constraints).
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys


class _Recorder:
    """Callable that records every value it was handed and always accepts."""

    def __init__(self, verdict: bool = True) -> None:
        self.calls: list[Any] = []
        self._verdict = verdict

    def __call__(self, value: Any) -> bool:
        self.calls.append(value)
        return self._verdict


def _strict_element_validator_group(validator: Any) -> type[FeatureChainParserMixin]:
    """A feature group whose single key is strictly validated element-wise."""

    class ElementFeatureGroup(FeatureChainParserMixin):
        PROPERTY_MAPPING = {
            "ops": {
                "explanation": "Operations to apply",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
                DefaultOptionKeys.element_validator: validator,
            }
        }

    return ElementFeatureGroup


def _strict_membership_group() -> type[FeatureChainParserMixin]:
    """A feature group whose single key is strictly validated against an accepted set."""

    class MembershipFeatureGroup(FeatureChainParserMixin):
        PROPERTY_MAPPING = {
            "ops": {
                "explanation": "Operations to apply",
                DefaultOptionKeys.allowed_values: {"a": "A", "b": "B"},
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
            }
        }

    return MembershipFeatureGroup


def _match_guard_group(guard: Any) -> type[FeatureChainParserMixin]:
    """A feature group whose single key is guarded on its raw, whole value."""

    class GuardedFeatureGroup(FeatureChainParserMixin):
        PROPERTY_MAPPING = {
            "ops": {
                "explanation": "Operations to apply",
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: False,
                DefaultOptionKeys.match_guard: guard,
            }
        }

    return GuardedFeatureGroup


CONTAINERS: list[tuple[str, Any]] = [
    ("list", ["a", "b"]),
    ("tuple", ("a", "b")),
    ("set", {"a", "b"}),
    ("frozenset", frozenset({"a", "b"})),
]

INT_CONTAINERS: list[tuple[str, Any]] = [
    ("list", [1, 2]),
    ("tuple", (1, 2)),
    ("set", {1, 2}),
    ("frozenset", frozenset({1, 2})),
]

EMPTY_CONTAINERS: list[tuple[str, Any]] = [
    ("list", []),
    ("tuple", ()),
    ("set", set()),
    ("frozenset", frozenset()),
]


class TestUniformSequenceUnpacking:
    """Every sequence container unpacks element-wise and identically."""

    @pytest.mark.parametrize(("label", "value"), CONTAINERS, ids=[label for label, _ in CONTAINERS])
    def test_element_validator_sees_the_same_elements_for_every_container(self, label: str, value: Any) -> None:
        """list, tuple, set and frozenset all yield exactly the elements "a" and "b"."""
        validator = _Recorder()
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True
        assert set(validator.calls) == {"a", "b"}, f"{label} must unpack into its elements"
        assert len(validator.calls) == 2

    @pytest.mark.parametrize(("label", "value"), CONTAINERS, ids=[label for label, _ in CONTAINERS])
    def test_membership_check_unpacks_every_container(self, label: str, value: Any) -> None:
        """The membership check also runs per element, for every container type."""
        group = _strict_membership_group()

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("list", ["a", "nope"]),
            ("tuple", ("a", "nope")),
            ("set", {"a", "nope"}),
            ("frozenset", frozenset({"a", "nope"})),
        ],
        ids=["list", "tuple", "set", "frozenset"],
    )
    def test_membership_rejects_a_bad_element_in_every_container(self, label: str, value: Any) -> None:
        """One element outside the accepted set rejects the whole option, whatever the container."""
        group = _strict_membership_group()

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is False

    @pytest.mark.parametrize(("label", "value"), INT_CONTAINERS, ids=[label for label, _ in INT_CONTAINERS])
    def test_elements_keep_their_real_type_no_stringification(self, label: str, value: Any) -> None:
        """The ``str(tuple)`` workaround is gone: an int element arrives as ``1``, not ``"1"``.

        This is the regression that would silently come back: a tuple used to reach the
        validator as the single string ``"(1, 2)"``.
        """
        validator = _Recorder()
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True
        assert set(validator.calls) == {1, 2}
        assert all(isinstance(call, int) for call in validator.calls), f"{label} elements must stay ints"
        assert all(not isinstance(call, str) for call in validator.calls), "no stringification, ever"

    def test_single_element_tuple_is_not_stringified(self) -> None:
        """A one-element tuple yields that element, not ``"('a',)"``."""
        validator = _Recorder()
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": ("a",)})) is True
        assert validator.calls == ["a"]


class TestScalarsAreSingleElements:
    """A scalar value is exactly one element, and a string is a scalar."""

    def test_string_does_not_unpack_into_characters(self) -> None:
        """``"abc"`` validates as the single element ``"abc"``, never as 'a', 'b', 'c'."""
        validator = _Recorder()
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": "abc"})) is True
        assert validator.calls == ["abc"]

    def test_scalar_int_is_one_element(self) -> None:
        """A bare int is one element and keeps its type."""
        validator = _Recorder()
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": 7})) is True
        assert validator.calls == [7]

    def test_string_membership_matches_the_whole_string(self) -> None:
        """A scalar string is checked against the accepted set as a whole."""
        group = _strict_membership_group()

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": "a"})) is True
        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": "ab"})) is False


class TestEmptySequences:
    """A present-but-empty sequence still satisfies the required-presence check."""

    @pytest.mark.parametrize(("label", "value"), EMPTY_CONTAINERS, ids=[label for label, _ in EMPTY_CONTAINERS])
    def test_empty_container_is_present_and_vacuously_valid(self, label: str, value: Any) -> None:
        """An empty container has no elements: the validator never runs, and the key counts as present."""
        validator = _Recorder(verdict=False)
        group = _strict_element_validator_group(validator)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True
        assert validator.calls == [], f"an empty {label} has no elements to validate"

    @pytest.mark.parametrize(("label", "value"), EMPTY_CONTAINERS, ids=[label for label, _ in EMPTY_CONTAINERS])
    def test_empty_container_passes_membership_vacuously(self, label: str, value: Any) -> None:
        """No element means nothing to reject, so a strict membership key still matches."""
        group = _strict_membership_group()

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True


class TestDictIsACompositeValue:
    """A dict option is ONE composite value, not a sequence unpacked over its keys.

    Unpacking a dict over its keys would silently drop the values, and a dict has no
    element-wise meaning: shape checks on a mapping are exactly what ``match_guard``
    exists for. So the element_validator sees the dict itself, and nothing raises.
    """

    def test_element_validator_receives_the_whole_dict(self) -> None:
        """The dict arrives as one element, with its keys and values intact."""
        validator = _Recorder()
        group = _strict_element_validator_group(validator)
        value = {"a": 1, "b": 2}

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True
        assert validator.calls == [value]
        assert validator.calls[0] is value, "the dict must not be copied, unpacked or stringified"

    def test_dict_is_a_non_match_under_a_membership_key_without_raising(self) -> None:
        """A dict cannot be a member of an accepted set, so it is a clean non-match, not a crash."""
        group = _strict_membership_group()

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": {"a": 1}})) is False


class TestMatchGuardSeesTheRawValue:
    """``match_guard`` is unaffected by unpacking: raw, whole value, original container type."""

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("list", ["a", "b"]),
            ("tuple", ("a", "b")),
            ("set", {"a", "b"}),
            ("frozenset", frozenset({"a", "b"})),
            ("dict", {"a": 1}),
            ("str", "abc"),
        ],
        ids=["list", "tuple", "set", "frozenset", "dict", "str"],
    )
    def test_guard_receives_the_original_container(self, label: str, value: Any) -> None:
        """The guard sees the very object the user passed, with its type preserved."""
        guard = _Recorder()
        group = _match_guard_group(guard)

        assert group.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is True
        assert guard.calls == [value]
        assert type(guard.calls[0]) is type(value), f"{label} must reach the match_guard un-unpacked"

    def test_guard_can_still_reject_on_shape_while_elements_pass(self) -> None:
        """The two live side by side: the validator judges elements, the guard judges the whole."""
        validator = _Recorder()
        guard = _Recorder(verdict=False)

        class BothFeatureGroup(FeatureChainParserMixin):
            PROPERTY_MAPPING = {
                "ops": {
                    "explanation": "Operations to apply",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                    DefaultOptionKeys.element_validator: validator,
                    DefaultOptionKeys.match_guard: guard,
                }
            }

        value = ("a", "b")

        assert BothFeatureGroup.match_feature_group_criteria("any_feature", Options(context={"ops": value})) is False
        assert set(validator.calls) == {"a", "b"}, "the element_validator sees elements"
        assert guard.calls == [value], "the match_guard sees the raw tuple"
        assert type(guard.calls[0]) is tuple
