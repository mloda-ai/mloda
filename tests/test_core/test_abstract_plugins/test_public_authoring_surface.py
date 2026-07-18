"""The typed authoring surface must be complete on ``mloda.provider`` (issue #776).

Raw dict specs are rejected, so everything an author needs to declare and inspect a
PROPERTY_MAPPING has to be reachable from the public module. ``NO_DEFAULT`` was exported
without ``is_no_default``, which pushed callers toward the identity check the sentinel's
own docstring warns against.
"""

import copy

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import _NoDefault
from mloda.provider import NO_DEFAULT, PropertySpec, is_no_default, property_spec


class TestAuthoringSurfaceIsExported:
    @pytest.mark.parametrize("name", ["PropertySpec", "property_spec", "NO_DEFAULT", "is_no_default"])
    def test_name_is_public(self, name: str) -> None:
        assert hasattr(provider, name)
        assert name in provider.__all__, f"{name} should be listed in mloda.provider.__all__"


class TestIsNoDefault:
    def test_detects_an_undeclared_default(self) -> None:
        spec = property_spec("no default declared")
        assert is_no_default(spec.default) is True

    def test_a_declared_none_is_a_default(self) -> None:
        """``default=None`` declares None; it is not the absence of a declaration."""
        spec = property_spec("declares None", default=None)
        assert is_no_default(spec.default) is False

    @pytest.mark.parametrize("value", [None, 0, False, "", "add", [], NO_DEFAULT.__class__])
    def test_ordinary_values_are_not_the_sentinel(self, value: object) -> None:
        assert is_no_default(value) is False

    def test_sentinel_itself_is_detected(self) -> None:
        assert is_no_default(NO_DEFAULT) is True

    def test_detects_a_distinct_sentinel_instance(self) -> None:
        """Why this is a type test and not ``spec.default is NO_DEFAULT``.

        A second sentinel instance (a duplicated module copy constructing its own, or any
        caller doing ``_NoDefault()``) is still "declares no default". An identity check
        would read it as a *declared* default and silently make a required key optional.
        """
        other_sentinel = _NoDefault()

        assert other_sentinel is not NO_DEFAULT
        assert is_no_default(other_sentinel) is True

    def test_survives_copying(self) -> None:
        """deepcopy resolves back through __reduce__, and the check holds either way."""
        assert is_no_default(copy.deepcopy(NO_DEFAULT)) is True
        assert is_no_default(copy.copy(NO_DEFAULT)) is True


class TestBuilderAndTypeAgree:
    def test_builder_returns_the_lower_level_type(self) -> None:
        assert isinstance(property_spec("built"), PropertySpec)

    def test_strict_keyword_maps_to_strict_validation(self) -> None:
        spec = property_spec("strict one", strict=True, allowed_values={"add": "Addition"})
        assert spec.strict_validation is True

    def test_builder_and_constructor_produce_equal_specs(self) -> None:
        built = property_spec("same", strict=True, allowed_values={"add": "Addition"}, default="add")
        constructed = PropertySpec(
            "same",
            strict_validation=True,
            allowed_values={"add": "Addition"},
            default="add",
        )
        assert built == constructed
