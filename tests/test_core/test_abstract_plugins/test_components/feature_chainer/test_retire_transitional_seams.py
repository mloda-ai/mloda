"""Cycle A of retiring the transitional PROPERTY_MAPPING parser seams (os-005, was mloda#798).

Two production changes are pinned here:

1. ``option_key_is_present`` is the ONE module-level presence helper implementing the #768
   behavior matrix, centralizing the existing inline copies:

   * ``allow_explicit_none=True``: present iff the key is in the options, so an explicit
     None value counts as PRESENT.
   * flagless (default): present iff ``options.get(key)`` is not None, so a present-as-None
     value counts as ABSENT.

2. The trivial private wrappers and aliases are DELETED:

   * ``FeatureChainParser._is_context_parameter`` (read ``spec.context``)
   * ``FeatureChainParser._is_strict_validation`` (read ``spec.strict_validation``)
   * ``FeatureChainParser._get_element_validator`` (read ``spec.element_validator``)
   * ``FeatureChainParser._extract_property_values`` (call ``extract_property_values``)
   * ``FeatureChainParserMixin._build_effective_options`` (call
     ``FeatureChainParser.build_effective_options``)

   The public canonical names survive: ``FeatureChainParser.extract_property_values`` and
   ``FeatureChainParser.build_effective_options``.
"""

from __future__ import annotations

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    option_key_is_present,
)
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.options import Options

_FLAGLESS_SPEC = PropertySpec("a key with the default presence rule")
_EXPLICIT_NONE_SPEC = PropertySpec("a key opted into explicit-None presence", allow_explicit_none=True)


class TestOptionKeyIsPresentFlagless:
    """Flagless spec: presence means a non-None value; a present-as-None value counts as absent."""

    def test_absent_key_is_absent(self) -> None:
        """A key in neither group nor context is absent."""
        assert option_key_is_present(_FLAGLESS_SPEC, "os005_key", Options()) is False

    def test_present_as_none_in_group_is_absent(self) -> None:
        """An explicit None value counts as absent without the opt-in flag."""
        options = Options(group={"os005_key": None})
        assert option_key_is_present(_FLAGLESS_SPEC, "os005_key", options) is False

    def test_real_value_is_present(self) -> None:
        """A real value is present."""
        options = Options(group={"os005_key": "value"})
        assert option_key_is_present(_FLAGLESS_SPEC, "os005_key", options) is True


class TestOptionKeyIsPresentAllowExplicitNone:
    """``allow_explicit_none=True``: presence means the key is in the options, None value or not (#768)."""

    def test_present_as_none_is_present(self) -> None:
        """An explicit None value counts as present with the opt-in flag."""
        options = Options(group={"os005_key": None})
        assert option_key_is_present(_EXPLICIT_NONE_SPEC, "os005_key", options) is True

    def test_absent_key_is_absent(self) -> None:
        """The flag never fabricates presence for a key that is not in the options at all."""
        assert option_key_is_present(_EXPLICIT_NONE_SPEC, "os005_key", Options()) is False

    def test_real_value_is_present(self) -> None:
        """A real value is present, exactly as without the flag."""
        options = Options(group={"os005_key": "value"})
        assert option_key_is_present(_EXPLICIT_NONE_SPEC, "os005_key", options) is True


class TestOptionKeyIsPresentContextStorage:
    """Presence reads through ``Options`` as a whole, so a context-stored key behaves like a group-stored one."""

    def test_real_value_in_context_is_present_for_flagless_spec(self) -> None:
        """A real value stored in context is present."""
        options = Options(context={"os005_key": "value"})
        assert option_key_is_present(_FLAGLESS_SPEC, "os005_key", options) is True

    def test_none_in_context_is_absent_for_flagless_spec(self) -> None:
        """A present-as-None context value counts as absent without the opt-in flag."""
        options = Options(context={"os005_key": None})
        assert option_key_is_present(_FLAGLESS_SPEC, "os005_key", options) is False

    def test_none_in_context_is_present_for_explicit_none_spec(self) -> None:
        """A present-as-None context value counts as present with the opt-in flag."""
        options = Options(context={"os005_key": None})
        assert option_key_is_present(_EXPLICIT_NONE_SPEC, "os005_key", options) is True


class TestTransitionalSeamsAreGone:
    """The trivial private wrappers and aliases are deleted; callers use the spec fields or the public API."""

    def test_is_context_parameter_is_gone(self) -> None:
        """``_is_context_parameter`` is gone: read ``spec.context``."""
        assert not hasattr(FeatureChainParser, "_is_context_parameter")

    def test_is_strict_validation_is_gone(self) -> None:
        """``_is_strict_validation`` is gone: read ``spec.strict_validation``."""
        assert not hasattr(FeatureChainParser, "_is_strict_validation")

    def test_get_element_validator_is_gone(self) -> None:
        """``_get_element_validator`` is gone: read ``spec.element_validator``."""
        assert not hasattr(FeatureChainParser, "_get_element_validator")

    def test_private_extract_property_values_alias_is_gone(self) -> None:
        """``_extract_property_values`` is gone: call the public ``extract_property_values``."""
        assert not hasattr(FeatureChainParser, "_extract_property_values")

    def test_mixin_build_effective_options_passthrough_is_gone(self) -> None:
        """The mixin pass-through is gone: call ``FeatureChainParser.build_effective_options``."""
        assert not hasattr(FeatureChainParserMixin, "_build_effective_options")


class TestPublicSurvivorsStay:
    """The public canonical names the wrappers delegated to keep existing."""

    def test_extract_property_values_survives(self) -> None:
        """``FeatureChainParser.extract_property_values`` stays and keeps its semantics."""
        assert hasattr(FeatureChainParser, "extract_property_values")
        assert FeatureChainParser.extract_property_values(PropertySpec("no declared value space")) == {}

        spec = PropertySpec("declared value space", allowed_values={"add": "Addition"})
        assert FeatureChainParser.extract_property_values(spec) == {"add": "Addition"}

    def test_build_effective_options_survives(self) -> None:
        """``FeatureChainParser.build_effective_options`` stays; nothing to bind returns options by identity."""
        assert hasattr(FeatureChainParser, "build_effective_options")

        options = Options(group={"os005_key": "value"})
        assert FeatureChainParser.build_effective_options("plain_name", [], {}, options) is options
