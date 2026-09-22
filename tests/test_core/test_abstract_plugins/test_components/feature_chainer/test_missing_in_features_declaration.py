"""Definition-time warning for a mixin group that declares no in_features key.

Fixtures carry an "m951" marker in class names, keys, and values because they leak into the global registry.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer import feature_chain_author_guards
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys, PropertySpec

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"
FIX_WORDING = "MIN_IN_FEATURES = 0"


class _HiddenAttribute:
    """Descriptor that makes hasattr return False by raising AttributeError."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: object, objtype: type | None = None) -> None:
        raise AttributeError(self.name)


def _missing_in_features_warnings(caplog: pytest.LogCaptureFixture, class_name: str) -> list[logging.LogRecord]:
    """The missing-in_features definition-time warnings for one class name."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == AUTHOR_GUARDS_LOGGER
        and FIX_WORDING in record.getMessage()
        and class_name in record.getMessage()
    ]


class TestMissingInFeaturesWarns:
    """The guard warns when an absent in_features counts as zero sources."""

    def test_no_pattern_group_warns_with_fix_wording(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NoPatternM951a(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951a": PropertySpec("required", allowed_values=("v_m951a",))}

            assert "req_m951a" in _NoPatternM951a.PROPERTY_MAPPING

        warnings = _missing_in_features_warnings(caplog, "_NoPatternM951a")
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "in_features" in message
        assert "in_features key" in message
        assert "_NoPatternM951a" in message

    def test_empty_property_mapping_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _EmptyMappingM951o(FeatureChainParserMixin):
                PROPERTY_MAPPING: dict[str, Any] = {}

            assert _EmptyMappingM951o.PROPERTY_MAPPING == {}

        assert len(_missing_in_features_warnings(caplog, "_EmptyMappingM951o")) == 1

    def test_guard_takes_owner_and_mixin(self) -> None:
        signature = inspect.signature(feature_chain_author_guards.warn_missing_in_features_declaration)
        assert list(signature.parameters) == ["owner", "mixin"]

    def test_warned_class_carries_own_flag_constant(self, caplog: pytest.LogCaptureFixture) -> None:
        flag = getattr(feature_chain_author_guards, "MISSING_IN_FEATURES_DIAGNOSTIC_FLAG")

        with caplog.at_level(logging.WARNING):

            class _FlaggedM951p(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951p": PropertySpec("required", allowed_values=("v_m951p",))}

            assert "req_m951p" in _FlaggedM951p.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_FlaggedM951p")) == 1
        assert flag in vars(_FlaggedM951p)

    def test_warns_once_per_class_definition(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _OnceM951c(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951c": PropertySpec("required", allowed_values=("v_m951c",))}

            assert "req_m951c" in _OnceM951c.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_OnceM951c")) == 1

    def test_base_and_inheriting_subclasses_warn_once_in_total(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _BaseM951d(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951d": PropertySpec("required", allowed_values=("v_m951d",))}

            class _ChildAM951d(_BaseM951d):
                pass

            class _ChildBM951d(_ChildAM951d):
                pass

            assert issubclass(_ChildBM951d, _BaseM951d)

        warnings = _missing_in_features_warnings(caplog, "M951d")
        assert len(warnings) == 1
        assert "_BaseM951d" in warnings[0].getMessage()

    def test_subclass_that_fixes_the_violation_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _BaseM951q(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951q": PropertySpec("required", allowed_values=("v_m951q",))}

            class _FixedM951q(_BaseM951q):
                MIN_IN_FEATURES = 0

            assert _FixedM951q.MIN_IN_FEATURES == 0

        warnings = _missing_in_features_warnings(caplog, "M951q")
        assert len(warnings) == 1
        assert "_BaseM951q" in warnings[0].getMessage()

    def test_violating_subclass_of_clean_base_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _CleanBaseM951r(FeatureChainParserMixin):
                MIN_IN_FEATURES = 0
                PROPERTY_MAPPING = {"req_m951r": PropertySpec("required", allowed_values=("v_m951r",))}

            class _ViolatingM951r(_CleanBaseM951r):
                MIN_IN_FEATURES = 1

            assert _ViolatingM951r.MIN_IN_FEATURES == 1

        warnings = _missing_in_features_warnings(caplog, "M951r")
        assert len(warnings) == 1
        assert "_ViolatingM951r" in warnings[0].getMessage()

    def test_required_when_guard_wrapper_is_unwrapped_and_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _RequiredWhenM951e(FeatureChainParserMixin):
                PROPERTY_MAPPING = {
                    "cond_m951e": PropertySpec("conditional", default=None, required_when=(lambda options: False)),
                }

            assert "cond_m951e" in _RequiredWhenM951e.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_RequiredWhenM951e")) == 1

    # A matcher wrapped by BOTH the required_when and name-path guards is not constructible here: the
    # name-path guard installs only for a group with a pattern, and such a group is exempt (source from the name).


class TestMissingInFeaturesDoesNotWarn:
    """The guard stays quiet when the source contract is declared or the class is out of scope."""

    def test_prefix_pattern_group_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _PrefixPatternM951b(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__([\w]+)_m951b$"
                PROPERTY_MAPPING = {"req_m951b": PropertySpec("required", allowed_values=("v_m951b",))}

            assert "req_m951b" in _PrefixPatternM951b.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_PrefixPatternM951b")

    def test_suffix_pattern_group_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _SuffixPatternM951s(FeatureChainParserMixin):
                SUFFIX_PATTERN = r"^m951s_([\w]+)__.*$"
                PROPERTY_MAPPING = {"req_m951s": PropertySpec("required", allowed_values=("v_m951s",))}

            assert "req_m951s" in _SuffixPatternM951s.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_SuffixPatternM951s")

    def test_name_path_guard_wrapped_pattern_group_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Formerly warned; a name-path guard installs only with a pattern, which now exempts the group."""
        with caplog.at_level(logging.WARNING):

            class _NamePathM951f(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__(?P<mode_m951f>\w+)$"
                PROPERTY_MAPPING = {
                    "mode_m951f": PropertySpec("mode", allowed_values=("special_m951f",), strict_validation=True),
                }

            assert "mode_m951f" in _NamePathM951f.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_NamePathM951f")

    def test_in_features_enum_member_key_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _EnumKeyM951t(FeatureChainParserMixin):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features: PropertySpec("sources"),
                    "req_m951t": PropertySpec("required", allowed_values=("v_m951t",)),
                }

            assert "req_m951t" in _EnumKeyM951t.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_EnumKeyM951t")

    def test_hidden_matcher_defines_without_raising_or_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _HiddenMatcherM951u(FeatureChainParserMixin):
                match_feature_group_criteria = _HiddenAttribute()  # type: ignore[assignment]
                PROPERTY_MAPPING = {"req_m951u": PropertySpec("required", allowed_values=("v_m951u",))}

            assert "req_m951u" in _HiddenMatcherM951u.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_HiddenMatcherM951u")

    def test_circular_wrapped_matcher_defines_without_raising(self, caplog: pytest.LogCaptureFixture) -> None:
        def _looping_matcher(
            cls: Any, feature_name: str | FeatureName, options: Options, data_access_collection: Any = None
        ) -> bool:
            return False

        _looping_matcher.__wrapped__ = _looping_matcher  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING):

            class _CycleMatcherM951v(FeatureChainParserMixin):
                match_feature_group_criteria = classmethod(_looping_matcher)
                PROPERTY_MAPPING = {"req_m951v": PropertySpec("required", allowed_values=("v_m951v",))}

            assert "req_m951v" in _CycleMatcherM951v.PROPERTY_MAPPING

    def test_min_in_features_zero_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _MinZeroM951g(FeatureChainParserMixin):
                MIN_IN_FEATURES = 0
                PROPERTY_MAPPING = {"req_m951g": PropertySpec("required", allowed_values=("v_m951g",))}

            assert _MinZeroM951g.MIN_IN_FEATURES == 0

        assert not _missing_in_features_warnings(caplog, "_MinZeroM951g")

    def test_in_features_key_without_default_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _KeyNoDefaultM951h(FeatureChainParserMixin):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features.value: PropertySpec("sources"),
                    "req_m951h": PropertySpec("required", allowed_values=("v_m951h",)),
                }

            assert DefaultOptionKeys.in_features.value in _KeyNoDefaultM951h.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_KeyNoDefaultM951h")

    def test_in_features_key_with_default_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _KeyDefaultM951i(FeatureChainParserMixin):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features.value: PropertySpec("sources", default="src_m951i"),
                    "req_m951i": PropertySpec("required", allowed_values=("v_m951i",)),
                }

            assert DefaultOptionKeys.in_features.value in _KeyDefaultM951i.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_KeyDefaultM951i")

    def test_input_features_override_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _InputFeaturesM951j(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951j": PropertySpec("required", allowed_values=("v_m951j",))}

                def input_features(self, options: Options, feature_name: FeatureName) -> Any:
                    return None

            assert "req_m951j" in _InputFeaturesM951j.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_InputFeaturesM951j")

    def test_custom_matcher_override_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _CustomMatcherM951k(FeatureChainParserMixin):
                PROPERTY_MAPPING = {"req_m951k": PropertySpec("required", allowed_values=("v_m951k",))}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return str(feature_name) == "specific_m951k"

            assert "req_m951k" in _CustomMatcherM951k.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_CustomMatcherM951k")

    def test_no_property_mapping_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NoMappingM951l(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__([\w]+)_m951l$"

            class _NoneMappingM951l(FeatureChainParserMixin):
                PREFIX_PATTERN = r".*__([\w]+)_m951l2$"
                PROPERTY_MAPPING = None

            assert issubclass(_NoMappingM951l, FeatureChainParserMixin)
            assert _NoneMappingM951l.PROPERTY_MAPPING is None

        assert not _missing_in_features_warnings(caplog, "_NoMappingM951l")
        assert not _missing_in_features_warnings(caplog, "_NoneMappingM951l")

    def test_non_int_min_in_features_neither_raises_nor_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NonIntMinM951m(FeatureChainParserMixin):
                MIN_IN_FEATURES = object()  # type: ignore[assignment]
                PROPERTY_MAPPING = {"req_m951m": PropertySpec("required", allowed_values=("v_m951m",))}

            assert "req_m951m" in _NonIntMinM951m.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_NonIntMinM951m")

    def test_class_without_min_max_attrs_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NoMinMaxM951n(FeatureChainParserMixin):
                MIN_IN_FEATURES = _HiddenAttribute()  # type: ignore[assignment]
                MAX_IN_FEATURES = _HiddenAttribute()
                PROPERTY_MAPPING = {"req_m951n": PropertySpec("required", allowed_values=("v_m951n",))}

            assert not hasattr(_NoMinMaxM951n, "MIN_IN_FEATURES")

        assert not _missing_in_features_warnings(caplog, "_NoMinMaxM951n")
