"""Definition-time diagnostic for a group that declares no in_features source contract (m951).

A FeatureChainParserMixin group with a dict PROPERTY_MAPPING lacking an in_features key, a MIN_IN_FEATURES >= 1,
and both the inherited input_features and the inherited matcher counts an absent in_features as zero sources.
The warning names the class and the two fixes: MIN_IN_FEATURES = 0 or an in_features PropertySpec key.
Every fixture carries an "m951" marker in its class name, keys, and values (fixtures leak into the global registry).
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser_mixin import FeatureChainParserMixin
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.provider import DefaultOptionKeys, PropertySpec

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"
FIX_WORDING = "MIN_IN_FEATURES = 0"


@pytest.fixture(autouse=True, scope="module")
def _collect_leaked_local_feature_groups() -> Iterator[None]:
    """Function-local FeatureGroup subclasses are cyclic garbage; collect them before the next module runs."""
    yield
    gc.collect()


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
    """The guard warns when an absent in_features silently counts as zero sources."""

    def test_no_pattern_group_warns_with_fix_wording(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NoPatternM951a(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"req_m951a": PropertySpec("required", allowed_values=("v_m951a",))}

            assert "req_m951a" in _NoPatternM951a.PROPERTY_MAPPING

        warnings = _missing_in_features_warnings(caplog, "_NoPatternM951a")
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "in_features" in message
        assert "in_features key" in message
        assert "_NoPatternM951a" in message

    def test_prefix_pattern_group_also_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """No pattern exemption: a PREFIX_PATTERN group without an in_features key warns too."""
        with caplog.at_level(logging.WARNING):

            class _PrefixPatternM951b(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_m951b$"
                PROPERTY_MAPPING = {"req_m951b": PropertySpec("required", allowed_values=("v_m951b",))}

            assert "req_m951b" in _PrefixPatternM951b.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_PrefixPatternM951b")) == 1

    def test_warns_once_per_class_definition(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _OnceM951c(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"req_m951c": PropertySpec("required", allowed_values=("v_m951c",))}

            assert "req_m951c" in _OnceM951c.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_OnceM951c")) == 1

    def test_violating_subclass_of_violating_base_warns_for_its_own_definition(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):

            class _BaseM951d(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"req_m951d": PropertySpec("required", allowed_values=("v_m951d",))}

            class _ChildM951d(_BaseM951d):
                pass

            assert issubclass(_ChildM951d, _BaseM951d)

        assert len(_missing_in_features_warnings(caplog, "_BaseM951d")) == 1
        assert len(_missing_in_features_warnings(caplog, "_ChildM951d")) == 1

    def test_required_when_guard_wrapper_is_unwrapped_and_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A mixin matcher wrapped by the required_when guard is still the mixin's own matcher."""
        with caplog.at_level(logging.WARNING):

            class _RequiredWhenM951e(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "cond_m951e": PropertySpec("conditional", default=None, required_when=(lambda options: False)),
                }

            assert "cond_m951e" in _RequiredWhenM951e.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_RequiredWhenM951e")) == 1

    def test_name_path_guard_wrapper_is_unwrapped_and_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A named-capture pattern installs the name-path presence guard; the guard must be unwrapped."""
        with caplog.at_level(logging.WARNING):

            class _NamePathM951f(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__(?P<mode_m951f>\w+)$"
                PROPERTY_MAPPING = {
                    "mode_m951f": PropertySpec("mode", allowed_values=("special_m951f",), strict_validation=True),
                }

            assert "mode_m951f" in _NamePathM951f.PROPERTY_MAPPING

        assert len(_missing_in_features_warnings(caplog, "_NamePathM951f")) == 1


class TestMissingInFeaturesDoesNotWarn:
    """The guard stays quiet when the source contract is declared or the class is out of scope."""

    def test_min_in_features_zero_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _MinZeroM951g(FeatureChainParserMixin, FeatureGroup):
                MIN_IN_FEATURES = 0
                PROPERTY_MAPPING = {"req_m951g": PropertySpec("required", allowed_values=("v_m951g",))}

            assert _MinZeroM951g.MIN_IN_FEATURES == 0

        assert not _missing_in_features_warnings(caplog, "_MinZeroM951g")

    def test_in_features_key_without_default_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _KeyNoDefaultM951h(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features.value: PropertySpec("sources"),
                    "req_m951h": PropertySpec("required", allowed_values=("v_m951h",)),
                }

            assert DefaultOptionKeys.in_features.value in _KeyNoDefaultM951h.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_KeyNoDefaultM951h")

    def test_in_features_key_with_default_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _KeyDefaultM951i(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features.value: PropertySpec("sources", default="src_m951i"),
                    "req_m951i": PropertySpec("required", allowed_values=("v_m951i",)),
                }

            assert DefaultOptionKeys.in_features.value in _KeyDefaultM951i.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_KeyDefaultM951i")

    def test_input_features_override_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _InputFeaturesM951j(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"req_m951j": PropertySpec("required", allowed_values=("v_m951j",))}

                def input_features(self, options: Options, feature_name: FeatureName) -> Any:
                    return None

            assert "req_m951j" in _InputFeaturesM951j.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_InputFeaturesM951j")

    def test_custom_matcher_override_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _CustomMatcherM951k(FeatureChainParserMixin, FeatureGroup):
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

            class _NoMappingM951l(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_m951l$"

            class _NoneMappingM951l(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__([\w]+)_m951l2$"
                PROPERTY_MAPPING = None

            assert issubclass(_NoMappingM951l, FeatureChainParserMixin)
            assert _NoneMappingM951l.PROPERTY_MAPPING is None

        assert not _missing_in_features_warnings(caplog, "_NoMappingM951l")
        assert not _missing_in_features_warnings(caplog, "_NoneMappingM951l")

    def test_non_int_min_in_features_neither_raises_nor_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NonIntMinM951m(FeatureChainParserMixin, FeatureGroup):
                MIN_IN_FEATURES = object()  # type: ignore[assignment]
                PROPERTY_MAPPING = {"req_m951m": PropertySpec("required", allowed_values=("v_m951m",))}

            assert "req_m951m" in _NonIntMinM951m.PROPERTY_MAPPING

        assert not _missing_in_features_warnings(caplog, "_NonIntMinM951m")

    def test_class_without_min_max_attrs_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Mirrors the runtime hasattr gate in _validate_in_features."""
        with caplog.at_level(logging.WARNING):

            class _NoMinMaxM951n(FeatureChainParserMixin, FeatureGroup):
                MIN_IN_FEATURES = _HiddenAttribute()  # type: ignore[assignment]
                MAX_IN_FEATURES = _HiddenAttribute()
                PROPERTY_MAPPING = {"req_m951n": PropertySpec("required", allowed_values=("v_m951n",))}

            assert not hasattr(_NoMinMaxM951n, "MIN_IN_FEATURES")

        assert not _missing_in_features_warnings(caplog, "_NoMinMaxM951n")
