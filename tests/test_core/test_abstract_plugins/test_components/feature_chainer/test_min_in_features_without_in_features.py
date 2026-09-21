"""Definition-time warning for a class that requires sources but declares no in_features key.

Fixtures carry a "u1519" marker in class names, keys, and values so they cannot collide in the global
plugin registry. The warning is scoped by logger name and the phrase "declares no in_features".
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
from mloda.user import Feature
from mloda_plugins.feature_group.experimental.aggregated_feature_group.base import AggregatedFeatureGroup

AUTHOR_GUARDS_LOGGER = "mloda.core.abstract_plugins.components.feature_chainer.feature_chain_author_guards"


@pytest.fixture(autouse=True, scope="module")
def _collect_leaked_local_feature_groups() -> Iterator[None]:
    """Function-local FeatureGroup subclasses are cyclic garbage; collect them before the next module runs."""
    yield
    gc.collect()


def _min_in_features_warnings(
    caplog: pytest.LogCaptureFixture, class_name: str | None = None
) -> list[logging.LogRecord]:
    """The MIN_IN_FEATURES definition-time warnings, optionally scoped to one class name."""
    records = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.name == AUTHOR_GUARDS_LOGGER
        and "declares no in_features" in record.getMessage()
    ]
    if class_name is not None:
        records = [record for record in records if class_name in record.getMessage()]
    return records


class TestMinInFeaturesWithoutInFeaturesWarns:
    def test_default_min_without_in_features_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _DefaultMinU1519a(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u1519a": PropertySpec("optional", default=None)}

            assert "opt_u1519a" in _DefaultMinU1519a.PROPERTY_MAPPING

        warnings = _min_in_features_warnings(caplog, "_DefaultMinU1519a")
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "MIN_IN_FEATURES" in message
        assert "in_features" in message
        assert "MIN_IN_FEATURES = 0" in message

    def test_min_two_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _MinTwoU1519b(FeatureChainParserMixin, FeatureGroup):
                MIN_IN_FEATURES = 2
                MAX_IN_FEATURES = None
                PROPERTY_MAPPING = {"opt_u1519b": PropertySpec("optional", default=None)}

            assert _MinTwoU1519b.MIN_IN_FEATURES == 2

        assert len(_min_in_features_warnings(caplog, "_MinTwoU1519b")) == 1

    def test_required_when_key_without_in_features_still_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The required_when guard re-wraps the matcher; that wrapper is not a custom matcher."""
        with caplog.at_level(logging.WARNING):

            class _RequiredWhenU1519c(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    "cond_u1519c": PropertySpec(
                        "conditionally required",
                        default=None,
                        required_when=(lambda options: options.get("other_u1519c") is None),
                    ),
                }

            assert "cond_u1519c" in _RequiredWhenU1519c.PROPERTY_MAPPING

        assert len(_min_in_features_warnings(caplog, "_RequiredWhenU1519c")) == 1


class TestMinInFeaturesWithoutInFeaturesDoesNotWarn:
    def test_in_features_key_declared_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _DeclaredU1519d(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {
                    DefaultOptionKeys.in_features: PropertySpec("sources", context=True, default=None),
                    "opt_u1519d": PropertySpec("optional", default=None),
                }

            assert DefaultOptionKeys.in_features in _DeclaredU1519d.PROPERTY_MAPPING

        assert not _min_in_features_warnings(caplog, "_DeclaredU1519d")

    def test_min_zero_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _MinZeroU1519e(FeatureChainParserMixin, FeatureGroup):
                MIN_IN_FEATURES = 0
                PROPERTY_MAPPING = {"opt_u1519e": PropertySpec("optional", default=None)}

            assert _MinZeroU1519e.MIN_IN_FEATURES == 0

        assert not _min_in_features_warnings(caplog, "_MinZeroU1519e")

    def test_overridden_input_features_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _InputFeaturesU1519f(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u1519f": PropertySpec("optional", default=None)}

                def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
                    return {Feature("src_u1519f")}

            assert "opt_u1519f" in _InputFeaturesU1519f.PROPERTY_MAPPING

        assert not _min_in_features_warnings(caplog, "_InputFeaturesU1519f")

    def test_prefix_pattern_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _PrefixU1519g(FeatureChainParserMixin, FeatureGroup):
                PREFIX_PATTERN = r".*__(?P<mode_u1519g>\w+)_u1519g$"
                PROPERTY_MAPPING = {"opt_u1519g": PropertySpec("optional", default=None)}

            assert _PrefixU1519g.PREFIX_PATTERN

        assert not _min_in_features_warnings(caplog, "_PrefixU1519g")

    def test_property_mapping_none_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _NoMappingU1519h(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = None

            assert _NoMappingU1519h.PROPERTY_MAPPING is None

        assert not _min_in_features_warnings(caplog, "_NoMappingU1519h")

    def test_custom_matcher_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _CustomMatcherU1519i(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u1519i": PropertySpec("optional", default=None)}

                @classmethod
                def match_feature_group_criteria(
                    cls,
                    feature_name: str | FeatureName,
                    options: Options,
                    data_access_collection: Any = None,
                ) -> bool:
                    return str(feature_name) == "specific_u1519i"

            assert _CustomMatcherU1519i.match_feature_group_criteria("specific_u1519i", Options()) is True

        assert not _min_in_features_warnings(caplog, "_CustomMatcherU1519i")

    def test_custom_source_extraction_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _CustomExtractU1519j(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u1519j": PropertySpec("optional", default=None)}

                @classmethod
                def _extract_source_features(cls, feature: Feature) -> list[str]:
                    return ["src_u1519j"]

            assert _CustomExtractU1519j._extract_source_features(Feature("x_u1519j")) == ["src_u1519j"]

        assert not _min_in_features_warnings(caplog, "_CustomExtractU1519j")

    def test_shipped_aggregated_feature_group_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _ShippedSubclassU1519k(AggregatedFeatureGroup):
                pass

            assert issubclass(_ShippedSubclassU1519k, AggregatedFeatureGroup)

        assert not _min_in_features_warnings(caplog, "AggregatedFeatureGroup")
        assert not _min_in_features_warnings(caplog, "_ShippedSubclassU1519k")

    def test_subclass_adding_in_features_key_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):

            class _ParentU1519l(FeatureChainParserMixin, FeatureGroup):
                PROPERTY_MAPPING = {"opt_u1519l": PropertySpec("optional", default=None)}

            class _ChildU1519l(_ParentU1519l):
                PROPERTY_MAPPING = {
                    **_ParentU1519l.PROPERTY_MAPPING,
                    DefaultOptionKeys.in_features: PropertySpec("sources", context=True, default=None),
                }

            assert DefaultOptionKeys.in_features in _ChildU1519l.PROPERTY_MAPPING

        assert len(_min_in_features_warnings(caplog, "_ParentU1519l")) == 1
        assert not _min_in_features_warnings(caplog, "_ChildU1519l")
