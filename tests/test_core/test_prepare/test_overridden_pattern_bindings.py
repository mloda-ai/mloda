"""Intake and subtype resolution must share a mixin group's declared matcher."""

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import FeatureChainParserMixin, FeatureGroup, SubtypeDeclaration, property_spec


class ExtraPattern2326(FeatureChainParserMixin, FeatureGroup):
    PREFIX_PATTERN = r".*__(?P<operation2326>sum|max)_rolling2326$"
    PROPERTY_MAPPING = {
        "operation2326": property_spec("Operation", allowed_values={"sum", "max"}, strict=True),
    }
    SUBTYPES = SubtypeDeclaration(key="operation2326")

    @classmethod
    def _get_prefix_patterns(cls) -> list[str]:
        return [cls.PREFIX_PATTERN, r".*__(?P<operation2326>sum|max)_window2326$"]


def test_additional_pattern_is_recognized() -> None:
    assert ExtraPattern2326.resolve_feature_name("s__max_window2326").bindings == {"operation2326": "max"}


def test_name_bindings_include_additional_pattern() -> None:
    assert ExtraPattern2326.name_bindings("s__max_window2326") == {"operation2326": "max"}


def test_subtype_uses_additional_pattern_before_options() -> None:
    assert ExtraPattern2326.resolve_subtype("s__max_window2326", Options(context={"operation2326": "sum"})) == "max"


def test_primary_pattern_remains_available() -> None:
    assert ExtraPattern2326.name_bindings("s__sum_rolling2326") == {"operation2326": "sum"}
    assert ExtraPattern2326.resolve_subtype("s__sum_rolling2326", Options()) == "sum"


def test_unmatched_name_keeps_option_fallback() -> None:
    assert ExtraPattern2326.name_bindings("unmatched2326") == {}
    assert ExtraPattern2326.resolve_subtype("unmatched2326", Options(context={"operation2326": "sum"})) == "sum"
