"""Catalog search and describe over a settings-scoped set of FeatureGroups and configured sources."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import RESERVED_READER_OPTION_KEY
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.interfaces.service.catalog import (
    MAX_DESCRIPTION_LENGTH,
    MAX_SEARCH_LIMIT,
    Catalog,
    plugin_collector_for,
    sanitize_text,
)
from mloda.interfaces.service.config import Settings, SourceConfig
from mloda.interfaces.service.errors import is_error
from mloda.provider import FeatureChainParserMixin, property_spec
from mloda.core.abstract_plugins.components.plugin_option.plugin_collector import VALID_STRICT_MODES
from mloda.user import PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import PythonDictFramework
from tests.helpers.docs_scope import small_feature_group_walk
from tests.helpers.plugin_stubs import StubFeatureGroup
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_source_group import ToySourceFG
from tests.test_interfaces.source_groups import (
    CALLER_LOCKED_FEATURE,
    CALLER_LOCKED_KEY,
    CALLER_OPEN_FEATURE,
    CallerLockedMcp6,
    CallerOpenMcp7,
)

# Cold worker under xdist load: PluginLoader.all plus hashing the scoped shipped groups can exceed the default timeout.
pytestmark = pytest.mark.timeout(30)

SECRET = "s3://user:pw@host/x"  # nosec B105
DENIED = "CatalogDeniedMcp3"
DENIED_CALLS: list[str] = []


class CatalogAlphaMcp3(StubFeatureGroup):
    """Rolling totals over ledger rows."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_alpha_total", "mcp3_alpha_count"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES
    PROPERTY_MAPPING = {
        "mcp3_window": property_spec("Window size in rows.", deferred_binding=True),
        "mcp3_mode": property_spec("Aggregation mode.", strict=True, allowed_values=("sum", "mean"), default="sum"),
        "mcp3_note": property_spec("", default="none"),
    }


class CatalogBetaMcp3(StubFeatureGroup):
    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset(f"mcp3_beta_{i:02d}" for i in range(25))
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class CatalogChainMcp3(FeatureChainParserMixin, StubFeatureGroup):
    """Chained sums."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_chain_sum"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 2

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) in cls.MATCHED_NAMES


class CatalogGammaMcp3(StubFeatureGroup):
    """Gamma \x1b[31mentries\x1b[0m."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_ledger_balance"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class LedgerArchiveMcp3(StubFeatureGroup):
    """Archived pages."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_archive"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES

    @classmethod
    def description(cls) -> str:
        return "Archived   pages.\n" + "x" * 400


class LedgerBookMcp3(StubFeatureGroup):
    """Book entries."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_book"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = MATCHED_NAMES


class CatalogDeniedMcp3(StubFeatureGroup):
    """Restricted ledger payroll data."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset({"mcp3_salary"})
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = {PythonDictFramework}
    ARMED: ClassVar[bool] = False

    @classmethod
    def _record(cls, hook: str) -> None:
        if cls.ARMED:
            DENIED_CALLS.append(hook)

    @classmethod
    def description(cls) -> str:
        cls._record("description")
        return "Restricted ledger payroll data."

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        cls._record("feature_names_supported")
        return {"mcp3_salary"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        cls._record("match_feature_group_criteria")
        return str(feature_name) in cls.MATCHED_NAMES


ALLOWED: list[type[FeatureGroup]] = [
    CatalogAlphaMcp3,
    CatalogBetaMcp3,
    CatalogChainMcp3,
    CatalogGammaMcp3,
    LedgerArchiveMcp3,
    LedgerBookMcp3,
]
ALLOWED_NAMES = sorted(fg.__name__ for fg in ALLOWED)


def _base() -> PluginCollector:
    return PluginCollector.enabled_feature_groups({*ALLOWED, CatalogDeniedMcp3})


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    sources = {
        "sales": SourceConfig(alias="sales", files=(str(tmp_path / "sales.csv"),)),
        "archive_dir": SourceConfig(alias="archive_dir", folders=(str(tmp_path / "archive"),)),
    }
    return Settings(deny_feature_groups=(DENIED,), sources=sources, **kwargs)


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    return Catalog.build(_settings(tmp_path), _base())


@pytest.fixture
def denied_calls() -> Iterator[list[str]]:
    DENIED_CALLS.clear()
    CatalogDeniedMcp3.ARMED = True
    yield DENIED_CALLS
    CatalogDeniedMcp3.ARMED = False
    DENIED_CALLS.clear()


def _fg_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in result["items"] if item["kind"] == "feature_group"]


def _fg_names(result: dict[str, Any]) -> list[str]:
    return [item["feature_group"] for item in _fg_items(result)]


def _error_code(result: dict[str, Any]) -> str:
    assert is_error(result), result
    code: str = result["error"]["code"]
    return code


def _walk(catalog: Catalog, query: str, limit: int) -> list[list[dict[str, Any]]]:
    pages: list[list[dict[str, Any]]] = []
    cursor: str | None = None
    while True:
        result = catalog.search(query, limit=limit, cursor=cursor)
        assert not is_error(result), result
        pages.append(result["items"])
        cursor = result["next_cursor"]
        if cursor is None:
            return pages
        assert len(pages) <= 20, "paging does not terminate"


class TestSanitizeText:
    def test_constants(self) -> None:
        assert MAX_DESCRIPTION_LENGTH == 300
        assert MAX_SEARCH_LIMIT == 50

    def test_none_is_empty(self) -> None:
        assert sanitize_text(None) == ""

    def test_strips_ansi_escapes_and_control_characters(self) -> None:
        assert sanitize_text("\x1b[31mred\x1b[0m\x00\x07") == "red"

    def test_collapses_whitespace(self) -> None:
        assert sanitize_text("  a \n\t b   c  ") == "a b c"

    def test_short_text_is_unchanged(self) -> None:
        assert sanitize_text("abc", limit=3) == "abc"

    def test_truncates_with_ellipsis(self) -> None:
        assert sanitize_text("abcdef", limit=4) == "abc…"

    def test_default_limit(self) -> None:
        result = sanitize_text("a" * 400)
        assert len(result) == MAX_DESCRIPTION_LENGTH
        assert result.endswith("…")


class TestPluginCollectorFor:
    def test_denies_by_class_name_and_keeps_base_enabled_set(self) -> None:
        base = PluginCollector.enabled_feature_groups({CatalogAlphaMcp3, CatalogDeniedMcp3})

        collector = plugin_collector_for(Settings(deny_feature_groups=(DENIED,)), base)

        assert isinstance(collector, PluginCollector)
        assert collector.applicable_feature_group_class(CatalogAlphaMcp3) is True
        assert collector.applicable_feature_group_class(CatalogDeniedMcp3) is False
        assert collector.applicable_feature_group_class(CatalogBetaMcp3) is False

    def test_without_base_only_denied_are_disabled(self) -> None:
        collector = plugin_collector_for(Settings(deny_feature_groups=(DENIED,)))

        assert collector.applicable_feature_group_class(CatalogAlphaMcp3) is True
        assert collector.applicable_feature_group_class(CatalogBetaMcp3) is True
        assert collector.applicable_feature_group_class(CatalogDeniedMcp3) is False

    def test_keeps_base_disabled_implementations(self, make_implementation: Any) -> None:
        disabled = make_implementation(ToySourceFG, PythonDictFramework, name="CatalogDisabledImpl")
        kept = make_implementation(ToySourceFG, PythonDictFramework, name="CatalogKeptImpl")
        base = PluginCollector()
        base.add_disabled_implementation_classes({disabled})

        collector = plugin_collector_for(Settings(), base)

        assert collector.applicable_implementation_class(disabled) is False
        assert collector.applicable_implementation_class(kept) is True

    def test_copy_keeps_every_base_attribute_and_isolates_sets(self, make_implementation: Any) -> None:
        base = PluginCollector()
        registry_sentinel = object()
        impl = make_implementation(ToySourceFG, PythonDictFramework, name="CatalogDriftImpl")
        for name, value in vars(PluginCollector()).items():
            if name == "disabled_feature_group_classes":
                base.disabled_feature_group_classes.add(CatalogBetaMcp3)
            elif name == "enabled_feature_group_classes":
                base.enabled_feature_group_classes.add(CatalogAlphaMcp3)
            elif name == "disabled_implementation_classes":
                base.disabled_implementation_classes.add(impl)
            elif name == "strict_mode":
                base.strict_mode = next(m for m in VALID_STRICT_MODES if m != base.strict_mode)
            elif name == "registry":
                setattr(base, name, registry_sentinel)
            elif isinstance(value, bool):
                setattr(base, name, not value)
            else:
                pytest.fail(f"PluginCollector.{name} ({type(value).__name__}) is not handled by the copy drift test")
        disabled_before = set(base.disabled_feature_group_classes)

        direct = base.copy()
        collector = plugin_collector_for(Settings(deny_feature_groups=(DENIED,)), base)

        assert base.disabled_feature_group_classes == disabled_before
        assert collector.registry is base.registry
        for name, value in vars(base).items():
            for copy_of in (direct, collector):
                kept = getattr(copy_of, name)
                if isinstance(value, set):
                    assert kept is not value
                    if copy_of is collector and name == "disabled_feature_group_classes":
                        continue
                    assert kept == value
                else:
                    assert kept == value
        assert direct.disabled_feature_group_classes == disabled_before
        assert collector.disabled_feature_group_classes == disabled_before | {CatalogDeniedMcp3}


class TestBuildDoesNotTouchDenied:
    def test_build_and_search_never_call_into_denied(self, tmp_path: Path, denied_calls: list[str]) -> None:
        catalog = Catalog.build(_settings(tmp_path), _base())
        result = catalog.search("", limit=MAX_SEARCH_LIMIT)

        assert DENIED not in _fg_names(result)
        assert denied_calls == []

    def test_build_without_collector_excludes_denied(
        self, tmp_path: Path, denied_calls: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        walk = small_feature_group_walk(*ALLOWED, CatalogDeniedMcp3)
        monkeypatch.setattr("mloda.core.api.plugin_docs.get_all_subclasses", walk)
        catalog = Catalog.build(_settings(tmp_path))
        result = catalog.search("Mcp3", kinds=["feature_group"], limit=MAX_SEARCH_LIMIT)

        names = _fg_names(result)
        assert DENIED not in names
        assert set(ALLOWED_NAMES) <= set(names)
        assert denied_calls == []


class TestSearch:
    def test_empty_query_lists_everything(self, catalog: Catalog) -> None:
        result = catalog.search("", limit=MAX_SEARCH_LIMIT)

        assert set(result) == {"items", "next_cursor"}
        assert result["next_cursor"] is None
        assert sorted(_fg_names(result)) == ALLOWED_NAMES
        assert all(item["match"] == "all" for item in _fg_items(result))
        assert sorted(item["alias"] for item in result["items"] if item["kind"] == "source") == [
            "archive_dir",
            "sales",
        ]

    def test_empty_query_orders_feature_groups_alphabetically(self, catalog: Catalog) -> None:
        result = catalog.search("", kinds=["feature_group"], limit=MAX_SEARCH_LIMIT)
        assert _fg_names(result) == ALLOWED_NAMES

    def test_feature_group_item_shape(self, catalog: Catalog) -> None:
        result = catalog.search("CatalogAlphaMcp3", kinds=["feature_group"])

        assert _fg_items(result) == [
            {
                "kind": "feature_group",
                "feature_group": "CatalogAlphaMcp3",
                "description": "Rolling totals over ledger rows.",
                "prefix": "CatalogAlphaMcp3_",
                "supported_feature_names": ["mcp3_alpha_count", "mcp3_alpha_total"],
                "match": "name",
            }
        ]

    def test_supported_feature_names_are_capped_at_twenty(self, catalog: Catalog) -> None:
        (item,) = _fg_items(catalog.search("CatalogBetaMcp3", kinds=["feature_group"]))
        assert item["supported_feature_names"] == [f"mcp3_beta_{i:02d}" for i in range(20)]

    def test_description_is_sanitized(self, catalog: Catalog) -> None:
        (gamma,) = _fg_items(catalog.search("CatalogGammaMcp3", kinds=["feature_group"]))
        (archive,) = _fg_items(catalog.search("LedgerArchiveMcp3", kinds=["feature_group"]))

        assert gamma["description"] == "Gamma entries."
        assert archive["description"].startswith("Archived pages. xxx")
        assert len(archive["description"]) == MAX_DESCRIPTION_LENGTH
        assert archive["description"].endswith("…")

    def test_ranking_name_then_feature_name_then_description(self, catalog: Catalog) -> None:
        result = catalog.search("LEDGER", kinds=["feature_group"])

        assert [(item["feature_group"], item["match"]) for item in result["items"]] == [
            ("LedgerArchiveMcp3", "name"),
            ("LedgerBookMcp3", "name"),
            ("CatalogGammaMcp3", "feature_name"),
            ("CatalogAlphaMcp3", "description"),
        ]

    def test_feature_name_match(self, catalog: Catalog) -> None:
        result = catalog.search("beta_07", kinds=["feature_group"])
        assert [(item["feature_group"], item["match"]) for item in result["items"]] == [
            ("CatalogBetaMcp3", "feature_name")
        ]

    def test_no_match(self, catalog: Catalog) -> None:
        assert catalog.search("zzz_matches_nothing_mcp3") == {"items": [], "next_cursor": None}

    def test_denied_feature_group_never_appears(self, catalog: Catalog) -> None:
        assert catalog.search("payroll")["items"] == []
        assert catalog.search("mcp3_salary")["items"] == []
        assert DENIED not in _fg_names(catalog.search("ledger", limit=MAX_SEARCH_LIMIT))

    def test_kinds_filter_feature_groups(self, catalog: Catalog) -> None:
        result = catalog.search("", kinds=["feature_group"], limit=MAX_SEARCH_LIMIT)
        assert {item["kind"] for item in result["items"]} == {"feature_group"}

    def test_kinds_filter_sources(self, catalog: Catalog) -> None:
        result = catalog.search("", kinds=["source"], limit=MAX_SEARCH_LIMIT)

        by_alias = {item["alias"]: item for item in result["items"]}
        assert {item["kind"] for item in result["items"]} == {"source"}
        assert sorted(by_alias) == ["archive_dir", "sales"]
        assert by_alias["sales"]["type"] == "files"
        assert by_alias["archive_dir"]["type"] == "folders"

    def test_sources_match_on_alias(self, catalog: Catalog) -> None:
        result = catalog.search("SAL", kinds=["source"])

        assert [(item["kind"], item["alias"], item["type"]) for item in result["items"]] == [
            ("source", "sales", "files")
        ]
        assert catalog.search("csv", kinds=["source"])["items"] == []

    def test_sources_never_expose_paths(self, catalog: Catalog, tmp_path: Path) -> None:
        dumped = json.dumps(catalog.search("", limit=MAX_SEARCH_LIMIT))
        assert str(tmp_path) not in dumped
        assert "sales.csv" not in dumped


class TestSearchPaging:
    @pytest.mark.parametrize("limit", [1, 3, 7])
    def test_walking_cursors_returns_every_item_once_in_order(self, catalog: Catalog, limit: int) -> None:
        full = catalog.search("", limit=MAX_SEARCH_LIMIT)["items"]

        pages = _walk(catalog, "", limit)

        assert all(len(page) <= limit for page in pages)
        assert all(len(page) == limit for page in pages[:-1])
        assert [item for page in pages for item in page] == full

    def test_first_page_has_cursor(self, catalog: Catalog) -> None:
        result = catalog.search("", limit=2)

        assert len(result["items"]) == 2
        cursor = result["next_cursor"]
        assert isinstance(cursor, str)
        assert not cursor.strip().lstrip("-").isdigit()

    def test_malformed_cursor_is_rejected(self, catalog: Catalog) -> None:
        assert _error_code(catalog.search("", cursor="not-a-cursor")) == "validation_failed"

    def test_integer_cursor_is_rejected(self, catalog: Catalog) -> None:
        assert _error_code(catalog.search("", cursor="2")) == "validation_failed"

    def test_cursor_from_another_query_is_rejected(self, catalog: Catalog) -> None:
        cursor = catalog.search("", limit=2)["next_cursor"]
        assert cursor is not None

        assert _error_code(catalog.search("ledger", limit=2, cursor=cursor)) == "validation_failed"

    @pytest.mark.parametrize("limit", [0, -1, MAX_SEARCH_LIMIT + 1])
    def test_limit_out_of_range_is_rejected(self, catalog: Catalog, limit: int) -> None:
        assert _error_code(catalog.search("", limit=limit)) == "validation_failed"

    @pytest.mark.parametrize("limit", [1, MAX_SEARCH_LIMIT])
    def test_limit_bounds_are_accepted(self, catalog: Catalog, limit: int) -> None:
        assert not is_error(catalog.search("", limit=limit))


class TestDescribe:
    def test_success_shape(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_alpha_total")

        assert not is_error(result), result
        assert set(result) == {
            "name",
            "feature_group",
            "candidates",
            "description",
            "options",
            "in_features",
            "metadata_missing",
        }
        assert result["name"] == "mcp3_alpha_total"
        assert result["feature_group"] == "CatalogAlphaMcp3"
        assert result["candidates"] == ["CatalogAlphaMcp3"]
        assert result["description"] == "Rolling totals over ledger rows."
        assert result["in_features"] is None
        json.dumps(result)

    def test_options_from_property_mapping(self, catalog: Catalog) -> None:
        options = catalog.describe("mcp3_alpha_total")["options"]

        by_name = {entry["name"]: entry for entry in options}
        assert len(options) == 3
        assert sorted(by_name) == ["mcp3_mode", "mcp3_note", "mcp3_window"]

        window = by_name["mcp3_window"]
        assert window["explanation"] == "Window size in rows."
        assert window["required"] is True
        assert "default" not in window
        assert "allowed_values" not in window

        mode = by_name["mcp3_mode"]
        assert mode["explanation"] == "Aggregation mode."
        assert mode["required"] is False
        assert mode["default"] == "sum"
        assert sorted(mode["allowed_values"]) == ["mean", "sum"]

        note = by_name["mcp3_note"]
        assert note["explanation"] == ""
        assert note["required"] is False
        assert note["default"] == "none"

    def test_options_are_not_caller_settable_by_default(self, catalog: Catalog) -> None:
        options = catalog.describe("mcp3_alpha_total")["options"]

        assert [entry["caller_settable"] for entry in options] == [False, False, False]

    def test_listed_options_are_caller_settable(self, tmp_path: Path) -> None:
        listed = Catalog.build(_settings(tmp_path, allow_options=("mcp3_mode", "mcp3_window")), _base())

        options = listed.describe("mcp3_alpha_total")["options"]

        assert {entry["name"]: entry["caller_settable"] for entry in options} == {
            "mcp3_mode": True,
            "mcp3_note": False,
            "mcp3_window": True,
        }

    @pytest.mark.parametrize("listed", [(), (CALLER_LOCKED_KEY,)])
    def test_flagged_option_is_not_caller_settable_even_when_listed(
        self, tmp_path: Path, listed: tuple[str, ...]
    ) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6})
        flagged = Catalog.build(_settings(tmp_path, allow_options=listed), collector)

        options = flagged.describe(CALLER_LOCKED_FEATURE)["options"]

        assert {entry["name"]: entry["caller_settable"] for entry in options} == {CALLER_LOCKED_KEY: False}

    @pytest.mark.parametrize("listed", [(), (CALLER_LOCKED_KEY,)])
    def test_key_flagged_by_another_group_is_not_caller_settable(self, tmp_path: Path, listed: tuple[str, ...]) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6, CallerOpenMcp7})
        flagged = Catalog.build(_settings(tmp_path, allow_options=listed), collector)

        options = flagged.describe(CALLER_OPEN_FEATURE)["options"]

        assert {entry["name"]: entry["caller_settable"] for entry in options} == {CALLER_LOCKED_KEY: False}

    def test_metadata_missing_flags_empty_explanation(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_alpha_total")
        assert result["metadata_missing"] == ["explanation for option 'mcp3_note'"]

    def test_metadata_missing_flags_missing_docstring(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_beta_03")

        assert not is_error(result), result
        assert result["feature_group"] == "CatalogBetaMcp3"
        assert result["options"] == []
        assert result["metadata_missing"] == ["description"]

    def test_in_features_from_chain_parser_mixin(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_chain_sum")

        assert not is_error(result), result
        assert result["feature_group"] == "CatalogChainMcp3"
        assert result["in_features"] == {"min": 1, "max": 2}
        assert result["metadata_missing"] == []

    def test_unknown_name_with_suggestion(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_alpha_totl")

        assert _error_code(result) == "validation_failed"
        (detail,) = result["error"]["details"]
        assert detail["path"] == "/name"
        assert detail["problem"] == "unknown_feature"
        assert detail["suggestion"] == "mcp3_alpha_total"

    def test_unknown_name_without_close_match(self, catalog: Catalog) -> None:
        result = catalog.describe("qqqqzzzzwwww")

        assert _error_code(result) == "validation_failed"
        (detail,) = result["error"]["details"]
        assert detail["path"] == "/name"
        assert detail["problem"] == "unknown_feature"
        assert "suggestion" not in detail

    def test_denied_name_is_policy_denied(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_salary")

        assert _error_code(result) == "policy_denied"
        assert "alternatives" not in result["error"]
        assert "deny_feature_groups" not in json.dumps(result)

    def test_denied_name_carries_alternatives(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, alternatives={"mcp3_salary": ("mcp3_alpha_total",)})
        catalog = Catalog.build(settings, _base())

        result = catalog.describe("mcp3_salary")

        assert _error_code(result) == "policy_denied"
        assert result["error"]["alternatives"] == ["mcp3_alpha_total"]
        assert "deny_feature_groups" not in json.dumps(result)

    @pytest.mark.parametrize("key", ["CsvFG", "JsonFG", RESERVED_READER_OPTION_KEY])
    def test_data_access_option_is_policy_denied(self, catalog: Catalog, key: str) -> None:
        result = catalog.describe("mcp3_alpha_total", options={"mcp3_mode": "sum", key: SECRET})

        assert _error_code(result) == "policy_denied"
        assert [detail["path"] for detail in result["error"]["details"]] == [f"/options/{key}"]
        assert SECRET not in json.dumps(result)


class TestDescribeOptionAllowList:
    OPTIONS: ClassVar[dict[str, Any]] = {"mcp3_mode": "sum"}

    def test_unlisted_option_key_is_refused_by_pointer(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_alpha_total", options=self.OPTIONS)

        assert _error_code(result) == "policy_denied"
        (detail,) = result["error"]["details"]
        assert detail["path"] == "/options/mcp3_mode"
        assert detail["problem"] == "option_not_allowed"

    def test_listed_option_key_is_accepted(self, tmp_path: Path) -> None:
        catalog = Catalog.build(_settings(tmp_path, allow_options=("mcp3_mode",)), _base())

        assert not is_error(catalog.describe("mcp3_alpha_total", options=self.OPTIONS))

    def test_trusted_caller_may_set_any_option(self, tmp_path: Path) -> None:
        catalog = Catalog.build(_settings(tmp_path, trusted_caller=True), _base())

        assert not is_error(catalog.describe("mcp3_alpha_total", options=self.OPTIONS))

    def test_data_access_check_comes_first(self, catalog: Catalog) -> None:
        result = catalog.describe("mcp3_alpha_total", options={"CsvFG": SECRET, "mcp3_mode": "sum"})

        assert [detail["problem"] for detail in result["error"]["details"]] != ["option_not_allowed"]
        assert SECRET not in json.dumps(result)

    @pytest.mark.parametrize("listed", [(), (CALLER_LOCKED_KEY,)])
    def test_flagged_key_is_refused_even_when_listed(self, tmp_path: Path, listed: tuple[str, ...]) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6})
        flagged = Catalog.build(_settings(tmp_path, allow_options=listed), collector)

        result = flagged.describe(CALLER_LOCKED_FEATURE, options={CALLER_LOCKED_KEY: SECRET})

        assert _error_code(result) == "policy_denied"
        (detail,) = result["error"]["details"]
        assert detail["path"] == f"/options/{CALLER_LOCKED_KEY}"
        assert detail["problem"] == "option_not_caller_settable"
        assert SECRET not in json.dumps(result)

    @pytest.mark.parametrize("listed", [(), (CALLER_LOCKED_KEY,)])
    def test_key_flagged_by_another_group_is_refused(self, tmp_path: Path, listed: tuple[str, ...]) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6, CallerOpenMcp7})
        flagged = Catalog.build(_settings(tmp_path, allow_options=listed), collector)

        result = flagged.describe(CALLER_OPEN_FEATURE, options={CALLER_LOCKED_KEY: SECRET})

        assert _error_code(result) == "policy_denied"
        (detail,) = result["error"]["details"]
        assert detail["problem"] == "option_not_caller_settable"
        assert SECRET not in json.dumps(result)

    def test_trusted_caller_may_set_flagged_key(self, tmp_path: Path) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6})
        trusted = Catalog.build(_settings(tmp_path, trusted_caller=True), collector)

        assert not is_error(trusted.describe(CALLER_LOCKED_FEATURE, options={CALLER_LOCKED_KEY: SECRET}))

    def test_data_access_check_precedes_flagged_key_check(self, tmp_path: Path) -> None:
        collector = PluginCollector.enabled_feature_groups({*ALLOWED, CallerLockedMcp6})
        flagged = Catalog.build(_settings(tmp_path, allow_options=(CALLER_LOCKED_KEY,)), collector)

        result = flagged.describe(CALLER_LOCKED_FEATURE, options={"CsvFG": SECRET, CALLER_LOCKED_KEY: SECRET})

        assert [detail["problem"] for detail in result["error"]["details"]] == ["data_access_option"]
