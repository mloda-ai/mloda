"""Pins the PROPERTY_MAPPING declarations of the file, document and database groups and that every key their
matchers read is declared. Matching is called directly or through evaluate with an explicit plugin mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import ReadDBFG, ReadDocumentFG, ReadFileFG
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from mloda_plugins.feature_group.input_data.db_sources.sqlite_fg import SqliteFG
from tests.mixins.reader_feature_groups.lazy_source_group import load_document_group


class _RodRecordingOptions(Options):
    """Empty Options that records every key read through ``get``."""

    def __init__(self) -> None:
        super().__init__()
        self.read_keys: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.read_keys.append(key)
        return super().get(key, default)


@pytest.fixture
def json_path(tmp_path: Path) -> str:
    """A real ``.json`` file path in an isolated tmp dir."""
    path = tmp_path / "rod_payload.json"
    path.write_text('{"rod_value": 1}', encoding="utf-8")
    return str(path)


@pytest.fixture
def csv_path(tmp_path: Path) -> str:
    """A real ``.csv`` file path in an isolated tmp dir."""
    path = tmp_path / "rod_rows.csv"
    path.write_text("id,amount\n1,10\n", encoding="utf-8")
    return str(path)


class TestReadFileFGDeclarations:
    """ReadFileFG declares the two keys its matcher reads in PROPERTY_MAPPING, and its formats inherit them."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadFileFG.declared_option_keys() == {"document_suffixes", "data_access_handle"}

    def test_declared_values_are_property_specs(self) -> None:
        assert ReadFileFG.PROPERTY_MAPPING is not None
        assert all(isinstance(spec, PropertySpec) for spec in ReadFileFG.PROPERTY_MAPPING.values())

    def test_declared_defaults(self) -> None:
        assert ReadFileFG.PROPERTY_MAPPING is not None
        assert ReadFileFG.PROPERTY_MAPPING["document_suffixes"].default is None
        assert ReadFileFG.PROPERTY_MAPPING["data_access_handle"].default is None

    def test_json_group_inherits_without_redeclaring(self) -> None:
        assert "PROPERTY_MAPPING" not in JsonFG.__dict__
        assert JsonFG.declared_option_keys() == ReadFileFG.declared_option_keys()


class TestReadDocumentFGDeclarations:
    """ReadDocumentFG declares the same two keys as the file groups, and its formats inherit them."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadDocumentFG.declared_option_keys() == {"document_suffixes", "data_access_handle"}

    def test_declared_values_are_property_specs_with_none_defaults(self) -> None:
        assert all(isinstance(spec, PropertySpec) for spec in ReadDocumentFG.PROPERTY_MAPPING.values())
        assert ReadDocumentFG.PROPERTY_MAPPING["document_suffixes"].default is None
        assert ReadDocumentFG.PROPERTY_MAPPING["data_access_handle"].default is None

    @pytest.mark.parametrize("module,name", [("markdown_fg", "MarkdownFG"), ("json_document_fg", "JsonDocumentFG")])
    def test_document_groups_inherit_without_redeclaring(self, module: str, name: str) -> None:
        group = load_document_group(module, name)
        assert "PROPERTY_MAPPING" not in group.__dict__
        assert group.declared_option_keys() == ReadDocumentFG.declared_option_keys()

    def test_text_group_adds_only_text_lines(self) -> None:
        group = load_document_group("text_fg", "TextFG")
        assert group.declared_option_keys() == ReadDocumentFG.declared_option_keys() | {"text_lines"}


class TestReadDBFGDeclarations:
    """ReadDBFG reads only the handle hint, so it declares only that key, and its formats inherit it."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadDBFG.declared_option_keys() == {"data_access_handle"}

    def test_declared_values_are_property_specs(self) -> None:
        assert ReadDBFG.PROPERTY_MAPPING is not None
        assert all(isinstance(spec, PropertySpec) for spec in ReadDBFG.PROPERTY_MAPPING.values())

    def test_declared_default(self) -> None:
        assert ReadDBFG.PROPERTY_MAPPING is not None
        assert ReadDBFG.PROPERTY_MAPPING["data_access_handle"].default is None

    def test_document_suffixes_is_not_a_read_db_key(self) -> None:
        assert "document_suffixes" not in ReadDBFG.declared_option_keys()

    def test_sqlite_group_inherits_without_redeclaring(self) -> None:
        assert "PROPERTY_MAPPING" not in SqliteFG.__dict__
        assert SqliteFG.declared_option_keys() == ReadDBFG.declared_option_keys()


class TestDuckdbFGDeclarations:
    """DuckdbFG selects its connection with connection_handle and declares exactly that key."""

    def test_declares_exactly_connection_handle(self) -> None:
        from mloda_plugins.feature_group.input_data.db_sources.duckdb_fg import DuckdbFG

        assert DuckdbFG.declared_option_keys() == {"connection_handle"}


class TestEveryOptionKeyReadIsDeclared:
    """Observed match-time reads are a subset of the declared keys, per source reader."""

    def test_read_file_group_reads_only_declared_keys(self, json_path: str) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        assert JsonFG.match_feature_group_criteria("rod_value", options, data_access)
        read = set(options.read_keys) - {JsonFG.get_class_name()}
        assert read == {"document_suffixes", "data_access_handle"}
        assert read <= ReadFileFG.declared_option_keys()

    def test_read_document_group_reads_only_declared_keys(self, tmp_path: Path) -> None:
        text_group = load_document_group("text_fg", "TextFG")
        path = tmp_path / "rod_note.txt"
        path.write_text("note", encoding="utf-8")
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_note": str(path)})

        assert text_group.match_feature_group_criteria("TextFG", options, data_access)
        read = set(options.read_keys) - {text_group.get_class_name()}
        assert read == {"document_suffixes", "data_access_handle", "text_lines"}
        assert read <= text_group.declared_option_keys()

    def test_read_db_group_reads_only_declared_keys(self, tmp_path: Path) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(credentials=[{"sqlite": str(tmp_path / "rod.sqlite")}])

        assert not SqliteFG.match_feature_group_criteria("rod_any", options, data_access)
        read = set(options.read_keys) - {SqliteFG.get_class_name()}
        assert read == {"data_access_handle"}
        assert read <= ReadDBFG.declared_option_keys()


def _json_group_claims(json_path: str, options: Options) -> bool:
    feature = Feature("rod_value", options)
    dac = DataAccessCollection(files={"rod_payload": json_path})
    result = IdentifyFeatureGroupClass.evaluate(feature, {JsonFG: {PyArrowTable}}, None, dac)
    return JsonFG in result.identified


class TestJsonFGDocumentSuffixes:
    """``document_suffixes`` is read through the declared PROPERTY_MAPPING key and only ever excludes."""

    def test_explicit_empty_option_excludes_nothing(self, json_path: str) -> None:
        assert _json_group_claims(json_path, Options(context={"document_suffixes": frozenset()}))

    def test_explicit_none_reads_as_absent(self, json_path: str) -> None:
        assert _json_group_claims(json_path, Options(context={"document_suffixes": None}))

    @pytest.mark.parametrize("value", [5, frozenset({1}), [1, "a"]])
    def test_a_non_str_collection_value_is_rejected_by_the_option_declaration(self, json_path: str, value: Any) -> None:
        feature = Feature("rod_value", Options(context={"document_suffixes": value}))
        dac = DataAccessCollection(files={"rod_payload": json_path})
        result = IdentifyFeatureGroupClass.evaluate(feature, {JsonFG: {PyArrowTable}}, None, dac)
        assert JsonFG not in result.identified
        reason = result.eliminations[JsonFG].reason
        assert "document_suffixes" in reason
        assert "raised TypeError" not in reason


def _json_document_claims(json_path: str, options: Options) -> bool:
    group = load_document_group("json_document_fg", "JsonDocumentFG")
    dac = DataAccessCollection(files={"rod_payload": json_path})
    result = IdentifyFeatureGroupClass.evaluate(Feature("JsonDocumentFG", options), {group: {PyArrowTable}}, None, dac)
    return group in result.identified


class TestJsonDocumentFGDocumentSuffixes:
    """JsonDocumentFG reads document_suffixes through the same declared key: only a listing of ``.json`` claims."""

    def test_an_explicit_empty_option_reads_as_not_listed(self, json_path: str) -> None:
        assert not _json_document_claims(json_path, Options(context={"document_suffixes": frozenset()}))

    def test_an_explicit_none_reads_as_absent(self, json_path: str) -> None:
        assert not _json_document_claims(json_path, Options(context={"document_suffixes": None}))

    @pytest.mark.parametrize("value", [".json", frozenset({".json"}), [".json", ".csv"]])
    def test_a_listing_of_the_json_suffix_claims(self, json_path: str, value: Any) -> None:
        assert _json_document_claims(json_path, Options(context={"document_suffixes": value}))

    @pytest.mark.parametrize("value", [5, frozenset({1}), [1, "a"]])
    def test_a_non_str_collection_value_never_claims_and_never_raises(self, json_path: str, value: Any) -> None:
        group = load_document_group("json_document_fg", "JsonDocumentFG")
        feature = Feature("JsonDocumentFG", Options(context={"document_suffixes": value}))
        dac = DataAccessCollection(files={"rod_payload": json_path})
        result = IdentifyFeatureGroupClass.evaluate(feature, {group: {PyArrowTable}}, None, dac)
        assert group not in result.identified
        assert not [e for e in result.eliminations.values() if "raised TypeError" in e.reason]


def _extra_spec() -> PropertySpec:
    return PropertySpec("An extra declared key.", default=None, context=True)


class TestOwnPropertyMappingIsMergedOverTheBase:
    @pytest.mark.parametrize("own_mapping", [True, False])
    def test_a_plain_mixin_first_in_the_bases_keeps_the_mixin_and_base_keys(self, own_mapping: bool) -> None:
        mixin: Any = type("RodMixin", (), {"PROPERTY_MAPPING": {"mixin_key": _extra_spec()}})
        body: dict[str, Any] = {"PROPERTY_MAPPING": {"own_key": _extra_spec()}} if own_mapping else {}
        child: Any = type("RodMixinFG", (mixin, ReadFileFG), body)
        expected = {"mixin_key", "data_access_handle", "document_suffixes"} | ({"own_key"} if own_mapping else set())
        assert child.declared_option_keys() == expected
        assert not child.passes_option_declarations(Options(context={"data_access_handle": 5}))
        assert child.passes_option_declarations(Options(context={"data_access_handle": "h"}))

    @pytest.mark.parametrize("base", [ReadFileFG, ReadDocumentFG, ReadDBFG])
    def test_an_own_extra_key_keeps_every_base_key(self, base: Any) -> None:
        child: Any = type("RodExtraKeyFG", (base,), {"PROPERTY_MAPPING": {"rod_extra": _extra_spec()}})
        assert child.declared_option_keys() == base.declared_option_keys() | {"rod_extra"}
        assert "rod_extra" not in base.declared_option_keys()

    @pytest.mark.parametrize("base", [ReadFileFG, ReadDocumentFG])
    def test_an_own_key_overrides_the_base_key(self, base: Any) -> None:
        own = _extra_spec()
        child: Any = type("RodOverrideKeyFG", (base,), {"PROPERTY_MAPPING": {"data_access_handle": own}})
        assert child.PROPERTY_MAPPING["data_access_handle"] is own

    @pytest.mark.parametrize("own_mapping", [True, False])
    def test_a_diamond_takes_each_key_from_the_first_class_in_the_mro_that_declares_it(self, own_mapping: bool) -> None:
        own = _extra_spec()
        a: Any = type("RodDiamondA", (ReadFileFG,), {"PROPERTY_MAPPING": {"a_key": _extra_spec()}})
        b: Any = type("RodDiamondB", (ReadFileFG,), {"PROPERTY_MAPPING": {"data_access_handle": own}})
        body: dict[str, Any] = {"PROPERTY_MAPPING": {"c_key": _extra_spec()}} if own_mapping else {}
        c: Any = type("RodDiamondC", (a, b), body)
        assert c.PROPERTY_MAPPING["data_access_handle"] is own
        assert {"a_key"} | ReadFileFG.declared_option_keys() <= c.declared_option_keys()
        assert ("c_key" in c.declared_option_keys()) == own_mapping

    def test_the_inherited_handle_check_still_rejects_a_non_str_handle(self) -> None:
        child: Any = type("RodHandleCheckFG", (ReadFileFG,), {"PROPERTY_MAPPING": {"rod_extra": _extra_spec()}})
        assert not child.passes_option_declarations(Options(context={"data_access_handle": 5}))
        assert child.passes_option_declarations(Options(context={"data_access_handle": "h"}))

    def test_a_parent_query_text_spec_survives_a_child_with_its_own_mapping(self) -> None:
        custom = PropertySpec("Custom query.", default="select 1", context=True)
        routes = (*ReadDBFG.CLAIM_ROUTES, ReadDBFG.QUERY_ROUTE)
        parent: Any = type(
            "RodQueryParentDB", (ReadDBFG,), {"CLAIM_ROUTES": routes, "PROPERTY_MAPPING": {"query_text": custom}}
        )
        child: Any = type("RodQueryChildDB", (parent,), {"PROPERTY_MAPPING": {"rod_extra": _extra_spec()}})
        assert parent.PROPERTY_MAPPING["query_text"] is custom
        assert child.PROPERTY_MAPPING["query_text"] is custom
        assert child.declared_option_keys() == parent.declared_option_keys() | {"rod_extra"}


class TestSourceGroupPropertyMappingMustBeAMappingOfSpecs:
    @pytest.mark.parametrize("bad", [None, ["k"]], ids=["none", "list"])
    @pytest.mark.parametrize("base", [ReadFileFG, ReadDocumentFG, ReadDBFG])
    def test_a_non_mapping_own_value_raises_a_type_error_naming_the_class(self, base: Any, bad: Any) -> None:
        with pytest.raises(TypeError, match="RodBadMappingFG.*PROPERTY_MAPPING must be a mapping of PropertySpec"):
            type("RodBadMappingFG", (base,), {"PROPERTY_MAPPING": bad})

    @pytest.mark.parametrize("base", [ReadFileFG, ReadDocumentFG, ReadDBFG])
    def test_a_non_mapping_mixin_value_raises_a_type_error_naming_the_defined_class(self, base: Any) -> None:
        mixin: Any = type("RodListMixin", (), {"PROPERTY_MAPPING": ["k"]})
        with pytest.raises(TypeError, match="RodListMixinFG.*PROPERTY_MAPPING must be a mapping of PropertySpec"):
            type("RodListMixinFG", (mixin, base), {})

    @pytest.mark.parametrize("own", [False, True], ids=["no_own_mapping", "own_mapping"])
    def test_a_mixin_with_a_none_mapping_is_not_an_error_and_keeps_the_base_keys(self, own: bool) -> None:
        mixin: Any = type("RodNoneMixin", (), {"PROPERTY_MAPPING": None})
        body: dict[str, Any] = {"PROPERTY_MAPPING": {"own_key": _extra_spec()}} if own else {}
        child: Any = type("RodNoneMixinFG", (mixin, ReadFileFG), body)
        expected = ReadFileFG.declared_option_keys() | ({"own_key"} if own else set())
        assert child.declared_option_keys() == expected
