"""Pins the per-reader ``READER_OPTIONS`` declarations (issue #949: ``PropertySpec`` values) and
makes the declared ``document_suffixes`` default load-bearing in ReadDocument matching, plus the ReadFileFG/JsonFG PROPERTY_MAPPING declarations.
Leak policy: the leaked readers here are never final; matching is called directly or through evaluate with
an explicit plugin mapping, never via mlodaAPI.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.provider import ReadDBFG, ReadFileFG
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.feature_group.input_data.file_formats.json_fg import JsonFG
from mloda_plugins.feature_group.input_data.db_formats.sqlite_fg import SqliteFG
from mloda_plugins.feature_group.input_data.read_document import ReadDocument
from mloda_plugins.feature_group.input_data.read_files.markdown_document_reader import MarkdownDocumentReader


_RESERVED_KEY = "BaseInputData"


class _RodRecordingOptions(Options):
    """Empty Options that records every key read through ``get``."""

    def __init__(self) -> None:
        super().__init__()
        self.read_keys: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.read_keys.append(key)
        return super().get(key, default)


class _RodStockJsonReadDocument(ReadDocument):
    """Stock ReadDocument reader for ``.json``: skips the structured suffix by default."""

    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".json",)


@cache
def _json_claiming_read_document() -> type[ReadDocument]:
    """ReadDocument reader declaring ``.json`` as a document suffix, so it must claim ``.json`` files."""

    class RodJsonClaimingReadDocument(ReadDocument):
        READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
            "document_suffixes": PropertySpec(
                "Structured suffixes this document reader owns; declared non-empty to claim .json.",
                default=frozenset({".json"}),
            ),
        }

        @classmethod
        def suffix(cls) -> tuple[str, ...]:
            return (".json",)

    return RodJsonClaimingReadDocument


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


class TestReadDocumentDeclarations:
    """ReadDocument declares the same two keys, and its concrete readers inherit them."""

    def test_declares_exactly_its_match_time_keys(self) -> None:
        assert ReadDocument.declared_reader_option_keys() == {"document_suffixes", "data_access_handle", _RESERVED_KEY}

    def test_declared_values_are_property_specs(self) -> None:
        assert all(isinstance(spec, PropertySpec) for spec in ReadDocument.reader_option_specs().values())

    def test_declared_defaults(self) -> None:
        specs = ReadDocument.reader_option_specs()
        assert specs["document_suffixes"].default == frozenset()
        assert specs["data_access_handle"].default is None
        assert ReadDocument.reader_option_default("document_suffixes") == frozenset()
        assert ReadDocument.reader_option_default("data_access_handle") is None

    def test_markdown_reader_inherits_without_redeclaring(self) -> None:
        assert "READER_OPTIONS" not in MarkdownDocumentReader.__dict__
        assert MarkdownDocumentReader.declared_reader_option_keys() == ReadDocument.declared_reader_option_keys()
        assert MarkdownDocumentReader.reader_option_default("document_suffixes") == frozenset()


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


class TestEveryOptionKeyReadIsDeclared:
    """Observed match-time reads are a subset of the declared keys, per reader family."""

    def test_read_file_group_reads_only_declared_keys(self, json_path: str) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        assert JsonFG.match_feature_group_criteria("rod_value", options, data_access)
        read = set(options.read_keys) - {JsonFG.get_class_name()}
        assert read == {"document_suffixes", "data_access_handle"}
        assert read <= ReadFileFG.declared_option_keys()

    def test_read_document_reads_only_declared_keys(self, csv_path: str) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(files={"rod_rows": csv_path})

        assert ReadDocument.match_subclass_data_access(data_access, ["content"], options) is None
        assert set(options.read_keys) == {"document_suffixes", "data_access_handle"}
        assert set(options.read_keys) <= ReadDocument.declared_reader_option_keys()

    def test_read_db_group_reads_only_declared_keys(self, tmp_path: Path) -> None:
        options = _RodRecordingOptions()
        data_access = DataAccessCollection(credentials=[{"sqlite": str(tmp_path / "rod.sqlite")}])

        assert not SqliteFG.match_feature_group_criteria("rod_any", options, data_access)
        read = set(options.read_keys) - {SqliteFG.get_class_name()}
        assert read == {"data_access_handle"}
        assert read <= ReadDBFG.declared_option_keys()


class TestDeclaredDefaultIsLoadBearing:
    """The document_suffixes fallback comes from the declaration, not a hard-coded frozenset()."""

    def test_stock_read_document_declines_a_json_file(self, json_path: str) -> None:
        """Control: with the stock default, ``.json`` stays a structured suffix ReadDocument skips."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        assert _RodStockJsonReadDocument.match_subclass_data_access(data_access, ["content"], Options()) is None

    def test_declared_default_makes_read_document_claim_json(self, json_path: str) -> None:
        """The declared ``frozenset({".json"})`` default claims ``.json`` with no option set."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        matched = _json_claiming_read_document().match_subclass_data_access(data_access, ["content"], Options())
        assert matched == json_path

    def test_explicit_option_still_overrides_the_read_document_default(self, json_path: str) -> None:
        """A user-set ``document_suffixes`` wins over the declared default."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})
        options = Options({"document_suffixes": frozenset({".json"})})

        matched = _RodStockJsonReadDocument.match_subclass_data_access(data_access, ["content"], options)
        assert matched == json_path


class TestAnExplicitEmptyOptionBeatsTheDeclaredDefault:
    """Presence, not truthiness: an explicit ``frozenset()`` turns the declared option OFF."""

    def test_explicit_empty_makes_read_document_skip_json_again(self, json_path: str) -> None:
        """The declaring document reader claims ``.json`` by default; an explicit empty set undoes that."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})
        options = Options({"document_suffixes": frozenset()})

        matched = _json_claiming_read_document().match_subclass_data_access(data_access, ["content"], options)

        assert matched is None

    def test_read_document_still_claims_without_the_option(self, json_path: str) -> None:
        """Control for the pair above: absent means the declared default applies."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})

        matched = _json_claiming_read_document().match_subclass_data_access(data_access, ["content"], Options())

        assert matched == json_path

    def test_explicit_none_reads_as_absent_for_read_document(self, json_path: str) -> None:
        """An explicit ``None`` is absence here too."""
        data_access = DataAccessCollection(files={"rod_payload": json_path})
        options = Options({"document_suffixes": None})

        matched = _json_claiming_read_document().match_subclass_data_access(data_access, ["content"], options)

        assert matched == json_path

    def test_read_document_reads_the_option_only_for_a_data_access_collection(self, json_path: str) -> None:
        """Documented asymmetry: the bare-path branch never consults ``document_suffixes`` at all.

        ``ReadDocument.match_subclass_data_access`` reads the key inside its ``DataAccessCollection``
        branch only, so a resolved path claims the file whatever the option says. Pinned so the
        presence read is not mistaken for a behaviour change on this branch.
        """
        claimed_with_option = _json_claiming_read_document().match_subclass_data_access(
            json_path, ["content"], Options({"document_suffixes": frozenset()})
        )
        claimed_without_option = _json_claiming_read_document().match_subclass_data_access(
            json_path, ["content"], Options()
        )

        assert claimed_with_option == json_path
        assert claimed_without_option == json_path


class TestLocalReadersStayOutOfDiscovery:
    """None of the readers defined here can hijack reader selection elsewhere."""

    def test_no_local_reader_is_a_final_reader(self) -> None:
        for reader in (
            _RodStockJsonReadDocument,
            _json_claiming_read_document(),
        ):
            assert reader.is_final_reader() is False


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
