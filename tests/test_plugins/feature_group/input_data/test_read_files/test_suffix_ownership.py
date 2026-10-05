"""Suffix ownership: JsonFG owns .json by default, JsonDocumentFG only when the feature lists it in document_suffixes."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.format_file_writers import write_csv, write_json
from tests.mixins.reader_feature_groups.lazy_format_group import load_document_group, load_group

SUFFIX_COLUMN = "suffixown_id"
DOCUMENT_NAME = "JsonDocumentFG"


def _listed() -> Options:
    return Options(context={"document_suffixes": frozenset({".json"})})


def _json_group() -> Any:
    return load_group("json_fg", "JsonFG")


def _json_document_group() -> Any:
    return load_document_group("json_document_fg", "JsonDocumentFG")


def _document_options(*suffixes: str, pointer: str | None = None) -> Options:
    group = {DOCUMENT_NAME: pointer} if pointer is not None else None
    return Options(group, context={"document_suffixes": frozenset(suffixes)} if suffixes else None)


@pytest.fixture
def json_file(tmp_path: Path) -> str:
    path = tmp_path / "data.json"
    write_json(path, {SUFFIX_COLUMN: [1, 2], "suffixown_value": [3, 4]})
    return str(path)


class TestDefaultSuffixOwnership:
    """Without document_suffixes, JsonFG owns .json and JsonDocumentFG declines it."""

    def test_json_fg_matches_json_by_default(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert _json_group().match_feature_group_criteria(SUFFIX_COLUMN, Options(), dac)

    def test_json_document_fg_declines_json_by_default(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert not _json_document_group().match_feature_group_criteria(DOCUMENT_NAME, Options(), dac)

    def test_json_document_fg_declines_a_json_in_a_folder_by_default(self, tmp_path: Path) -> None:
        write_json(tmp_path / "data.json", {SUFFIX_COLUMN: [1]})
        dac = DataAccessCollection(folders={str(tmp_path)})
        assert not _json_document_group().match_feature_group_criteria(DOCUMENT_NAME, Options(), dac)

    def test_both_groups_read_the_one_shared_json_suffix_constant(self) -> None:
        suffixes = importlib.import_module("mloda_plugins.feature_group.input_data.file_suffixes")
        assert _json_document_group().suffixes() == _json_group().suffixes() == suffixes.JSON_SUFFIXES


class TestDocumentSuffixesOverride:
    """With .json listed, JsonDocumentFG claims it and JsonFG auto-excludes it."""

    def test_json_document_fg_matches_json_with_the_override(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert _json_document_group().match_feature_group_criteria(DOCUMENT_NAME, _listed(), dac)

    def test_json_document_fg_matches_a_json_in_a_folder_with_the_override(self, tmp_path: Path) -> None:
        write_json(tmp_path / "data.json", {SUFFIX_COLUMN: [1]})
        dac = DataAccessCollection(folders={str(tmp_path)})
        assert _json_document_group().match_feature_group_criteria(DOCUMENT_NAME, _listed(), dac)

    def test_json_document_fg_matches_a_pointer_with_the_override(self, json_file: str) -> None:
        options = _document_options(".json", pointer=json_file)
        assert _json_document_group().match_feature_group_criteria(DOCUMENT_NAME, options, None)

    def test_json_fg_excludes_json_with_the_override(self, json_file: str) -> None:
        dac = DataAccessCollection(files={json_file})
        assert not _json_group().match_feature_group_criteria(SUFFIX_COLUMN, _listed(), dac)

    def test_the_override_is_suffix_specific(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "data.csv"
        write_csv(csv_path, {"suffixown_csv_col": [1]})
        dac = DataAccessCollection(files={str(csv_path)})

        assert load_group("csv_fg", "CsvFG").match_feature_group_criteria("suffixown_csv_col", _listed(), dac)


class TestAJsonRequestNeverMatchesBothGroups:
    """A JSON file holding a column named JsonDocumentFG is claimed by exactly one of the two groups."""

    @pytest.mark.parametrize("file_suffix", [".json", ".JSON"])
    @pytest.mark.parametrize("listed", [(), (".json",), (".JSON",), (".json", ".JSON")])
    def test_exactly_one_group_claims_whatever_the_listing(
        self, tmp_path: Path, file_suffix: str, listed: tuple[str, ...]
    ) -> None:
        path = tmp_path / f"collide{file_suffix}"
        write_json(path, {DOCUMENT_NAME: [1, 2]})
        mapping: Any = {_json_group(): {PyArrowTable}, _json_document_group(): {PyArrowTable}}
        feature = Feature(DOCUMENT_NAME, _document_options(*listed))

        result = IdentifyFeatureGroupClass.evaluate(feature, mapping, None, DataAccessCollection(files={str(path)}))

        expected = _json_document_group() if file_suffix in listed else _json_group()
        assert set(result.identified) == {expected}

    def test_a_pointer_to_the_file_is_claimed_by_one_group_only(self, tmp_path: Path) -> None:
        path = tmp_path / "pointed.json"
        write_json(path, {DOCUMENT_NAME: [1]})
        mapping: Any = {_json_group(): {PyArrowTable}, _json_document_group(): {PyArrowTable}}
        for listed, expected in (((), _json_group()), ((".json",), _json_document_group())):
            options = Options(
                {"JsonFG": str(path), DOCUMENT_NAME: str(path)}, context=_document_options(*listed).context
            )
            result = IdentifyFeatureGroupClass.evaluate(Feature(DOCUMENT_NAME, options), mapping, None, None)
            assert set(result.identified) == {expected}


class TestPointersUnaffected:
    def test_json_fg_pointer_path_still_works(self, json_file: str) -> None:
        options = Options({"JsonFG": json_file})
        assert _json_group().match_feature_group_criteria(SUFFIX_COLUMN, options, None)
