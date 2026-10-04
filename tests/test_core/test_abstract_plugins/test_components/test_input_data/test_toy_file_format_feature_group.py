"""ReadFileFG through the stdlib ``.toyfmtfile`` toy: shared file contract, framework loads, sample hook, G6."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

import mloda.provider as provider
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.claim_route import ClaimRoute, NamePolicy
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.provider import FormatFeatureGroup, ReadFileFG
from mloda.user import DataAccessCollection, Options
from tests.mixins.compute_frameworks.framework_adapter_mixins import (
    FileLoadsIntoFrameworkMixin,
    PyArrowTableAdapter,
    PythonDictAdapter,
)
from tests.mixins.reader_feature_groups.file_format_feature_group_test_mixin import FileFormatFeatureGroupTestMixin
from tests.test_core.test_abstract_plugins.test_components.test_input_data.toy_file_format_group import (
    SAMPLE_SIZE,
    TOY_SUFFIX,
    ToyFileFormatFG,
    toy_match,
    write_toy_file,
)


class TestToyFileFormatFeatureGroup(FileFormatFeatureGroupTestMixin):
    feature_group_class = ToyFileFormatFG
    present_column = "toyfmt_filecol"
    missing_column = "toyfmt_filemissing"

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class TestToyLoadsIntoPyArrowTable(PyArrowTableAdapter, FileLoadsIntoFrameworkMixin):
    file_group = ToyFileFormatFG

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class TestToyLoadsIntoPythonDict(PythonDictAdapter, FileLoadsIntoFrameworkMixin):
    file_group = ToyFileFormatFG

    def write_file(self, path: Path, columns: dict[str, list[Any]]) -> None:
        write_toy_file(path, columns)


class _Features:
    def get_all_names(self) -> set[str]:
        return {"toyfmt_b", "toyfmt_a"}


class TestReadFileFGBase:
    def test_is_an_abstract_format_group_exported_from_provider(self) -> None:
        assert provider.ReadFileFG is ReadFileFG
        assert "ReadFileFG" in provider.__all__
        assert issubclass(ReadFileFG, FormatFeatureGroup)
        assert inspect.isabstract(ReadFileFG)
        assert {"suffixes", "column_names"} <= set(ReadFileFG.__abstractmethods__)
        assert not {"find_sources", "load_neutral"} & set(ReadFileFG.__abstractmethods__)
        assert not inspect.isabstract(ToyFileFormatFG)

    def test_declares_checked_searched_file_and_folder_routes(self) -> None:
        assert ReadFileFG.CLAIM_ROUTES == (
            ClaimRoute("file", NamePolicy.CHECKED, True),
            ClaimRoute("folder", NamePolicy.CHECKED, True),
        )

    def test_abstract_base_never_claims(self, tmp_path: Path) -> None:
        path = tmp_path / f"base{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_base_col": [1]})
        dac = DataAccessCollection(files={"h": str(path)})
        assert ReadFileFG.match_feature_group_criteria("toyfmt_base_col", Options(), dac) is False

    def test_property_mapping_declares_the_two_reader_options(self) -> None:
        handle = ReadFileFG.PROPERTY_MAPPING["data_access_handle"]
        assert handle.default is None
        assert handle.strict_validation is True
        assert handle.context is True
        assert handle.element_validator is not None
        assert handle.element_validator("a_handle")
        assert not handle.element_validator(5)
        assert handle.match_guard is not None
        assert handle.match_guard("a_handle")
        assert not handle.match_guard(["a_handle"])
        suffixes = ReadFileFG.PROPERTY_MAPPING["document_suffixes"]
        assert suffixes.default is None
        assert suffixes.context is True

    def test_file_format_is_the_first_suffix_without_dot_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert ToyFileFormatFG.file_format() == "toyfmtfile"
        monkeypatch.setattr(ToyFileFormatFG, "suffixes", classmethod(lambda cls: (".ToyFmtFile", ".other")))
        assert ToyFileFormatFG.file_format() == "toyfmtfile"

    def test_sample_listing_defaults_to_none_and_the_sample_size_is_an_int(self) -> None:
        assert ReadFileFG.sample_column_names("toyfmt_unused") is None
        assert isinstance(ReadFileFG.SAMPLE_SIZE_BYTES, int)

    def test_load_neutral_is_a_file_source_of_the_sorted_features(self, tmp_path: Path) -> None:
        path = tmp_path / f"neutral{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_a": [1], "toyfmt_b": [2]})
        neutral = ToyFileFormatFG.load_neutral(toy_match(path), _Features())
        assert neutral == FileSource(path=str(path), format="toyfmtfile", columns=("toyfmt_a", "toyfmt_b"))


class TestSampleListing:
    def _small(self, tmp_path: Path) -> Path:
        path = tmp_path / f"small{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_s_a": [1], "toyfmt_s_b": [2]})
        assert path.stat().st_size <= SAMPLE_SIZE
        return path

    def _wide(self, tmp_path: Path) -> Path:
        path = tmp_path / f"wide{TOY_SUFFIX}"
        write_toy_file(path, {f"toyfmt_wide_{i:02d}": [i] for i in range(20)})
        assert path.stat().st_size > SAMPLE_SIZE
        return path

    def _reads(self, path: Path) -> tuple[list[str], list[str]]:
        names = [p for p in ToyFileFormatFG.COLUMN_NAMES_CALLS if p == str(path)]
        samples = [p for p in ToyFileFormatFG.SAMPLE_CALLS if p == str(path)]
        return names, samples

    def test_a_sample_hit_needs_no_full_listing(self, tmp_path: Path) -> None:
        path = self._wide(tmp_path)
        assert ToyFileFormatFG.has_column(toy_match(path), "toyfmt_wide_00") is True
        names, samples = self._reads(path)
        assert samples and not names

    def test_a_sample_miss_on_a_file_larger_than_the_sample_falls_back_to_the_full_listing_once(
        self, tmp_path: Path
    ) -> None:
        path = self._wide(tmp_path)
        match = toy_match(path)
        with run_match_cache():
            assert ToyFileFormatFG.has_column(match, "toyfmt_wide_19") is True
            assert ToyFileFormatFG.has_column(match, "toyfmt_wide_18") is True
        names, _ = self._reads(path)
        assert len(names) == 1

    def test_a_sample_miss_on_a_file_within_the_sample_size_is_final(self, tmp_path: Path) -> None:
        path = self._small(tmp_path)
        assert ToyFileFormatFG.has_column(toy_match(path), "toyfmt_s_absent") is False
        names, samples = self._reads(path)
        assert samples
        assert not names

    def test_the_sample_listing_is_cached_separately_within_a_run(self, tmp_path: Path) -> None:
        path = self._small(tmp_path)
        match = toy_match(path)
        with run_match_cache():
            assert ToyFileFormatFG.has_column(match, "toyfmt_s_a") is True
            assert ToyFileFormatFG.has_column(match, "toyfmt_s_b") is True
            assert ToyFileFormatFG.columns(match) is not None
            assert ToyFileFormatFG.columns(match) is not None
        names, samples = self._reads(path)
        assert len(samples) == 1
        assert len(names) == 1


class TestPinnedFileOwnedBySomeFileGroup:
    def _dac(self, path: Path, column: str) -> DataAccessCollection:
        return DataAccessCollection(files={"pinned": str(path)}, column_to_file={column: "pinned"})

    def _unowned_pin_reasons(self, window: dict[str, MatchRejection]) -> list[str]:
        return [r.reason for r in window.values() if "no registered reader owns" in r.reason]

    def test_a_pin_to_a_file_group_suffix_records_no_unowned_pin(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        path = tmp_path / f"pinned{TOY_SUFFIX}"
        write_toy_file(path, {"toyfmt_g6_col": [1]})
        BaseInputData._record_unowned_pin(self._dac(path, "toyfmt_g6_col"), ["toyfmt_g6_col"])
        assert self._unowned_pin_reasons(rejection_window) == []

    def test_a_pin_to_a_suffix_nobody_owns_still_records_it(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        path = tmp_path / "pinned.toyfmtnobody"
        path.write_text("toyfmt_g6_other\n1\n")
        BaseInputData._record_unowned_pin(self._dac(path, "toyfmt_g6_other"), ["toyfmt_g6_other"])
        assert len(self._unowned_pin_reasons(rejection_window)) == 1
