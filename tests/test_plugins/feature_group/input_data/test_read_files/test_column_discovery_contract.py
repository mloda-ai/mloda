"""Concrete column_names/describe_columns contract runs per pyarrow-backed format group, plus JSON sampling."""

import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.input_data.match_cache import run_match_cache
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import DataAccessCollection, Feature
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.column_discovery_contract_test_mixin import (
    PHYSICAL_COLUMNS,
    ColumnDiscoveryContractTestMixin,
    match_of,
)
from tests.mixins.reader_feature_groups.lazy_format_group import lazy_group, load_group


class TestFeatherColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    group_cls = lazy_group("feather_fg", "FeatherFG")
    dependency_module = "pyarrow.ipc"
    row_count_module = "pyarrow.dataset"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.feather")
        with pa.OSFile(file_path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        return file_path


class TestOrcColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    group_cls = lazy_group("orc_fg", "OrcFG")
    dependency_module = "pyarrow.orc"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        import pyarrow.orc as pyarrow_orc

        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.orc")
        with pa.OSFile(file_path, "wb") as f:
            pyarrow_orc.write_table(table, f)
        return file_path


class TestJsonColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    group_cls = lazy_group("json_fg", "JsonFG")
    dependency_module = "pyarrow.json"
    expected_row_count = None

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = tmp_path / "test.json"
        rows = [{col: table.column(col)[i].as_py() for col in PHYSICAL_COLUMNS} for i in range(table.num_rows)]
        with open(file_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(file_path)


class TestParquetColumnDiscoveryContract(ColumnDiscoveryContractTestMixin):
    group_cls = lazy_group("parquet_fg", "ParquetFG")
    dependency_module = "pyarrow.parquet"

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        import pyarrow.parquet as pyarrow_parquet

        table = pa.Table.from_pydict({name: [1, 2, 3] for name in PHYSICAL_COLUMNS})
        file_path = str(tmp_path / "test.parquet")
        pyarrow_parquet.write_table(table, file_path)
        return file_path


EARLY = "jsample_early"
LATE = "jsample_late"


def _write_big_json(path: Path) -> None:
    """Rows of EARLY fill well past the sample; LATE first appears on the last line only."""
    with open(path, "w") as handle:
        for i in range(600):
            handle.write(json.dumps({EARLY: i, "jsample_pad": "x" * 200}) + "\n")
        handle.write(json.dumps({EARLY: -1, LATE: 1}) + "\n")


def _write_small_json(path: Path) -> None:
    with open(path, "w") as handle:
        for i in range(3):
            handle.write(json.dumps({EARLY: i}) + "\n")


class TestJsonSampling:
    """JsonFG reads the first 64 KiB first and falls back to a full parse only when that cannot decide."""

    def _spies(self, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
        group = load_group("json_fg", "JsonFG")
        full: list[str] = []
        sample: list[str] = []
        original_full = group.column_names
        original_sample = group.sample_column_names

        def column_names(klass: Any, path: str) -> Any:
            full.append(path)
            return original_full(path)

        def sample_column_names(klass: Any, path: str) -> Any:
            sample.append(path)
            return original_sample(path)

        monkeypatch.setattr(group, "column_names", classmethod(column_names))
        monkeypatch.setattr(group, "sample_column_names", classmethod(sample_column_names))
        return full, sample

    def _claims(self, path: Path, column: str) -> bool:
        group = load_group("json_fg", "JsonFG")
        dac = DataAccessCollection(files={"jsample_handle": str(path)})
        result = IdentifyFeatureGroupClass.evaluate(Feature(column), {group: {PyArrowTable}}, None, dac)
        return group in result.identified

    def test_sample_listing_holds_early_columns_only_and_the_full_listing_all(self, tmp_path: Path) -> None:
        group = load_group("json_fg", "JsonFG")
        path = tmp_path / "sample_vs_full.json"
        _write_big_json(path)
        assert os.path.getsize(path) > group.SAMPLE_SIZE_BYTES

        sample = group.sample_column_names(str(path))

        assert sample is not None
        assert EARLY in sample
        assert LATE not in sample
        assert LATE in group.column_names(str(path))

    def test_a_column_first_appearing_after_the_sample_still_claims(self, tmp_path: Path) -> None:
        path = tmp_path / "late_column.json"
        _write_big_json(path)

        assert self._claims(path, LATE)

    def test_a_late_column_lookup_falls_back_to_one_full_parse_per_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = load_group("json_fg", "JsonFG")
        path = tmp_path / "late_column_once.json"
        _write_big_json(path)
        full, _ = self._spies(monkeypatch)

        with run_match_cache():
            assert group.has_column(match_of(path), LATE) is True
            assert group.has_column(match_of(path), LATE) is True

        assert len(full) == 1

    def test_a_column_inside_the_sample_does_not_trigger_the_full_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = load_group("json_fg", "JsonFG")
        path = tmp_path / "early_column.json"
        _write_big_json(path)
        full, sample = self._spies(monkeypatch)

        assert group.has_column(match_of(path), EARLY) is True
        assert self._claims(path, EARLY)

        assert sample
        assert not full

    def test_a_missing_column_in_a_file_smaller_than_the_sample_declines_without_a_second_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        group = load_group("json_fg", "JsonFG")
        path = tmp_path / "small_file.json"
        _write_small_json(path)
        assert os.path.getsize(path) <= group.SAMPLE_SIZE_BYTES
        full, sample = self._spies(monkeypatch)

        assert group.has_column(match_of(path), "jsample_absent") is False

        assert sample
        assert not full

    def test_a_partial_sample_is_none_and_never_raises(self, tmp_path: Path) -> None:
        group = load_group("json_fg", "JsonFG")
        path = tmp_path / "partial_sample.json"
        path.write_text('{"jsample_one": 1, "jsample_two": ')

        assert group.sample_column_names(str(path)) is None
