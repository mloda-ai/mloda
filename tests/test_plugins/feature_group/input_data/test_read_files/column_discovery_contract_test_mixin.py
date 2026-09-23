"""Shared describe_columns/get_column_names/count_rows contract for pyarrow-backed ReadFile readers.

Without its pyarrow submodule, a reader raises ImportError naming mloda[pyarrow] from
get_column_names, describe_columns, load_data, and count_rows. Unprefixed so pytest skips it standalone.
"""

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.provider import CHAIN_SEPARATOR, FeatureSet
from mloda.user import DataAccessCollection, DataType, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.input_data.read_file import ReadFile

PHYSICAL_COLUMNS = ["c1", "a1", "b1"]


class ColumnDiscoveryContractTestMixin:
    """Shared get_column_names/describe_columns/count_rows contract for one ReadFile reader."""

    reader_cls: type[ReadFile]
    dependency_module: str
    expected_row_count: int | None = 3
    row_count_module: str | None = None

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        """Return a file path with columns c1, a1, b1 (all INT64). Override per reader."""
        raise NotImplementedError

    @pytest.fixture()
    def rejection_window(self) -> Iterator[dict[str, MatchRejection]]:
        """Open a recording window around one direct matcher call, mirroring the engine's per-candidate window."""
        window: dict[str, MatchRejection] = {}
        token = MATCH_REJECTION_REASONS.set(window)
        yield window
        MATCH_REJECTION_REASONS.reset(token)

    def test_describe_columns_returns_real_types(self, data_file: str) -> None:
        described = self.reader_cls.describe_columns(data_file)
        assert described == {name: DataType.INT64 for name in PHYSICAL_COLUMNS}

    def test_get_column_names_matches_describe_columns_keys(self, data_file: str) -> None:
        names = self.reader_cls.get_column_names(data_file)
        described = self.reader_cls.describe_columns(data_file)
        assert set(names) == set(described.keys())

    def test_describe_columns_accepts_path(self, data_file: str) -> None:
        from_str = self.reader_cls.describe_columns(data_file)
        from_path = self.reader_cls.describe_columns(Path(data_file))
        assert from_path == from_str

    def test_describe_columns_rejects_non_path_data_access(self) -> None:
        with pytest.raises(ValueError):
            self.reader_cls.describe_columns(DataAccessCollection(files={"dummy.csv"}))

    def test_raises_import_error_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Guard fires before any filesystem access, so a nonexistent path also raises."""
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), self.dependency_module, None)
        absent = str(tmp_path / "absent")
        features = FeatureSet()
        features.add(Feature("a1"))

        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.reader_cls.get_column_names(absent)
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.reader_cls.describe_columns(absent)
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.reader_cls.load_data(absent, features)

    def test_declines_chain_separated_name_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Without the optional dependency, a plain name still matches; a chain-separated name declines."""
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), self.dependency_module, None)
        absent = str(tmp_path / f"absent{self.reader_cls.suffix()[0]}")

        assert self.reader_cls.match_read_file_data_access([absent], ["a1"]) == absent

        chained = f"a1{CHAIN_SEPARATOR}b1"
        result = self.reader_cls.match_read_file_data_access([absent], [chained])

        assert result is None
        stored = rejection_window[self.reader_cls.get_class_name()]
        assert "cannot enumerate" in stored.reason
        assert "get_column_names" in stored.reason
        assert chained in stored.reason

    @pytest.mark.parametrize("compute_framework", [PythonDictFramework, PyArrowTable])
    def test_count_rows_matches_expected_row_count(
        self, data_file: str, compute_framework: type[ComputeFramework]
    ) -> None:
        for candidate in (data_file, Path(data_file)):
            assert self.reader_cls.count_rows(candidate, compute_framework) == self.expected_row_count

    def test_count_rows_rejects_non_path_data_access(self) -> None:
        non_path = DataAccessCollection(files={"dummy.csv"})
        if self.expected_row_count is None:
            assert self.reader_cls.count_rows(non_path, PyArrowTable) is None
            return
        with pytest.raises(ValueError):
            self.reader_cls.count_rows(non_path, PyArrowTable)

    def test_count_rows_raises_oserror_for_absent_file(self, tmp_path: Path) -> None:
        absent = str(tmp_path / "absent")
        if self.expected_row_count is None:
            assert self.reader_cls.count_rows(absent, PyArrowTable) is None
            return
        with pytest.raises(OSError):
            self.reader_cls.count_rows(absent, PyArrowTable)

    def test_count_rows_without_optional_dependency(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Guard fires before any filesystem access, so a nonexistent path also raises."""
        module = self.row_count_module or self.dependency_module
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), module, None)
        absent = str(tmp_path / "absent")

        if self.expected_row_count is None:
            assert self.reader_cls.count_rows(absent, PyArrowTable) is None
            return
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.reader_cls.count_rows(absent, PyArrowTable)

    def test_count_rows_raises_oserror_for_a_directory_data_access(self, data_file: str, tmp_path: Path) -> None:
        """A directory is not this reader's file; it must not be silently summed as a dataset."""
        directory = tmp_path / f"dir{self.reader_cls.suffix()[0]}"
        directory.mkdir()
        shutil.copyfile(data_file, directory / "inner.arrow")

        if self.expected_row_count is None:
            assert self.reader_cls.count_rows(str(directory), PyArrowTable) is None
            return
        with pytest.raises(OSError):
            self.reader_cls.count_rows(str(directory), PyArrowTable)

    def test_count_rows_reports_none_for_a_load_data_overriding_subclass(self, data_file: str) -> None:
        base = self.reader_cls

        class _CountRowsProbeReader(base):  # type: ignore[misc,valid-type]
            @classmethod
            def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
                return super().load_data(data_access, features)

            @classmethod
            def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
                return None

        assert _CountRowsProbeReader.count_rows(data_file, PyArrowTable) is None
