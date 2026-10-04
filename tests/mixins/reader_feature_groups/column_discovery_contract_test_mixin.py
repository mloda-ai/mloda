"""Shared column_names/describe_columns/count_rows contract for pyarrow-backed file format groups.

Without its pyarrow submodule a group raises ImportError naming mloda[pyarrow] from column_names,
describe_columns, load_neutral and count_rows, and declines at match time with a rejection naming it.
Unprefixed so pytest skips it standalone.
"""

import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.provider import CHAIN_SEPARATOR, FeatureSet, ReadFileFG
from mloda.user import DataAccessCollection, DataType, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

PHYSICAL_COLUMNS = ["c1", "a1", "b1"]


def match_of(path: str | Path) -> SourceMatch:
    return SourceMatch(source=os.path.abspath(path), access=str(path))


class ColumnDiscoveryContractTestMixin:
    """Shared column_names/describe_columns/count_rows contract for one pyarrow-backed file format group."""

    group_cls: type[ReadFileFG]
    dependency_module: str
    expected_row_count: int | None = 3
    row_count_module: str | None = None

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        """Return a file path with columns c1, a1, b1 (all INT64). Override per format."""
        raise NotImplementedError

    def column_names(self, path: str) -> Any:
        return self.group_cls.column_names(path)

    def describe_columns(self, match: SourceMatch) -> dict[str, DataType | None]:
        return self.group_cls.describe_columns(match)

    def count_rows(self, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        return self.group_cls.count_rows(match, compute_framework)

    def test_describe_columns_returns_real_types(self, data_file: str) -> None:
        described = self.describe_columns(match_of(data_file))
        assert described == {name: DataType.INT64 for name in PHYSICAL_COLUMNS}

    def test_column_names_match_describe_columns_keys(self, data_file: str) -> None:
        names = self.column_names(data_file)
        described = self.describe_columns(match_of(data_file))
        assert set(names) == set(described.keys())

    def test_raises_import_error_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Guard fires before any filesystem access, so a nonexistent path also raises."""
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), self.dependency_module, None)
        absent = str(tmp_path / f"absent{self.group_cls.suffixes()[0]}")
        features = FeatureSet()
        features.add(Feature("a1"))

        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.column_names(absent)
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.describe_columns(match_of(absent))
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.group_cls.load_neutral(match_of(absent), features)

    def test_declines_with_a_rejection_naming_the_install_hint_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, data_file: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        """Without the dependency even a plain name declines, recording why and how to fix it."""
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), self.dependency_module, None)
        dac = DataAccessCollection(files={"nodep_handle": data_file})

        assert not self.group_cls.match_feature_group_criteria("a1", Options(), dac)

        stored = rejection_window[self.group_cls.get_class_name()]
        assert "mloda[pyarrow]" in stored.reason
        assert os.path.abspath(data_file) in stored.reason

    def test_declines_a_chain_separated_name_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, data_file: str, rejection_window: dict[str, MatchRejection]
    ) -> None:
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), self.dependency_module, None)
        dac = DataAccessCollection(files={"nodep_chain_handle": data_file})
        chained = f"a1{CHAIN_SEPARATOR}b1"

        assert not self.group_cls.match_feature_group_criteria(chained, Options(), dac)

        assert "mloda[pyarrow]" in rejection_window[self.group_cls.get_class_name()].reason

    @pytest.mark.parametrize("compute_framework", [PythonDictFramework, PyArrowTable])
    def test_count_rows_matches_expected_row_count(
        self, data_file: str, compute_framework: type[ComputeFramework]
    ) -> None:
        assert self.count_rows(match_of(data_file), compute_framework) == self.expected_row_count

    def test_count_rows_raises_oserror_for_absent_file(self, tmp_path: Path) -> None:
        absent = match_of(tmp_path / f"absent{self.group_cls.suffixes()[0]}")
        if self.expected_row_count is None:
            assert self.count_rows(absent, PyArrowTable) is None
            return
        with pytest.raises(OSError):
            self.count_rows(absent, PyArrowTable)

    def test_count_rows_without_optional_dependency(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Guard fires before any filesystem access, so a nonexistent path also raises."""
        module = self.row_count_module or self.dependency_module
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), module, None)
        absent = match_of(tmp_path / f"absent{self.group_cls.suffixes()[0]}")

        if self.expected_row_count is None:
            assert self.count_rows(absent, PyArrowTable) is None
            return
        with pytest.raises(ImportError, match=r"mloda\[pyarrow\]"):
            self.count_rows(absent, PyArrowTable)

    def test_count_rows_raises_oserror_for_a_directory_access(self, data_file: str, tmp_path: Path) -> None:
        """A directory is not this group's file; it must not be silently summed as a dataset."""
        directory = tmp_path / f"dir{self.group_cls.suffixes()[0]}"
        directory.mkdir()
        shutil.copyfile(data_file, directory / "inner.arrow")

        if self.expected_row_count is None:
            assert self.count_rows(match_of(directory), PyArrowTable) is None
            return
        with pytest.raises(OSError):
            self.count_rows(match_of(directory), PyArrowTable)

    def test_count_rows_reports_none_for_a_load_neutral_overriding_subclass(self, data_file: str) -> None:
        base = self.group_cls

        class _CountRowsProbeFG(base):  # type: ignore[misc,valid-type]
            @classmethod
            def match_feature_group_criteria(cls, *args: Any, **kwargs: Any) -> bool:
                return False

            @classmethod
            def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
                return super().load_neutral(match, features)

        assert _CountRowsProbeFG.count_rows(match_of(data_file), PyArrowTable) is None
        del _CountRowsProbeFG
        gc.collect()
