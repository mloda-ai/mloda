"""Shared describe_columns/get_column_names contract for pyarrow-backed ReadFile readers.

Without its optional pyarrow submodule, a reader must raise a bare NotImplementedError from
both methods; ReadFile.validate_columns relies on that to treat "can't confirm" as non-fatal.
Named without a ``Test`` prefix so pytest does not collect it standalone.
"""

import importlib
from pathlib import Path

import pytest

from mloda.user import DataType
from mloda_plugins.feature_group.input_data.read_file import ReadFile

PHYSICAL_COLUMNS = ["c1", "a1", "b1"]


class ColumnDiscoveryContractTestMixin:
    """Shared get_column_names/describe_columns contract for one ReadFile reader."""

    reader_cls: type[ReadFile]
    dependency_module: str
    dependency_attr: str

    @pytest.fixture
    def data_file(self, tmp_path: Path) -> str:
        """Return a file path with columns c1, a1, b1 (all INT64). Override per reader."""
        raise NotImplementedError

    def test_describe_columns_returns_real_types(self, data_file: str) -> None:
        described = self.reader_cls.describe_columns(data_file)
        assert described == {name: DataType.INT64 for name in PHYSICAL_COLUMNS}

    def test_get_column_names_matches_describe_columns_keys(self, data_file: str) -> None:
        names = self.reader_cls.get_column_names(data_file)
        described = self.reader_cls.describe_columns(data_file)
        assert set(names) == set(described.keys())

    def test_raises_not_implemented_without_optional_dependency(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Guard fires before any filesystem access, so a nonexistent path also raises."""
        module = importlib.import_module(self.dependency_module)
        monkeypatch.setattr(module, self.dependency_attr, None)
        absent = str(tmp_path / "absent")

        with pytest.raises(NotImplementedError):
            self.reader_cls.get_column_names(absent)
        with pytest.raises(NotImplementedError):
            self.reader_cls.describe_columns(absent)
