"""FeatherReader and OrcReader override get_column_names, so matching opens the real
file and checks real columns: a plain name matches only when the column genuinely
exists, and a chain/column-separated name is validated (not blindly declined). This
needs a real pyarrow install and a real file on disk, not a fake nonexistent path.
"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.orc as pyarrow_orc
import pytest

import mloda_plugins.feature_group.input_data.read_files.feather as feather_module
import mloda_plugins.feature_group.input_data.read_files.json as json_module
import mloda_plugins.feature_group.input_data.read_files.orc as orc_module
import mloda_plugins.feature_group.input_data.read_files.parquet as parquet_module
from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.provider import CHAIN_SEPARATOR
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.json import JsonReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader
from mloda_plugins.feature_group.input_data.read_files.parquet import ParquetReader


@pytest.fixture()
def rejection_window() -> Iterator[dict[str, MatchRejection]]:
    """Open a recording window around one direct matcher call, mirroring the engine's per-candidate window."""
    window: dict[str, MatchRejection] = {}
    token = MATCH_REJECTION_REASONS.set(window)
    yield window
    MATCH_REJECTION_REASONS.reset(token)


def _write_feather(directory: Path, columns: list[str]) -> str:
    table = pa.Table.from_pydict({name: [1, 2, 3] for name in columns})
    file_path = str(directory / "test.feather")
    with pa.OSFile(file_path, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return file_path


def _write_orc(directory: Path, columns: list[str]) -> str:
    table = pa.Table.from_pydict({name: [1, 2, 3] for name in columns})
    file_path = str(directory / "test.orc")
    with pa.OSFile(file_path, "wb") as f:
        pyarrow_orc.write_table(table, f)
    return file_path


class TestShippedUnvalidatedReadersDeclineChainSeparatedNames:
    def test_feather_reader_declines_chain_separated_name(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        real_path = _write_feather(tmp_path, ["a", "b"])

        result = FeatherReader.match_read_file_data_access([real_path], [f"a{CHAIN_SEPARATOR}b"])

        assert result is None
        stored = rejection_window[FeatherReader.get_class_name()]
        assert "lacks the column(s)" in stored.reason
        assert f"a{CHAIN_SEPARATOR}b" in stored.reason

    def test_orc_reader_declines_chain_separated_name(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        real_path = _write_orc(tmp_path, ["a", "b"])

        result = OrcReader.match_read_file_data_access([real_path], [f"a{CHAIN_SEPARATOR}b"])

        assert result is None
        stored = rejection_window[OrcReader.get_class_name()]
        assert "lacks the column(s)" in stored.reason
        assert f"a{CHAIN_SEPARATOR}b" in stored.reason

    def test_feather_reader_matches_real_plain_column(self, tmp_path: Path) -> None:
        real_path = _write_feather(tmp_path, ["a", "b"])

        assert FeatherReader.match_read_file_data_access([real_path], ["a"]) == real_path

    def test_orc_reader_matches_real_plain_column(self, tmp_path: Path) -> None:
        real_path = _write_orc(tmp_path, ["a", "b"])

        assert OrcReader.match_read_file_data_access([real_path], ["a"]) == real_path


class TestFeatherOrcPyarrowAbsenceGuard:
    """get_column_names/describe_columns run during matching, where an aborting exception other
    than NotImplementedError takes down every reader sharing a DataAccessCollection; both must
    raise NotImplementedError, not AttributeError, when their pyarrow submodule is absent."""

    def test_feather_reader_raises_not_implemented_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(feather_module, "pyarrow_ipc", None)
        missing_path = str(tmp_path / "absent.feather")

        with pytest.raises(NotImplementedError):
            FeatherReader.get_column_names(missing_path)
        with pytest.raises(NotImplementedError):
            FeatherReader.describe_columns(missing_path)

    def test_orc_reader_raises_not_implemented_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(orc_module, "pyarrow_orc", None)
        missing_path = str(tmp_path / "absent.orc")

        with pytest.raises(NotImplementedError):
            OrcReader.get_column_names(missing_path)
        with pytest.raises(NotImplementedError):
            OrcReader.describe_columns(missing_path)


class TestJsonParquetPyarrowAbsenceGuard:
    """Same contract as TestFeatherOrcPyarrowAbsenceGuard, for the two other pyarrow-conditional
    file readers: JsonReader and ParquetReader, covering both their pre-existing
    get_column_names and their new describe_columns."""

    def test_json_reader_raises_not_implemented_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(json_module, "pyarrow_json", None)
        missing_path = str(tmp_path / "absent.json")

        with pytest.raises(NotImplementedError):
            JsonReader.get_column_names(missing_path)
        with pytest.raises(NotImplementedError):
            JsonReader.describe_columns(missing_path)

    def test_parquet_reader_raises_not_implemented_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(parquet_module, "pyarrow_parquet", None)
        missing_path = str(tmp_path / "absent.parquet")

        with pytest.raises(NotImplementedError):
            ParquetReader.get_column_names(missing_path)
        with pytest.raises(NotImplementedError):
            ParquetReader.describe_columns(missing_path)
