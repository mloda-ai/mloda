"""FeatherReader and OrcReader override get_column_names, so matching checks real columns instead of
blindly declining a chain/column-separated name. Missing pyarrow or an unreadable file both leave a
declined match on the unpinned path; a pinned unreadable file propagates instead of falling back to
a sibling file."""

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.orc as pyarrow_orc
import pytest

from mloda.core.abstract_plugins.components.match_rejection import MATCH_REJECTION_REASONS, MatchRejection
from mloda.provider import CHAIN_SEPARATOR
from mloda.user import DataAccessCollection, Options
from mloda_plugins.feature_group.input_data.read_files.feather import FeatherReader
from mloda_plugins.feature_group.input_data.read_files.orc import OrcReader


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


class TestFeatherOrcValidateRealColumns:
    """With pyarrow present and a real file, matching checks the file's actual columns."""

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


class TestFeatherOrcWithoutPyarrow:
    """Without pyarrow, get_column_names cannot enumerate columns at all; the unpinned path declines."""

    def test_feather_reader_declines_chain_separated_name_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), "pyarrow.ipc", None)

        result = FeatherReader.match_read_file_data_access(["dummy.feather"], [f"a{CHAIN_SEPARATOR}b"])

        assert result is None
        stored = rejection_window[FeatherReader.get_class_name()]
        assert "cannot enumerate" in stored.reason
        assert "get_column_names" in stored.reason
        assert f"a{CHAIN_SEPARATOR}b" in stored.reason

    def test_orc_reader_declines_chain_separated_name_without_pyarrow(
        self, monkeypatch: pytest.MonkeyPatch, rejection_window: dict[str, MatchRejection]
    ) -> None:
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), "pyarrow.orc", None)

        result = OrcReader.match_read_file_data_access(["dummy.orc"], [f"a{CHAIN_SEPARATOR}b"])

        assert result is None
        stored = rejection_window[OrcReader.get_class_name()]
        assert "cannot enumerate" in stored.reason
        assert "get_column_names" in stored.reason
        assert f"a{CHAIN_SEPARATOR}b" in stored.reason

    def test_feather_reader_matches_plain_name_without_pyarrow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), "pyarrow.ipc", None)

        result = FeatherReader.match_read_file_data_access(["dummy.feather"], ["a"])

        assert result == "dummy.feather"

    def test_orc_reader_matches_plain_name_without_pyarrow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(cast(dict[str, Any], sys.modules), "pyarrow.orc", None)

        result = OrcReader.match_read_file_data_access(["dummy.orc"], ["a"])

        assert result == "dummy.orc"


class TestFeatherOrcUnreadableFile:
    """A real file get_column_names cannot read (corrupt, truncated, or missing) declines on the
    unpinned path, but propagates instead of falling back to a sibling when the file was pinned."""

    def test_feather_reader_declines_corrupt_file(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        bad_path = tmp_path / "corrupt.feather"
        bad_path.write_bytes(b"not a real feather file")

        result = FeatherReader.match_read_file_data_access([str(bad_path)], ["a"])

        assert result is None
        stored = rejection_window[FeatherReader.get_class_name()]
        assert "could not read" in stored.reason

    def test_orc_reader_declines_corrupt_file(
        self, tmp_path: Path, rejection_window: dict[str, MatchRejection]
    ) -> None:
        bad_path = tmp_path / "corrupt.orc"
        bad_path.write_bytes(b"")

        result = OrcReader.match_read_file_data_access([str(bad_path)], ["a"])

        assert result is None
        stored = rejection_window[OrcReader.get_class_name()]
        assert "could not read" in stored.reason

    def test_feather_reader_declines_nonexistent_path_plain_name(self) -> None:
        assert FeatherReader.match_read_file_data_access(["dummy.feather"], ["a"]) is None

    def test_orc_reader_declines_nonexistent_path_plain_name(self) -> None:
        assert OrcReader.match_read_file_data_access(["dummy.orc"], ["a"]) is None

    def test_pinned_corrupt_feather_file_propagates_instead_of_falling_back_to_sibling(self, tmp_path: Path) -> None:
        """A pinned unreadable file must not silently fall through to an unpinned sibling file."""
        good = _write_feather(tmp_path, ["a"])
        corrupt = tmp_path / "corrupt.feather"
        corrupt.write_bytes(b"not a real feather file")
        dac = DataAccessCollection(
            files={"pinned": str(corrupt), "sibling": good},
            column_to_file={"a": "pinned"},
        )

        with pytest.raises(ValueError, match="Not an Arrow file"):
            FeatherReader.match_subclass_data_access(dac, ["a"], Options())
