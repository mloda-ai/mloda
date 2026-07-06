"""RED tests: an empty CSV must not crash header discovery or poison folder matching.

Target behavior:

  * ``CsvReader.get_column_names(<zero-byte file>)`` returns ``[]`` instead of raising
    ``StopIteration`` (there is simply no header to read).
  * A whitespace/newline-only file returns without raising.
  * File matching over several candidates skips an empty CSV (its ``validate_columns``
    returns ``False``) and resolves to the valid CSV, instead of the empty file poisoning
    discovery with a ``StopIteration``.

Fails today: ``get_column_names`` calls ``next(csv.reader(f))`` with no fallback, so a
zero-byte file raises ``StopIteration``, which propagates out of ``validate_columns`` and
aborts folder/file matching.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


class TestGetColumnNamesEmptyFile:
    def test_zero_byte_file_returns_empty_list(self, tmp_path: Path) -> None:
        """A zero-byte CSV has no header: return ``[]`` (today it raises StopIteration)."""
        empty = tmp_path / "empty.csv"
        empty.write_bytes(b"")

        assert list(CsvReader.get_column_names(str(empty))) == []

    def test_whitespace_only_file_does_not_raise(self, tmp_path: Path) -> None:
        """A newline-only file must be handled without raising StopIteration."""
        blank = tmp_path / "blank.csv"
        blank.write_text("\n", encoding="utf-8")

        # Must not raise; the exact value is unimportant, only that discovery survives.
        CsvReader.get_column_names(str(blank))


class TestEmptyFileDoesNotPoisonMatching:
    def test_matching_skips_empty_csv_and_resolves_valid_file(self, tmp_path: Path) -> None:
        """An empty CSV visited before the valid CSV must not abort matching.

        The empty file is listed FIRST so it is definitely inspected first; the deterministic
        RED reason today is a ``StopIteration`` from ``get_column_names`` on the empty file.
        After the fix, ``validate_columns`` on the empty file returns ``False`` (its columns
        are ``[]``) and matching continues to the valid CSV.
        """
        empty = tmp_path / "empty.csv"
        empty.write_bytes(b"")

        valid = tmp_path / "data.csv"
        with open(valid, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b"])
            writer.writerow(["1", "2"])

        matched = CsvReader.match_read_file_data_access([str(empty), str(valid)], ["a", "b"])

        assert matched == str(valid)
