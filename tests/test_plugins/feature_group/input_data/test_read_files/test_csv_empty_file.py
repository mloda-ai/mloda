"""An empty CSV does not crash header discovery or poison folder matching.

Contract:

  * ``CsvFG.column_names(<zero-byte file>)`` returns ``[]`` instead of raising (there is no header to read).
  * A whitespace/newline-only file returns without raising.
  * A folder holding an empty CSV next to a valid CSV still resolves to the valid CSV.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from mloda.core.abstract_plugins.components.input_data.claim_route import SourceMatch
from mloda.core.prepare.identify_feature_group import IdentifyFeatureGroupClass
from mloda.user import DataAccessCollection, Feature
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.lazy_format_group import load_group


class TestColumnNamesEmptyFile:
    def test_zero_byte_file_returns_empty_list(self, tmp_path: Path) -> None:
        """A zero-byte CSV has no header: return ``[]``."""
        empty = tmp_path / "empty.csv"
        empty.write_bytes(b"")

        assert list(load_group("csv_fg", "CsvFG").column_names(str(empty))) == []

    def test_whitespace_only_file_does_not_raise(self, tmp_path: Path) -> None:
        """A newline-only file must be handled without raising."""
        blank = tmp_path / "blank.csv"
        blank.write_text("\n", encoding="utf-8")

        # Must not raise; the exact value is unimportant, only that discovery survives.
        load_group("csv_fg", "CsvFG").column_names(str(blank))


class TestEmptyFileDoesNotPoisonMatching:
    def test_a_folder_with_an_empty_csv_and_a_valid_csv_resolves_the_valid_one(self, tmp_path: Path) -> None:
        """The empty file sorts first and has no columns; matching goes on to the valid CSV."""
        csv_group = load_group("csv_fg", "CsvFG")
        folder = tmp_path / "empty_and_valid"
        folder.mkdir()
        (folder / "a_empty.csv").write_bytes(b"")
        valid = folder / "b_data.csv"
        with open(valid, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["emptyfile_a", "emptyfile_b"])
            writer.writerow(["1", "2"])

        feature = Feature("emptyfile_a")
        result = IdentifyFeatureGroupClass.evaluate(
            feature, {csv_group: {PyArrowTable}}, None, DataAccessCollection(folders={"emptyfile_dir": str(folder)})
        )

        assert csv_group in result.identified
        pair = feature.input_data_match
        assert pair is not None
        assert pair[0] is csv_group
        match = pair[1]
        assert isinstance(match, SourceMatch)
        assert match.source == os.path.abspath(valid)
