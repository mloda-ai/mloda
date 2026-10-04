"""FeatherFG and OrcFG list a file's real columns, so matching checks them: present claims, missing declines.

An unreadable file declines when unpinned (with a rejection naming it) and aborts when pinned.
"""

import os
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.match_rejection import MatchRejection
from mloda.core.prepare.identify_feature_group import resolve_or_raise
from mloda.provider import CHAIN_SEPARATOR
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from tests.mixins.reader_feature_groups.format_file_writers import write_feather, write_orc
from tests.mixins.reader_feature_groups.lazy_format_group import load_group


def _write_feather(directory: Path, columns: list[str]) -> str:
    path = directory / "test.feather"
    write_feather(path, {name: [1, 2, 3] for name in columns})
    return str(path)


def _write_orc(directory: Path, columns: list[str]) -> str:
    path = directory / "test.orc"
    write_orc(path, {name: [1, 2, 3] for name in columns})
    return str(path)


def _dac(path: str) -> DataAccessCollection:
    return DataAccessCollection(files={"fo_handle": path})


FEATHER = ("feather_fg", "FeatherFG")
ORC = ("orc_fg", "OrcFG")


class TestFeatherOrcValidateRealColumns:
    """With pyarrow present and a real file, matching checks the file's actual columns."""

    @pytest.mark.parametrize("group_ref,writer", [(FEATHER, _write_feather), (ORC, _write_orc)], ids=["feather", "orc"])
    def test_declines_a_chain_separated_name(
        self, tmp_path: Path, group_ref: tuple[str, str], writer: Any, rejection_window: dict[str, MatchRejection]
    ) -> None:
        group = load_group(*group_ref)
        real_path = writer(tmp_path, ["a", "b"])
        chained = f"a{CHAIN_SEPARATOR}b"

        assert not group.match_feature_group_criteria(chained, Options(), _dac(real_path))

        reason = rejection_window[group.get_class_name()].reason
        assert os.path.abspath(real_path) in reason
        assert chained in reason

    @pytest.mark.parametrize("group_ref,writer", [(FEATHER, _write_feather), (ORC, _write_orc)], ids=["feather", "orc"])
    def test_matches_a_real_plain_column(self, tmp_path: Path, group_ref: tuple[str, str], writer: Any) -> None:
        group = load_group(*group_ref)
        real_path = writer(tmp_path, ["a", "b"])

        assert group.match_feature_group_criteria("a", Options(), _dac(real_path))


class TestFeatherOrcUnreadableFile:
    """An unreadable file (corrupt, truncated, missing) declines when unpinned, but aborts when pinned."""

    def test_feather_declines_a_corrupt_file(self, tmp_path: Path, rejection_window: dict[str, MatchRejection]) -> None:
        group = load_group(*FEATHER)
        bad_path = tmp_path / "corrupt.feather"
        bad_path.write_bytes(b"not a real feather file")

        assert not group.match_feature_group_criteria("a", Options(), _dac(str(bad_path)))

        reason = rejection_window[group.get_class_name()].reason
        assert "could not read" in reason
        assert os.path.abspath(bad_path) in reason

    def test_orc_declines_a_corrupt_file(self, tmp_path: Path, rejection_window: dict[str, MatchRejection]) -> None:
        group = load_group(*ORC)
        bad_path = tmp_path / "corrupt.orc"
        bad_path.write_bytes(b"")

        assert not group.match_feature_group_criteria("a", Options(), _dac(str(bad_path)))

        reason = rejection_window[group.get_class_name()].reason
        assert "could not read" in reason
        assert os.path.abspath(bad_path) in reason

    def test_feather_declines_a_nonexistent_path_plain_name(self) -> None:
        group = load_group(*FEATHER)
        assert not group.match_feature_group_criteria("a", Options(), _dac("dummy.feather"))

    def test_orc_declines_a_nonexistent_path_plain_name(self) -> None:
        group = load_group(*ORC)
        assert not group.match_feature_group_criteria("a", Options(), _dac("dummy.orc"))

    def test_a_pinned_corrupt_feather_file_aborts_instead_of_falling_back_to_a_sibling(self, tmp_path: Path) -> None:
        group = load_group(*FEATHER)
        good = _write_feather(tmp_path, ["a"])
        corrupt = tmp_path / "corrupt.feather"
        corrupt.write_bytes(b"not a real feather file")
        dac = DataAccessCollection(
            files={"pinned": str(corrupt), "sibling": good},
            column_to_file={"a": "pinned"},
        )

        with pytest.raises(ValueError) as exc_info:
            resolve_or_raise(Feature("a"), {group: {PyArrowTable}}, None, dac)

        assert os.path.abspath(corrupt) in str(exc_info.value)
