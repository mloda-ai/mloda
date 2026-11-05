"""Shared setup for tests of the credential handle a source step binds at match time.

Defines no FeatureGroup subclass: module-level source groups leak into other tests in a worker.
"""

from pathlib import Path
from typing import Any

from mloda.user import Credential, DataAccessCollection
from tests.mixins.reader_feature_groups.format_file_writers import write_sqlite

BOUND_KINDS = ["named", "unnamed", "pointer", "pointer-no-collection"]


def write_bound_sqlite(tmp_path: Path) -> Path:
    """Write the small sqlite db holding bound_col and return its path."""
    path = tmp_path / "bound.db"
    write_sqlite(path, {"bound_table": {"bound_col": [1, 2]}})
    return path


def bound_case(tmp_path: Path, kind: str) -> tuple[DataAccessCollection | None, dict[str, Any], tuple[str, ...]]:
    """A SqliteFG source setup per binding kind: (collection, feature options, expected handles)."""
    path = write_bound_sqlite(tmp_path)
    if kind == "named":
        return DataAccessCollection(credentials={"db-main": {"sqlite": str(path)}}), {}, ("db-main",)
    if kind == "unnamed":
        return DataAccessCollection(credentials=[{"sqlite": str(path)}]), {}, ("<unnamed>",)
    pointer = {"SqliteFG": Credential(sqlite=str(path))}
    if kind == "pointer":
        return DataAccessCollection(credentials={"other": {"sqlite": str(path)}}), pointer, ("<pointer>",)
    return None, pointer, ("<pointer>",)


def csv_file_source(tmp_path: Path) -> DataAccessCollection:
    """A collection pointing at a small csv file holding bound_col."""
    path = tmp_path / "bound.csv"
    path.write_text("bound_col\n1\n2\n")
    return DataAccessCollection(files={str(path)})
