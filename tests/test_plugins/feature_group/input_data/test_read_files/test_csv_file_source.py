"""Pins the compute-framework-neutral CSV seam.

Contract:

  1. ``CsvReader.load_data`` RESOLVES a CSV into a lightweight, immutable ``FileSource``
     descriptor (``.path``, ``.format``, ``.columns``) instead of eagerly materializing a
     ``pyarrow.Table``. This decouples CSV input from pyarrow: any compute framework can
     later materialize the descriptor into its own native type.

  2. ``FileSource`` is a frozen dataclass subclassing the ``InputDataDescriptor`` marker;
     ``columns`` is normalized to a tuple in ``__post_init__`` so the descriptor stays
     hashable even when built from a list.

  3. A ``(FileSource, dict)`` transformer is registered with
     ``ComputeFrameworkTransformer`` and materializes a ``FileSource`` into a columnar
     ``dict[str, list[Any]]`` using only the stdlib.

  4. ``CsvReader.get_column_names`` discovers the header with the stdlib ``csv`` module,
     decoding UTF-8 explicitly and stripping a leading BOM.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.core.abstract_plugins.components.input_data.input_data_descriptor import InputDataDescriptor
from mloda.provider import FeatureSet
from mloda.user import DataAccessCollection, Feature, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader


@pytest.fixture()
def csv_path() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    # A: integer column, B: non-numeric (stays str), C: float column.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B", "C"])
        writer.writerow(["1", "x", "1.5"])
        writer.writerow(["2", "y", "2.5"])
    yield path
    os.remove(path)


def _feature_set(names: list[str]) -> FeatureSet:
    features = FeatureSet()
    for name in names:
        features.add(Feature(name))
    return features


class TestCsvLoadDataResolvesFileSource:
    """CsvReader.load_data returns a FileSource descriptor, not a materialized table."""

    def test_load_data_returns_file_source(self, csv_path: str) -> None:
        """load_data resolves to a FileSource descriptor, not a materialized pyarrow.Table."""
        features = _feature_set(["A", "B"])
        result = CsvReader.load_data(csv_path, features)

        assert isinstance(result, FileSource), f"Expected a FileSource descriptor, got {type(result)!r}"

    def test_file_source_carries_path_format_and_columns(self, csv_path: str) -> None:
        """FileSource pins ``.path`` / ``.format`` / ``.columns`` as an immutable, sorted tuple."""
        features = _feature_set(["A", "B"])
        result = CsvReader.load_data(csv_path, features)

        assert result.path == csv_path
        assert result.format == "csv"

        # Deterministic, stable column order: sorted, as an immutable tuple.
        assert isinstance(result.columns, tuple)
        assert result.columns == tuple(sorted(features.get_all_names()))
        assert set(result.columns) == {"A", "B"}


class TestFileSourceIsImmutableDescriptor:
    """FileSource is a small immutable value object marked as an input-data descriptor."""

    def test_file_source_is_an_input_data_descriptor(self) -> None:
        """The descriptor marker gates multi-hop materialization in root ingestion."""
        source = FileSource(path="data/x.csv", format="csv", columns=("A", "B"))
        assert isinstance(source, InputDataDescriptor)

    def test_file_source_columns_is_a_tuple(self) -> None:
        """``columns`` is normalized to an immutable ``tuple`` in ``__post_init__``."""
        source = FileSource(path="data/x.csv", format="csv", columns=["A", "B"])
        assert source.path == "data/x.csv"
        assert source.format == "csv"
        assert isinstance(source.columns, tuple)
        assert source.columns == ("A", "B")

    def test_file_source_is_frozen(self) -> None:
        """A FileSource must not allow attribute reassignment (immutable descriptor)."""
        source = FileSource(path="data/x.csv", format="csv", columns=["A", "B"])
        with pytest.raises(Exception):
            source.path = "data/y.csv"  # type: ignore[misc]

    def test_file_source_is_hashable(self) -> None:
        """A genuinely immutable descriptor must be hashable, even when built from a list."""
        source = FileSource(path="data/x.csv", format="csv", columns=["A", "B"])
        hash(source)  # must NOT raise


class TestFileSourceToDictTransformer:
    """A (FileSource, dict) transformer materializes the descriptor via the stdlib."""

    def test_transformer_map_has_file_source_dict_pair(self) -> None:
        """The registry exposes a ``(FileSource, dict)`` transformer."""
        transformer_map = ComputeFrameworkTransformer().transformer_map
        assert (FileSource, dict) in transformer_map, (
            f"Expected a (FileSource, dict) transformer. Registered pairs: {list(transformer_map)!r}"
        )

    def test_transformer_materializes_columnar_dict_with_typed_values(self, csv_path: str) -> None:
        """FileSource -> columnar ``dict[str, list]`` with typed (not stringified) values.

        Before the FileSource seam, PythonDict+CSV received typed values via the
        ``pa.Table -> dict`` path. The stdlib ``(FileSource -> dict)`` transformer must
        preserve that: an integer column stays ``int``, a float column stays ``float``,
        and a non-numeric column stays ``str``.
        """
        transformer_map = ComputeFrameworkTransformer().transformer_map
        transformer = transformer_map[(FileSource, dict)]

        source = FileSource(path=csv_path, format="csv", columns=["A", "B", "C"])
        materialized = transformer.transform(FileSource, dict, source, None)

        assert isinstance(materialized, dict)
        assert set(materialized.keys()) == {"A", "B", "C"}

        # Integer column: real ints, not strings.
        assert materialized["A"] == [1, 2]
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in materialized["A"])

        # Float column: real floats, not strings.
        assert materialized["C"] == [1.5, 2.5]
        assert all(isinstance(v, float) for v in materialized["C"])

        # Non-numeric column: stays str.
        assert materialized["B"] == ["x", "y"]
        assert all(isinstance(v, str) for v in materialized["B"])


class TestCsvHeaderIsUtf8Decoded:
    """Header discovery must decode UTF-8 explicitly and strip a leading BOM."""

    def test_get_column_names_decodes_utf8_and_strips_bom(self) -> None:
        """A UTF-8 header (with a non-ASCII name and a BOM) is decoded correctly.

        The file is written as explicit UTF-8 bytes prefixed with a UTF-8 BOM
        (``\\xef\\xbb\\xbf``). This makes the assertion deterministic regardless of the
        CI locale: it can only pass with an explicit ``utf-8-sig`` decode.
        """
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        content = "café,B\n1,3\n2,4\n"
        with open(path, "wb") as f:
            f.write(b"\xef\xbb\xbf" + content.encode("utf-8"))
        try:
            names = list(CsvReader.get_column_names(path))

            assert names[0] == "café", f"expected 'café' as the first header, got {names[0]!r}"
            assert not names[0].startswith("\ufeff"), f"BOM leaked into the first header: {names[0]!r}"
            assert names == ["café", "B"]
        finally:
            os.remove(path)


@pytest.fixture()
def csv_path_with_blanks_and_embedded_newline() -> Iterator[str]:
    """One quoted embedded-newline row, a blank interior line, and a trailing blank line."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B"])
        writer.writerow(["1", "line1\nline2"])
        writer.writerow([])
        writer.writerow(["2", "y"])
        writer.writerow([])
    yield path
    os.remove(path)


class TestCsvReaderCountRows:
    """CsvReader.count_rows counts like FileSourceDictTransformer, dict framework only."""

    def test_count_rows_matches_transformer_row_count_for_str_and_path(
        self, csv_path_with_blanks_and_embedded_newline: str
    ) -> None:
        transformer_map = ComputeFrameworkTransformer().transformer_map
        transformer = transformer_map[(FileSource, dict)]
        source = FileSource(path=csv_path_with_blanks_and_embedded_newline, format="csv", columns=("A", "B"))
        materialized = transformer.transform(FileSource, dict, source, None)
        expected = len(materialized["A"])

        for candidate in (csv_path_with_blanks_and_embedded_newline, Path(csv_path_with_blanks_and_embedded_newline)):
            assert CsvReader.count_rows(candidate, PythonDictFramework) == expected

    def test_count_rows_raises_value_error_on_ragged_row_like_the_transformer(self, tmp_path: Path) -> None:
        """A short data row is a ValueError in FileSourceDictTransformer; count_rows must match."""
        path = tmp_path / "ragged.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["A", "B"])
            writer.writerow(["1"])

        transformer_map = ComputeFrameworkTransformer().transformer_map
        transformer = transformer_map[(FileSource, dict)]
        source = FileSource(path=str(path), format="csv", columns=("A", "B"))

        with pytest.raises(ValueError):
            transformer.transform(FileSource, dict, source, None)
        with pytest.raises(ValueError):
            CsvReader.count_rows(str(path), PythonDictFramework)

    def test_count_rows_rejects_non_path_data_access_under_dict_framework(self) -> None:
        with pytest.raises(ValueError):
            CsvReader.count_rows(DataAccessCollection(files={"dummy.csv"}), PythonDictFramework)

    def test_count_rows_raises_oserror_for_absent_file_under_dict_framework(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            CsvReader.count_rows(str(tmp_path / "absent.csv"), PythonDictFramework)

    def test_count_rows_counts_zero_for_empty_and_header_only_files(self, tmp_path: Path) -> None:
        empty_path = tmp_path / "empty.csv"
        empty_path.write_bytes(b"")
        assert CsvReader.count_rows(str(empty_path), PythonDictFramework) == 0

        header_only_path = tmp_path / "header_only.csv"
        with open(header_only_path, "w", newline="") as f:
            csv.writer(f).writerow(["A", "B"])
        assert CsvReader.count_rows(str(header_only_path), PythonDictFramework) == 0

    def test_count_rows_is_none_under_non_dict_frameworks(self, csv_path_with_blanks_and_embedded_newline: str) -> None:
        assert CsvReader.count_rows(csv_path_with_blanks_and_embedded_newline, PyArrowTable) is None
        assert CsvReader.count_rows(csv_path_with_blanks_and_embedded_newline, PandasDataFrame) is None

    def test_count_rows_is_none_for_non_path_data_access_under_non_dict_framework(self) -> None:
        """expected_data_framework is checked first, so a non-path access never reaches ValueError."""
        non_path = DataAccessCollection(files={"dummy.csv"})
        assert CsvReader.count_rows(non_path, PyArrowTable) is None
        assert CsvReader.count_rows(non_path, PandasDataFrame) is None

    def test_count_rows_reports_none_for_a_load_data_overriding_subclass(
        self, csv_path_with_blanks_and_embedded_newline: str
    ) -> None:
        class _CsvCountRowsProbeReader(CsvReader):
            @classmethod
            def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
                return super().load_data(data_access, features)

            @classmethod
            def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
                return None

        assert (
            _CsvCountRowsProbeReader.count_rows(csv_path_with_blanks_and_embedded_newline, PythonDictFramework) is None
        )
