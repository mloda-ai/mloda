"""RED tests pinning the compute-framework-neutral CSV seam.

Target design (these FAIL today):

  1. ``CsvReader.load_data`` RESOLVES a CSV into a lightweight, immutable ``FileSource``
     descriptor (``.path``, ``.format``, ``.columns``) instead of eagerly materializing a
     ``pyarrow.Table``. This decouples CSV input from pyarrow: any compute framework can
     later materialize the descriptor into its own native type.

  2. A ``(FileSource, dict)`` transformer is registered with
     ``ComputeFrameworkTransformer`` and materializes a ``FileSource`` into a columnar
     ``dict[str, list[Any]]`` using only the stdlib.

Failure reasons today:
  * ``mloda.core.abstract_plugins.components.input_data.file_source`` does not exist, so
    importing ``FileSource`` raises ModuleNotFoundError (module-level, right RED reason).
  * Once that lands, ``CsvReader.load_data`` still returns a ``pyarrow.Table`` (not a
    ``FileSource``) and no ``(FileSource, dict)`` transformer is registered.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterator

import pytest

from mloda.core.abstract_plugins.components.framework_transformer.cfw_transformer import (
    ComputeFrameworkTransformer,
)
from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.provider import FeatureSet
from mloda.user import Feature
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
        """Fails today: load_data returns a ``pyarrow.Table`` selection, not a FileSource."""
        features = _feature_set(["A", "B"])
        result = CsvReader.load_data(csv_path, features)

        assert isinstance(result, FileSource), f"Expected a FileSource descriptor, got {type(result)!r}"

    def test_file_source_carries_path_format_and_columns(self, csv_path: str) -> None:
        """FileSource pins ``.path`` / ``.format`` / ``.columns`` as an immutable, sorted tuple.

        Fails today: ``columns`` is ``list(features.get_all_names())`` (a list, in
        nondeterministic set order), not a deterministically sorted tuple.
        """
        features = _feature_set(["A", "B"])
        result = CsvReader.load_data(csv_path, features)

        assert result.path == csv_path
        assert result.format == "csv"

        # Deterministic, stable column order: sorted, as an immutable tuple.
        assert isinstance(result.columns, tuple)
        assert result.columns == tuple(sorted(features.get_all_names()))
        assert set(result.columns) == {"A", "B"}


class TestFileSourceIsImmutableDescriptor:
    """FileSource is a small immutable value object."""

    def test_file_source_columns_is_a_tuple(self) -> None:
        """``columns`` is normalized to an immutable ``tuple``, not a list.

        Fails today: ``FileSource`` stores ``columns`` verbatim, so a list stays a list.
        """
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
        """A genuinely immutable descriptor must be hashable.

        Fails today: ``columns`` is a list, so ``hash(source)`` raises ``TypeError``.
        """
        source = FileSource(path="data/x.csv", format="csv", columns=["A", "B"])
        hash(source)  # must NOT raise


class TestFileSourceToDictTransformer:
    """A (FileSource, dict) transformer materializes the descriptor via the stdlib."""

    def test_transformer_map_has_file_source_dict_pair(self) -> None:
        """The registry exposes a ``(FileSource, dict)`` transformer.

        Fails today: no transformer registers the ``(FileSource, dict)`` framework pair.
        """
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

        Fails today: the transformer appends raw ``csv.reader`` cells, so every value is
        a ``str`` (``materialized["A"] == ["1", "2"]``, not ``[1, 2]``).
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

        Fails today: ``get_column_names`` opens the file with the process default
        encoding and no BOM handling, so the first header name is ``"\\ufeffcafé"`` (BOM
        leaked in) rather than ``"café"``.
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
