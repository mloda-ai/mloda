"""The stdlib ``FileSource -> dict`` transformer infers types COLUMN-WISE.

Contract for ``FileSourceDictTransformer.transform_fw_to_other_fw`` (matching the
column semantics pyarrow's CSV reader gives, without pyarrow):

  (a) Empty-cell handling matches pyarrow's default CSV reader (``strings_can_be_null=False``):
      a NUMERIC column with a missing cell is ``[None, 3]`` (typed where present, ``None`` for the
      gap); a STRING column's missing interior cell stays ``""`` (empty string, NOT ``None``);
      an ALL-EMPTY column (every cell empty) is inferred as all-``None``.
  (b) A column whose non-empty cells are all integers -> ``int``; all decimals -> ``float``;
      a column mixing numeric and text cells stays ``str``.
  (c) Numeric-LOOKING text stays ``str``: ``"1_000"`` must NOT parse to ``1000``; leading /
      trailing whitespace must NOT be stripped-and-parsed; the words ``"NaN"`` / ``"Inf"``
      alongside non-numeric text stay ``str`` (no ``float('nan')`` / ``float('inf')``).
  (d) A column whose cells are all in ``{true,false,True,False}`` -> Python ``bool``.
  (e) Blank interior / trailing lines are skipped (no ``IndexError``).
  (f) A ragged, non-empty short row raises a clear ``ValueError`` (naming the file path
      and/or a row number), not a bare ``IndexError``.
  (g) Duplicate header names raise a clear ``ValueError`` (naming the duplicated column)
      when a duplicated column is requested, matching pyarrow's refusal.
  (h) A requested column missing from the header raises a clear ``ValueError`` naming it.
  (i) A UTF-8 BOM at the start of the file is tolerated (``utf-8-sig``).
  (j) A non-csv ``FileSource.format`` raises ``ValueError``; the reverse direction
      (dict -> FileSource) raises ``NotImplementedError``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.input_data.file_source import FileSource
from mloda.user import DataAccessCollection
from mloda.user import mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_file_source_transformer import (
    FileSourceDictTransformer,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature  # noqa: F401
from mloda_plugins.feature_group.input_data.read_files.csv import CsvReader  # noqa: F401


def _materialize(tmp_path: Path, content: str, columns: tuple[str, ...]) -> Any:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return FileSourceDictTransformer.transform_fw_to_other_fw(FileSource(path=str(path), format="csv", columns=columns))


class TestColumnWiseTypeInference:
    def test_empty_cell_is_none_and_numeric_column_keeps_ints(self, tmp_path: Path) -> None:
        """(a) Empty cell -> None; a numeric column with a gap is ``[None, 3]``, not ``["", 3]``."""
        result = _materialize(tmp_path, "a,b,c\n1,,x\n2,3,y\n", ("a", "b", "c"))

        assert result["a"] == [1, 2]
        assert result["b"] == [None, 3]
        assert result["c"] == ["x", "y"]

    def test_string_column_missing_interior_cell_is_empty_string(self, tmp_path: Path) -> None:
        """(a) A STRING column's missing interior cell stays ``""`` (matches pyarrow), not ``None``."""
        result = _materialize(tmp_path, "a,b\nx,1\n,2\ny,3\n", ("a", "b"))

        assert result["a"] == ["x", "", "y"]
        assert result["b"] == [1, 2, 3]

    def test_all_empty_column_is_all_none(self, tmp_path: Path) -> None:
        """(a) A column whose every cell is empty is inferred as all-``None`` (matches pyarrow)."""
        result = _materialize(tmp_path, "a,b\n1,\n2,\n", ("a", "b"))

        assert result["a"] == [1, 2]
        assert result["b"] == [None, None]

    def test_integer_float_and_mixed_columns(self, tmp_path: Path) -> None:
        """(b) all-int -> int; all-decimal -> float; numeric+text -> str."""
        result = _materialize(
            tmp_path,
            "ints,floats,mixed\n1,1.5,1\n2,2.5,x\n",
            ("ints", "floats", "mixed"),
        )

        assert result["ints"] == [1, 2]
        assert all(isinstance(v, int) and not isinstance(v, bool) for v in result["ints"])

        assert result["floats"] == [1.5, 2.5]
        assert all(isinstance(v, float) for v in result["floats"])

        # A column mixing a number and text stays entirely string.
        assert result["mixed"] == ["1", "x"]
        assert all(isinstance(v, str) for v in result["mixed"])

    def test_numeric_looking_text_stays_string(self, tmp_path: Path) -> None:
        """(c) ``"1_000"`` and whitespace-padded numbers must NOT be parsed as numbers."""
        result = _materialize(tmp_path, "a\n1_000\n2\n", ("a",))
        assert result["a"] == ["1_000", "2"]
        assert all(isinstance(v, str) for v in result["a"])

    def test_whitespace_padded_number_is_not_parsed(self, tmp_path: Path) -> None:
        """(c) Leading/trailing whitespace must not be stripped-then-parsed into a number."""
        result = _materialize(tmp_path, "a\n 1 \n2\n", ("a",))
        assert result["a"] == [" 1 ", "2"]
        assert all(isinstance(v, str) for v in result["a"])

    def test_nan_and_inf_words_stay_string(self, tmp_path: Path) -> None:
        """(c) Text ``NaN`` / ``Inf`` next to non-numeric text stays str (no float coercion)."""
        result = _materialize(tmp_path, "a,b\nNaN,Inf\nfoo,bar\n", ("a", "b"))

        assert result["a"] == ["NaN", "foo"]
        assert result["b"] == ["Inf", "bar"]
        for values in (result["a"], result["b"]):
            for v in values:
                assert isinstance(v, str)
                assert not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))

    def test_boolean_column(self, tmp_path: Path) -> None:
        """(d) A column of ``true``/``false`` (any case) -> Python bools."""
        result = _materialize(tmp_path, "flag\ntrue\nfalse\nTrue\nFalse\n", ("flag",))
        assert result["flag"] == [True, False, True, False]
        assert all(isinstance(v, bool) for v in result["flag"])


class TestBlankAndRaggedRows:
    def test_blank_interior_and_trailing_lines_are_skipped(self, tmp_path: Path) -> None:
        """(e) Fully blank lines are skipped rather than raising IndexError."""
        result = _materialize(tmp_path, "a,b\n1,2\n\n3,4\n", ("a", "b"))
        assert result == {"a": [1, 3], "b": [2, 4]}

    def test_ragged_short_row_raises_valueerror_with_context(self, tmp_path: Path) -> None:
        """(f) A non-empty short row raises a clear ValueError, not a bare IndexError."""
        path = tmp_path / "data.csv"
        path.write_text("a,b,c\n1,2,3\n9,8\n", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            FileSourceDictTransformer.transform_fw_to_other_fw(
                FileSource(path=str(path), format="csv", columns=("a", "b", "c"))
            )
        message = str(excinfo.value)
        # Must name the file path and/or a row number so the error is actionable.
        assert str(path) in message or "row" in message.lower(), message

    def test_ragged_long_row_raises_valueerror_with_context(self, tmp_path: Path) -> None:
        """(f) A row with MORE fields than the header raises a clear ValueError (pyarrow errors too),
        rather than silently dropping the extra field(s)."""
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2,3\n", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            FileSourceDictTransformer.transform_fw_to_other_fw(
                FileSource(path=str(path), format="csv", columns=("a", "b"))
            )
        message = str(excinfo.value)
        assert str(path) in message or "row" in message.lower(), message

    def test_duplicate_header_raises_valueerror_naming_column(self, tmp_path: Path) -> None:
        """(g) A duplicated header raises a clear ValueError naming the duplicated column."""
        path = tmp_path / "data.csv"
        path.write_text("A,A,B\n1,2,3\n", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            FileSourceDictTransformer.transform_fw_to_other_fw(
                FileSource(path=str(path), format="csv", columns=("A", "B"))
            )
        message = str(excinfo.value)
        assert "duplicate" in message.lower()
        assert "A" in message

    def test_missing_requested_column_raises_valueerror_naming_column(self, tmp_path: Path) -> None:
        """(h) Requesting a column absent from the header raises a clear ValueError naming it."""
        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            FileSourceDictTransformer.transform_fw_to_other_fw(
                FileSource(path=str(path), format="csv", columns=("a", "zzz"))
            )
        assert "zzz" in str(excinfo.value)


class TestEncodingAndFormat:
    def test_utf8_bom_is_tolerated(self, tmp_path: Path) -> None:
        """(i) A UTF-8 BOM must not leak into the first header name (utf-8-sig decode)."""
        path = tmp_path / "data.csv"
        content = "a,b\n1,x\n2,y\n"
        path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        result = FileSourceDictTransformer.transform_fw_to_other_fw(
            FileSource(path=str(path), format="csv", columns=("a", "b"))
        )

        assert result["a"] == [1, 2]
        assert result["b"] == ["x", "y"]

    def test_non_csv_format_raises_valueerror(self, tmp_path: Path) -> None:
        """(j) The stdlib transformer only materializes the 'csv' format."""
        with pytest.raises(ValueError) as excinfo:
            FileSourceDictTransformer.transform_fw_to_other_fw(
                FileSource(path=str(tmp_path / "data.parquet"), format="parquet", columns=("a",))
            )
        assert "parquet" in str(excinfo.value)

    def test_reverse_direction_raises_not_implemented(self) -> None:
        """(j) dict -> FileSource makes no sense; the reverse direction must raise."""
        with pytest.raises(NotImplementedError):
            FileSourceDictTransformer.transform_other_fw_to_fw({"a": [1]})


class TestDeliberateNullTokenDivergenceFromPyArrow:
    """PASSING divergence pin: the stdlib materializer does NOT recognize null tokens.

    DELIBERATE divergence, pinned so it cannot drift silently (issue #619 tracks null-token
    inference as a follow-up): on the SAME CSV file with the column ``1,2,NA``, the stdlib
    ``FileSourceDictTransformer`` keeps the column entirely string
    (``["1", "2", "NA"]``, per its documented "such a column currently stays string" rule),
    while pyarrow's CSV reader treats ``NA`` as a null token and yields ``[1, 2, None]``.

    This test must PASS today. If it ever fails, either the stdlib transformer started
    inferring null tokens (close #619 and update this pin) or pyarrow changed its default
    null tokens; both deserve a conscious decision, not a silent drift.
    """

    def test_na_token_column_diverges_between_dict_and_pyarrow_materializers(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("a\n1\n2\nNA\n", encoding="utf-8")
        source = FileSource(path=str(path), format="csv", columns=("a",))

        dict_result = FileSourceDictTransformer.transform_fw_to_other_fw(source)
        assert dict_result["a"] == ["1", "2", "NA"]
        assert all(isinstance(v, str) for v in dict_result["a"])

        pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
            FileSourcePyArrowTransformer,
        )

        table = FileSourcePyArrowTransformer.transform_fw_to_other_fw(source)
        assert table.column("a").to_pylist() == [1, 2, None]


class TestEndToEndPythonDictCsv:
    """run_all on a CSV into PythonDict yields None/typed/bool values (mirrors CSV e2e tests)."""

    def test_run_all_reads_typed_none_and_bool(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        # a: int column, b: numeric column with a missing cell, c: boolean column.
        path.write_text("a,b,c\n1,,true\n2,3,false\n", encoding="utf-8")

        result = mloda.run_all(
            ["a", "b", "c"],
            compute_frameworks={PythonDictFramework},
            data_access_collection=DataAccessCollection(files={str(path)}),
        )

        payload = result[0]
        assert isinstance(payload, dict)
        assert payload["a"] == [1, 2]
        assert payload["b"] == [None, 3]
        assert payload["c"] == [True, False]
