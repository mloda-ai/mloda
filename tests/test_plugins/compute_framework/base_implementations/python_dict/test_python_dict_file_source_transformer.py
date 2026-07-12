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
  (k) Null tokens (pyarrow's default ``ConvertOptions.null_values``) become ``None`` in an
      int / float / bool column, stay literal strings in a STRING column
      (``strings_can_be_null=False``), and make a column of only tokens and/or empty cells all-``None``.
  (l) An all-int column with a value outside signed int64 range degrades entirely to ``float``
      (pyarrow's int64 -> double fallback); int64 min/max stay exact; beyond float64 range is ``inf``.
      Digit counts beyond 4300 are deliberate: CPython caps ``int(str)`` there (leading zeros count).
  (m) pyarrow's int64 parser rejects a leading ``+``, so a column containing ``+5`` is ``float``.
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


#: pyarrow's default ``ConvertOptions().null_values`` minus ``""``, pinned literally so a pyarrow
#: default change surfaces as a test failure.
_NULL_TOKENS: tuple[str, ...] = (
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
)


def _materialize(tmp_path: Path, content: str, columns: tuple[str, ...]) -> Any:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return FileSourceDictTransformer.transform_fw_to_other_fw(FileSource(path=str(path), format="csv", columns=columns))


def _types(values: list[Any]) -> list[type[Any]]:
    return [type(v) for v in values]


def _parity(tmp_path: Path, content: str, columns: tuple[str, ...], expected: dict[str, list[Any]]) -> dict[str, Any]:
    """Assert the dict materializer yields ``expected``, then that pyarrow agrees on the SAME file.

    The dict-side assertion runs BEFORE ``importorskip("pyarrow")``, so the stdlib expectations stay
    pinned without pyarrow; only the cross-check is skipped. Element TYPES are compared as well as
    values, because ``==`` is type-blind (``5 == 5.0``, ``True == 1``) and would accept exactly the
    int / float / bool divergence this suite exists to catch.
    """
    dict_result: dict[str, Any] = _materialize(tmp_path, content, columns)

    assert dict_result == expected
    for name in columns:
        assert _types(dict_result[name]) == _types(expected[name]), f"column {name!r} element types"

    pytest.importorskip("pyarrow")
    from mloda_plugins.compute_framework.base_implementations.pyarrow.pyarrow_file_source_transformer import (
        FileSourcePyArrowTransformer,
    )

    source = FileSource(path=str(tmp_path / "data.csv"), format="csv", columns=columns)
    table = FileSourcePyArrowTransformer.transform_fw_to_other_fw(source)
    for name in columns:
        pyarrow_column: list[Any] = table.column(name).to_pylist()
        assert dict_result[name] == pyarrow_column, f"column {name!r} values"
        assert _types(dict_result[name]) == _types(pyarrow_column), f"column {name!r} element types"

    return dict_result


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


class TestCsvInferenceParityWithPyArrow:
    """PARITY pin (issue #662): both materializers must agree on the SAME CSV file.

    pyarrow's CSV reader is the reference. Covers null tokens (k), int64 overflow (l) and the
    leading-``+`` int64 refusal (m).
    """

    def test_null_token_in_int_column_becomes_none(self, tmp_path: Path) -> None:
        """(k) ``NA`` in an otherwise-int column is a null: ``[1, 2, None]``, not all-string."""
        _parity(tmp_path, "a\n1\n2\nNA\n", ("a",), {"a": [1, 2, None]})

    def test_every_default_null_token_is_recognized_in_an_int_column(self, tmp_path: Path) -> None:
        """(k) The full pyarrow default ``null_values`` set (minus ``""``) nulls out in an int column."""
        content = "a\n1\n" + "\n".join(_NULL_TOKENS) + "\n2\n"
        _parity(tmp_path, content, ("a",), {"a": [1, *([None] * len(_NULL_TOKENS)), 2]})

    def test_null_token_in_float_column_becomes_none(self, tmp_path: Path) -> None:
        """(k) ``null`` in an otherwise-float column is a null; the column stays float."""
        _parity(tmp_path, "a\n1.5\nnull\n2.5\n", ("a",), {"a": [1.5, None, 2.5]})

    def test_null_token_in_bool_column_becomes_none(self, tmp_path: Path) -> None:
        """(k) ``NA`` between bools is a null; the column stays bool."""
        _parity(tmp_path, "a\ntrue\nNA\nfalse\n", ("a",), {"a": [True, None, False]})

    def test_null_tokens_stay_literal_strings_in_a_string_column(self, tmp_path: Path) -> None:
        """(k) ``strings_can_be_null=False``: in a string column a null token is just text."""
        _parity(tmp_path, "a\nx\nNA\nfoo\n", ("a",), {"a": ["x", "NA", "foo"]})

    def test_all_null_token_column_is_all_none(self, tmp_path: Path) -> None:
        """(k) A column of only null tokens (pyarrow infers the ``null`` type) is all-``None``."""
        _parity(tmp_path, "a\nNA\nnull\nNaN\n", ("a",), {"a": [None, None, None]})

    def test_null_tokens_mixed_with_empty_cells_are_all_none(self, tmp_path: Path) -> None:
        """(k) Null tokens plus empty cells (no other cells) -> all-``None``, not empty strings."""
        _parity(tmp_path, "a,b\nNA,1\n,2\nnull,3\n", ("a", "b"), {"a": [None, None, None], "b": [1, 2, 3]})

    def test_int64_overflow_degrades_whole_column_to_float(self, tmp_path: Path) -> None:
        """(l) One out-of-range value turns the whole column float, the in-range cell with it."""
        _parity(tmp_path, "a\n9223372036854775808\n1\n", ("a",), {"a": [9.223372036854776e18, 1.0]})

    def test_negative_int64_overflow_degrades_whole_column_to_float(self, tmp_path: Path) -> None:
        """(l) Below int64 min degrades the column to float as well."""
        _parity(tmp_path, "a\n-9223372036854775809\n1\n", ("a",), {"a": [-9.223372036854776e18, 1.0]})

    def test_int64_boundary_values_stay_exact_ints(self, tmp_path: Path) -> None:
        """(l) int64 max/min are IN range: they must stay exact ints, never lossy floats."""
        _parity(
            tmp_path,
            "a\n9223372036854775807\n-9223372036854775808\n",
            ("a",),
            {"a": [9223372036854775807, -9223372036854775808]},
        )

    def test_integer_too_large_for_float64_becomes_inf(self, tmp_path: Path) -> None:
        """(l) A 401-digit integer overflows float64: ``inf``, exactly as pyarrow yields."""
        result = _parity(tmp_path, "a\n1" + "0" * 400 + "\n2\n", ("a",), {"a": [math.inf, 2.0]})
        assert math.isinf(result["a"][0])

    def test_negative_integer_too_large_for_float64_becomes_negative_inf(self, tmp_path: Path) -> None:
        """(l) The negative counterpart yields ``-inf``."""
        result = _parity(tmp_path, "a\n-1" + "0" * 400 + "\n2\n", ("a",), {"a": [-math.inf, 2.0]})
        assert math.isinf(result["a"][0])

    def test_integer_beyond_int_max_str_digits_becomes_inf(self, tmp_path: Path) -> None:
        """(l) Past the 4300-digit ``int(str)`` cap, an ``int()``-based inference raises. pyarrow
        yields ``inf``; the dict path must too."""
        result = _parity(tmp_path, "a\n" + "1" * 5000 + "\n2\n", ("a",), {"a": [math.inf, 2.0]})
        assert math.isinf(result["a"][0])

    def test_leading_zeros_beyond_int_max_str_digits_stay_exact_int(self, tmp_path: Path) -> None:
        """(l) Leading zeros count toward the 4300-digit cap, so a value that fits int64 can still
        blow it. Both sides must read an exact ``1``, not raise."""
        _parity(tmp_path, "a\n" + "0" * 5000 + "1\n2\n", ("a",), {"a": [1, 2]})

    def test_leading_zeros_within_int_max_str_digits_stay_exact_int(self, tmp_path: Path) -> None:
        """(l) The short counterpart: fixing the huge zero-padded case must not push ``007`` to float."""
        _parity(tmp_path, "a\n007\n2\n", ("a",), {"a": [7, 2]})

    def test_leading_plus_integer_degrades_column_to_float(self, tmp_path: Path) -> None:
        """(m) ``+5`` is not an int for pyarrow: the column falls back to double, ``6`` with it."""
        _parity(tmp_path, "a\n+5\n6\n", ("a",), {"a": [5.0, 6.0]})

    def test_leading_plus_float_stays_float(self, tmp_path: Path) -> None:
        """(m) A leading ``+`` on a DECIMAL is fine for pyarrow's double parser: still float."""
        _parity(tmp_path, "a\n+5.5\n6.5\n", ("a",), {"a": [5.5, 6.5]})


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
