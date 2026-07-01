"""RED-phase tests for the result serialization helpers (issue #573).

These tests define the contract for two framework-aware serialization helpers
exposed as ``@staticmethod`` on ``mlodaAPI`` (aliased as ``mloda`` in
``mloda.user``):

- ``mloda.to_json_records(result) -> list[dict[str, Any]]``
- ``mloda.to_csv(result) -> str``

``result`` is a SINGLE compute-framework object (one element of the
``run_all`` list), not the whole list.

They use only the always-installed frameworks (pandas, python_dict, pyarrow)
so they never skip (which would break the tox EXPECTED_SKIP_COUNT). The tests
are expected to FAIL until the helpers are implemented.
"""

import csv
import io
from typing import Any, Optional

import pandas as pd
import pyarrow as pa

from mloda.provider import FeatureGroup, FeatureSet, DataCreator, BaseInputData
from mloda.user import Feature, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,  # noqa: F401
)


FEAT = "ResultSerializerFeature"
PY_DICT_FEAT = "ResultSerializerPyDictFeature"


class ResultSerializerFeatureGroup(FeatureGroup):
    """Root feature group producing a small deterministic frame.

    Declared module-level so it registers as a discoverable FeatureGroup. It
    produces a single root feature via ``DataCreator`` and returns a pandas
    DataFrame; the engine converts to the pinned framework as needed.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({FEAT})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({FEAT: [1, 2, 3]})


class ResultSerializerPyDictFeatureGroup(FeatureGroup):
    """Root feature group producing python_dict-native columnar data.

    Returns a columnar dict which ``PythonDictFramework`` converts to a
    row-based ``list[dict]`` natively, so no cross-framework (pandas -> list)
    engine conversion is required.
    """

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({PY_DICT_FEAT})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {PY_DICT_FEAT: [1, 2, 3]}


class TestToJsonRecordsPandas:
    """``to_json_records`` on a pandas ``run_all`` result."""

    def test_pandas_result_to_json_records(self) -> None:
        results = mloda.run_all([Feature(FEAT)], compute_frameworks=["PandasDataFrame"])
        assert isinstance(results, list) and len(results) == 1
        assert isinstance(results[0], pd.DataFrame)

        records = mloda.to_json_records(results[0])
        assert records == [{FEAT: 1}, {FEAT: 2}, {FEAT: 3}]


class TestToCsvPandas:
    """``to_csv`` on a pandas ``run_all`` result."""

    def test_pandas_result_to_csv(self) -> None:
        results = mloda.run_all([Feature(FEAT)], compute_frameworks=["PandasDataFrame"])
        csv_str = mloda.to_csv(results[0])

        assert isinstance(csv_str, str)
        lines = [line for line in csv_str.splitlines() if line != ""]
        assert lines[0] == FEAT, f"First line must be the header, got: {lines[0]!r}"
        assert lines[1:] == ["1", "2", "3"], f"Rows must be the column values, got: {lines[1:]!r}"


class TestToJsonRecordsPythonDict:
    """``to_json_records`` on a python_dict ``run_all`` result.

    Uses a python_dict-native feature group so the engine exercises
    ``PythonDictFramework`` without any cross-framework conversion.
    """

    def test_python_dict_result_to_json_records(self) -> None:
        results = mloda.run_all([Feature(PY_DICT_FEAT)], compute_frameworks=["PythonDictFramework"])
        assert isinstance(results, list) and len(results) == 1
        assert isinstance(results[0], list)

        records = mloda.to_json_records(results[0])
        assert records == [{PY_DICT_FEAT: 1}, {PY_DICT_FEAT: 2}, {PY_DICT_FEAT: 3}]


class TestToCsvPythonDict:
    """``to_csv`` on a python_dict ``run_all`` result.

    Uses a python_dict-native feature group so the engine exercises
    ``PythonDictFramework`` without any cross-framework conversion.
    """

    def test_python_dict_result_to_csv(self) -> None:
        results = mloda.run_all([Feature(PY_DICT_FEAT)], compute_frameworks=["PythonDictFramework"])
        csv_str = mloda.to_csv(results[0])

        assert isinstance(csv_str, str)
        lines = [line for line in csv_str.splitlines() if line != ""]
        assert lines[0] == PY_DICT_FEAT, f"First line must be the header, got: {lines[0]!r}"
        assert lines[1:] == ["1", "2", "3"], f"Rows must be the column values, got: {lines[1:]!r}"


class TestPyarrowTableDirectly:
    """Both helpers accept a ``pa.Table`` passed directly."""

    def test_pyarrow_table_to_json_records(self) -> None:
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        records = mloda.to_json_records(table)
        assert records == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_pyarrow_table_to_csv(self) -> None:
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        csv_str = mloda.to_csv(table)

        assert isinstance(csv_str, str)
        lines = [line for line in csv_str.splitlines() if line != ""]
        assert lines[0] == "a,b", f"First line must be the header, got: {lines[0]!r}"
        assert lines[1:] == ["1,x", "2,y"], f"Rows must match the table values, got: {lines[1:]!r}"


class TestSparsePythonDictRecords:
    """python_dict ``list[dict]`` results with sparse keys keep all columns.

    The KEY regression: ``to_csv`` must emit the UNION of keys across all
    records, not just the keys present in the first record. Missing values
    become empty CSV cells.
    """

    def test_sparse_records_to_json_records_unchanged(self) -> None:
        records = [{"a": 1}, {"b": 2}]
        assert mloda.to_json_records(records) == [{"a": 1}, {"b": 2}]

    def test_sparse_records_to_csv_keeps_all_columns(self) -> None:
        records = [{"a": 1}, {"b": 2}]
        csv_str = mloda.to_csv(records)

        lines = [line for line in csv_str.splitlines() if line != ""]
        header = lines[0]
        assert set(header.split(",")) == {"a", "b"}, f"Header must be union of keys, got: {header!r}"

        parsed = list(csv.DictReader(io.StringIO(csv_str)))
        assert parsed == [{"a": "1", "b": ""}, {"a": "", "b": "2"}]


class TestFloatsAndNullsPyarrow:
    """Floats and nulls via a ``pa.Table`` serialize cleanly."""

    def test_floats_and_nulls_to_json_records(self) -> None:
        table = pa.table({"a": [1.5, None], "b": ["x", None]})
        assert mloda.to_json_records(table) == [{"a": 1.5, "b": "x"}, {"a": None, "b": None}]

    def test_floats_and_nulls_to_csv(self) -> None:
        table = pa.table({"a": [1.5, None], "b": ["x", None]})
        csv_str = mloda.to_csv(table)

        lines = [line for line in csv_str.splitlines() if line != ""]
        assert lines[0] == "a,b", f"First line must be the header, got: {lines[0]!r}"

        parsed = list(csv.DictReader(io.StringIO(csv_str)))
        assert parsed == [{"a": "1.5", "b": "x"}, {"a": "", "b": ""}]


class TestCsvEscaping:
    """Values containing commas, quotes, and newlines round-trip correctly."""

    def test_special_characters_round_trip(self) -> None:
        records = [
            {"a": "x,y", "b": 'he said "hi"'},
            {"a": "line1\nline2", "b": "plain"},
        ]
        csv_str = mloda.to_csv(records)

        parsed = list(csv.DictReader(io.StringIO(csv_str)))
        assert parsed == [
            {"a": "x,y", "b": 'he said "hi"'},
            {"a": "line1\nline2", "b": "plain"},
        ]


class TestEmptyTableWithSchema:
    """An empty ``pa.Table`` that still carries a schema yields header-only CSV."""

    def test_empty_table_to_json_records(self) -> None:
        table = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.string())})
        assert mloda.to_json_records(table) == []

    def test_empty_table_to_csv_header_only(self) -> None:
        table = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.string())})
        csv_str = mloda.to_csv(table)

        lines = [line for line in csv_str.splitlines() if line != ""]
        assert len(lines) == 1, f"Empty table must yield header-only CSV, got: {lines!r}"
        assert set(lines[0].split(",")) == {"a", "b"}


class TestAmbiguousListRaises:
    """Passing the whole ``run_all`` list (non-dict elements) is a caller error."""

    def test_list_of_dataframes_raises_value_error(self) -> None:
        whole_output = [pd.DataFrame({FEAT: [1, 2, 3]})]

        import pytest

        with pytest.raises(ValueError):
            mloda.to_json_records(whole_output)

        with pytest.raises(ValueError):
            mloda.to_csv(whole_output)
