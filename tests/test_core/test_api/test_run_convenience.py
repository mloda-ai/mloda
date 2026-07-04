"""Failing tests for the fluent Results returned by ``mlodaAPI.run_all`` (issues #564 and #568).

``mlodaAPI.run_all`` must return a ``Results`` instance (a ``list`` subclass exported from
``mloda.user``) whose accessors replace the removed ``run_one`` / ``run_all_as_dataframe``:

- ``run_all(...).get_rows()``: one feature's result as a flat list of row dicts
- ``run_all(...).get_values(name)``: one column as a plain Python list
- ``run_all(...).get_df()``: all results horizontally concatenated into ONE DataFrame

``Results`` does not exist yet; this module must fail at collection with ImportError.

All FeatureGroups are module-local with unique ``rc564_`` / ``rc568_`` feature names
and each run is pinned to its own ``PluginCollector`` so the tests stay deterministic
and parallel-safe under pytest-xdist.
"""

from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, Results, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)

try:
    import polars as pl
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame  # noqa: F401
except ImportError:
    pl = None  # type: ignore[assignment]
    PolarsDataFrame = None  # type: ignore[assignment, misc]


class Rc564PandasFeatureGroup(FeatureGroup):
    """Root FG producing a single pandas column for the get_rows tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_pandas_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"rc564_pandas_feature": [1, 2]})


class Rc564DictFeatureGroup(FeatureGroup):
    """Root FG producing a single columnar-dict column for the get_rows tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_dict_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rc564_dict_feature": [1, 2]}


class Rc564ArrowFeatureGroup(FeatureGroup):
    """Root FG producing a single pyarrow column for the get_rows tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_arrow_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"rc564_arrow_feature": [1, 2]})


class Rc564PolarsFeatureGroup(FeatureGroup):
    """Root FG producing a single polars column for the get_rows tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_polars_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.DataFrame({"rc564_polars_feature": [1, 2]})


class Rc568PandasLeftFeatureGroup(FeatureGroup):
    """First of two pandas FGs for the horizontal-concat tests (3 rows)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_pd_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"rc568_pd_left": [1, 2, 3]})


class Rc568PandasRightFeatureGroup(FeatureGroup):
    """Second of two pandas FGs for the horizontal-concat tests (3 rows)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_pd_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"rc568_pd_right": [10, 20, 30]})


class Rc568PandasShortFeatureGroup(FeatureGroup):
    """Pandas FG with a DIFFERENT row count (2 rows) to force a concat mismatch."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_pd_short"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"rc568_pd_short": [7, 8]})


class Rc568PolarsLeftFeatureGroup(FeatureGroup):
    """First of two polars FGs for the horizontal-concat tests (3 rows)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_pl_left"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.DataFrame({"rc568_pl_left": [1, 2, 3]})


class Rc568PolarsRightFeatureGroup(FeatureGroup):
    """Second of two polars FGs for the horizontal-concat tests (3 rows)."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_pl_right"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pl.DataFrame({"rc568_pl_right": [10, 20, 30]})


class Rc568DictFeatureGroup(FeatureGroup):
    """Columnar-dict FG proving get_df rejects non-DataFrame results."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc568_dict_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rc568_dict_feature": [1, 2]}


_RC564_PANDAS = PluginCollector.enabled_feature_groups({Rc564PandasFeatureGroup})
_RC564_DICT = PluginCollector.enabled_feature_groups({Rc564DictFeatureGroup})
_RC564_ARROW = PluginCollector.enabled_feature_groups({Rc564ArrowFeatureGroup})
_RC564_POLARS = PluginCollector.enabled_feature_groups({Rc564PolarsFeatureGroup})
_RC568_PANDAS_PAIR = PluginCollector.enabled_feature_groups({Rc568PandasLeftFeatureGroup, Rc568PandasRightFeatureGroup})
_RC568_PANDAS_SINGLE = PluginCollector.enabled_feature_groups({Rc568PandasLeftFeatureGroup})
_RC568_PANDAS_MISMATCH = PluginCollector.enabled_feature_groups(
    {Rc568PandasLeftFeatureGroup, Rc568PandasShortFeatureGroup}
)
_RC568_POLARS_PAIR = PluginCollector.enabled_feature_groups({Rc568PolarsLeftFeatureGroup, Rc568PolarsRightFeatureGroup})
_RC568_DICT = PluginCollector.enabled_feature_groups({Rc568DictFeatureGroup})


class TestRunAllReturnType:
    """``mlodaAPI.run_all`` must return a fluent ``Results`` that is still a plain list."""

    def test_run_all_returns_results_instance_that_is_a_list(self) -> None:
        result = mlodaAPI.run_all(
            [Feature("rc568_pd_left")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_SINGLE,
        )

        assert isinstance(result, Results)
        assert isinstance(result, list)


class TestRunAllGetRows:
    """``run_all(...).get_rows()`` yields flat row dicts for every compute framework."""

    def test_pandas_result_returns_flat_row_dicts(self) -> None:
        rows = mlodaAPI.run_all(
            [Feature("rc564_pandas_feature")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC564_PANDAS,
        ).get_rows()

        assert rows == [{"rc564_pandas_feature": 1}, {"rc564_pandas_feature": 2}]

    def test_python_dict_result_returns_flat_row_dicts(self) -> None:
        rows = mlodaAPI.run_all(
            ["rc564_dict_feature"],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_RC564_DICT,
        ).get_rows()

        assert rows == [{"rc564_dict_feature": 1}, {"rc564_dict_feature": 2}]

    def test_pyarrow_result_returns_flat_row_dicts(self) -> None:
        rows = mlodaAPI.run_all(
            [Feature("rc564_arrow_feature")],
            compute_frameworks=["PyArrowTable"],
            plugin_collector=_RC564_ARROW,
        ).get_rows()

        assert rows == [{"rc564_arrow_feature": 1}, {"rc564_arrow_feature": 2}]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_result_returns_flat_row_dicts(self) -> None:
        rows = mlodaAPI.run_all(
            [Feature("rc564_polars_feature")],
            compute_frameworks=["PolarsDataFrame"],
            plugin_collector=_RC564_POLARS,
        ).get_rows()

        assert rows == [{"rc564_polars_feature": 1}, {"rc564_polars_feature": 2}]


class TestRunAllGetValues:
    """``run_all(...).get_values(name)`` yields one column as a plain Python list."""

    def test_pandas_result_returns_plain_column_values(self) -> None:
        values = mlodaAPI.run_all(
            [Feature("rc568_pd_left")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_SINGLE,
        ).get_values("rc568_pd_left")

        assert values == [1, 2, 3]
        assert type(values) is list


class TestRunAllGetDf:
    """``run_all(...).get_df()`` concatenates all results into ONE DataFrame."""

    def test_pandas_two_groups_concat_into_one_frame(self) -> None:
        result = mlodaAPI.run_all(
            [Feature("rc568_pd_left"), Feature("rc568_pd_right")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_PAIR,
        ).get_df()

        assert isinstance(result, pd.DataFrame)
        assert {"rc568_pd_left", "rc568_pd_right"} <= set(result.columns)
        assert len(result) == 3
        assert list(result["rc568_pd_left"]) == [1, 2, 3]
        assert list(result["rc568_pd_right"]) == [10, 20, 30]

    def test_pandas_single_group_returns_single_frame(self) -> None:
        result = mlodaAPI.run_all(
            [Feature("rc568_pd_left")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_SINGLE,
        ).get_df()

        assert isinstance(result, pd.DataFrame)
        assert "rc568_pd_left" in result.columns
        assert list(result["rc568_pd_left"]) == [1, 2, 3]

    @pytest.mark.skipif(pl is None, reason="polars is not installed")
    def test_polars_two_groups_concat_into_one_frame(self) -> None:
        result = mlodaAPI.run_all(
            [Feature("rc568_pl_left"), Feature("rc568_pl_right")],
            compute_frameworks=["PolarsDataFrame"],
            plugin_collector=_RC568_POLARS_PAIR,
        ).get_df()

        assert isinstance(result, pl.DataFrame)
        assert {"rc568_pl_left", "rc568_pl_right"} <= set(result.columns)
        assert result.height == 3
        assert result["rc568_pl_left"].to_list() == [1, 2, 3]
        assert result["rc568_pl_right"].to_list() == [10, 20, 30]

    def test_non_dataframe_results_raise_value_error(self) -> None:
        results = mlodaAPI.run_all(
            [Feature("rc568_dict_feature")],
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_RC568_DICT,
        )

        with pytest.raises(ValueError, match="DataFrame"):
            results.get_df()

    def test_row_count_mismatch_raises_value_error(self) -> None:
        results = mlodaAPI.run_all(
            [Feature("rc568_pd_left"), Feature("rc568_pd_short")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_MISMATCH,
        )

        with pytest.raises(ValueError, match="row"):
            results.get_df()
