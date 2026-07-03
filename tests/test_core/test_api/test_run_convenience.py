"""Failing tests for the mlodaAPI convenience classmethods (issues #564 and #568).

``mlodaAPI.run_one`` (issue #564): run a SINGLE feature and return a flat list of
row dicts, regardless of the compute framework that produced the result.

``mlodaAPI.run_all_as_dataframe`` (issue #568): run like ``run_all`` and concatenate
the per-feature-group results horizontally into ONE DataFrame (pandas or polars).

Both methods do not exist yet; every test in this module must fail with
``AttributeError`` until the Green agent implements them.

All FeatureGroups are module-local with unique ``rc564_`` / ``rc568_`` feature names
and each run is pinned to its own ``PluginCollector`` so the tests stay deterministic
and parallel-safe under pytest-xdist.
"""

from typing import Any, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginCollector, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.polars.dataframe import PolarsDataFrame  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable  # noqa: F401
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (  # noqa: F401
    PythonDictFramework,
)


class Rc564PandasFeatureGroup(FeatureGroup):
    """Root FG producing a single pandas column for the run_one tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_pandas_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"rc564_pandas_feature": [1, 2]})


class Rc564DictFeatureGroup(FeatureGroup):
    """Root FG producing a single columnar-dict column for the run_one tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_dict_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return {"rc564_dict_feature": [1, 2]}


class Rc564ArrowFeatureGroup(FeatureGroup):
    """Root FG producing a single pyarrow column for the run_one tests."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({"rc564_arrow_feature"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pa.table({"rc564_arrow_feature": [1, 2]})


class Rc564PolarsFeatureGroup(FeatureGroup):
    """Root FG producing a single polars column for the run_one tests."""

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
    """Columnar-dict FG proving run_all_as_dataframe rejects non-DataFrame results."""

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


class TestRunOne:
    """Contract for ``mlodaAPI.run_one`` (issue #564)."""

    def test_run_one_pandas_returns_flat_row_dicts(self) -> None:
        """A pandas-backed single feature comes back as a flat list of row dicts."""
        result = mlodaAPI.run_one(
            Feature("rc564_pandas_feature"),
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC564_PANDAS,
        )

        assert result == [{"rc564_pandas_feature": 1}, {"rc564_pandas_feature": 2}]

    def test_run_one_python_dict_returns_flat_row_dicts(self) -> None:
        """A PythonDict-backed single feature (given as str) yields identical flat rows."""
        result = mlodaAPI.run_one(
            "rc564_dict_feature",
            compute_frameworks=["PythonDictFramework"],
            plugin_collector=_RC564_DICT,
        )

        assert result == [{"rc564_dict_feature": 1}, {"rc564_dict_feature": 2}]

    def test_run_one_pyarrow_returns_flat_row_dicts(self) -> None:
        """A pyarrow-backed single feature comes back as a flat list of row dicts."""
        result = mlodaAPI.run_one(
            Feature("rc564_arrow_feature"),
            compute_frameworks=["PyArrowTable"],
            plugin_collector=_RC564_ARROW,
        )

        assert result == [{"rc564_arrow_feature": 1}, {"rc564_arrow_feature": 2}]

    def test_run_one_polars_returns_flat_row_dicts(self) -> None:
        """A polars-backed single feature comes back as a flat list of row dicts."""
        result = mlodaAPI.run_one(
            Feature("rc564_polars_feature"),
            compute_frameworks=["PolarsDataFrame"],
            plugin_collector=_RC564_POLARS,
        )

        assert result == [{"rc564_polars_feature": 1}, {"rc564_polars_feature": 2}]

    def test_run_one_rejects_list_input(self) -> None:
        """Passing a list instead of a single feature raises a clear ValueError."""
        with pytest.raises(ValueError, match="single"):
            mlodaAPI.run_one(
                [Feature("rc564_pandas_feature")],  # type: ignore[arg-type]
                compute_frameworks=["PandasDataFrame"],
                plugin_collector=_RC564_PANDAS,
            )


class TestRunAllAsDataframe:
    """Contract for ``mlodaAPI.run_all_as_dataframe`` (issue #568)."""

    def test_pandas_two_groups_concat_into_one_frame(self) -> None:
        """Two pandas feature-group results are horizontally concatenated into ONE frame."""
        result = mlodaAPI.run_all_as_dataframe(
            [Feature("rc568_pd_left"), Feature("rc568_pd_right")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_PAIR,
        )

        assert isinstance(result, pd.DataFrame)
        assert {"rc568_pd_left", "rc568_pd_right"} <= set(result.columns)
        assert len(result) == 3
        assert list(result["rc568_pd_left"]) == [1, 2, 3]
        assert list(result["rc568_pd_right"]) == [10, 20, 30]

    def test_pandas_single_group_returns_single_frame(self) -> None:
        """A single-group run returns that single frame with the requested column."""
        result = mlodaAPI.run_all_as_dataframe(
            [Feature("rc568_pd_left")],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=_RC568_PANDAS_SINGLE,
        )

        assert isinstance(result, pd.DataFrame)
        assert "rc568_pd_left" in result.columns
        assert list(result["rc568_pd_left"]) == [1, 2, 3]

    def test_polars_two_groups_concat_into_one_frame(self) -> None:
        """Two polars feature-group results are horizontally concatenated into ONE frame."""
        result = mlodaAPI.run_all_as_dataframe(
            [Feature("rc568_pl_left"), Feature("rc568_pl_right")],
            compute_frameworks=["PolarsDataFrame"],
            plugin_collector=_RC568_POLARS_PAIR,
        )

        assert isinstance(result, pl.DataFrame)
        assert {"rc568_pl_left", "rc568_pl_right"} <= set(result.columns)
        assert result.height == 3
        assert result["rc568_pl_left"].to_list() == [1, 2, 3]
        assert result["rc568_pl_right"].to_list() == [10, 20, 30]

    def test_non_dataframe_results_raise_value_error(self) -> None:
        """PythonDict columnar-dict results cannot be concatenated and raise ValueError."""
        with pytest.raises(ValueError, match="DataFrame"):
            mlodaAPI.run_all_as_dataframe(
                [Feature("rc568_dict_feature")],
                compute_frameworks=["PythonDictFramework"],
                plugin_collector=_RC568_DICT,
            )

    def test_row_count_mismatch_raises_value_error(self) -> None:
        """Two groups with different row counts (3 vs 2) raise ValueError on concat."""
        with pytest.raises(ValueError, match="row"):
            mlodaAPI.run_all_as_dataframe(
                [Feature("rc568_pd_left"), Feature("rc568_pd_short")],
                compute_frameworks=["PandasDataFrame"],
                plugin_collector=_RC568_PANDAS_MISMATCH,
            )
