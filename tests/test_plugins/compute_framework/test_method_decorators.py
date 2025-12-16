"""Tests verifying that framework helper methods use @classmethod instead of @staticmethod.

These tests ensure that methods accessing module-level imports are implemented as
@classmethod for semantic correctness and to enable proper subclass customization.
"""

import inspect
from typing import Any, Optional, Type

import pytest

# Import all framework classes
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_merge_engine import PandasMergeEngine

# Optional imports for polars
_pl: Any = None
_PolarsDataFrame: Optional[Type[Any]] = None
_PolarsLazyDataFrame: Optional[Type[Any]] = None
try:
    import polars as _pl
    from mloda_plugins.compute_framework.base_implementations.polars.dataframe import (
        PolarsDataFrame as _PolarsDataFrame,
    )
    from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import (
        PolarsLazyDataFrame as _PolarsLazyDataFrame,
    )
except ImportError:
    pass

# Optional imports for duckdb
_duckdb: Any = None
_DuckDBFramework: Optional[Type[Any]] = None
try:
    import duckdb as _duckdb
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import (
        DuckDBFramework as _DuckDBFramework,
    )
except ImportError:
    pass

# Optional imports for spark
_SparkSession: Any = None
_SparkFramework: Optional[Type[Any]] = None
try:
    from pyspark.sql import SparkSession as _SparkSessionImport
    from mloda_plugins.compute_framework.base_implementations.spark.spark_framework import (
        SparkFramework as _SparkFrameworkImport,
    )

    _SparkSession = _SparkSessionImport
    _SparkFramework = _SparkFrameworkImport
except ImportError:
    pass


class TestPandasDataFrameClassMethods:
    """Test that PandasDataFrame helper methods are classmethods."""

    def test_pd_dataframe_is_classmethod(self) -> None:
        """Verify pd_dataframe is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(PandasDataFrame, "pd_dataframe")
        assert inspect.ismethod(method), "pd_dataframe should be a classmethod (bound method on class)"
        # Verify it's bound to the class, not an instance
        assert method.__self__ is PandasDataFrame, "pd_dataframe should be bound to the PandasDataFrame class"

    def test_pd_series_is_classmethod(self) -> None:
        """Verify pd_series is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(PandasDataFrame, "pd_series")
        assert inspect.ismethod(method), "pd_series should be a classmethod (bound method on class)"
        assert method.__self__ is PandasDataFrame, "pd_series should be bound to the PandasDataFrame class"

    def test_pd_merge_is_classmethod(self) -> None:
        """Verify pd_merge is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(PandasDataFrame, "pd_merge")
        assert inspect.ismethod(method), "pd_merge should be a classmethod (bound method on class)"
        assert method.__self__ is PandasDataFrame, "pd_merge should be bound to the PandasDataFrame class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        # These should work without instantiation
        import pandas as pd

        assert PandasDataFrame.pd_dataframe() == pd.DataFrame
        assert PandasDataFrame.pd_series() == pd.Series
        assert PandasDataFrame.pd_merge() == pd.merge

    def test_methods_callable_on_instance(self) -> None:
        """Verify methods can be called on instances."""
        from mloda.user import ParallelizationMode

        import pandas as pd

        instance = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.pd_dataframe() == pd.DataFrame
        assert instance.pd_series() == pd.Series
        assert instance.pd_merge() == pd.merge


class TestPandasMergeEngineClassMethods:
    """Test that PandasMergeEngine helper methods are classmethods."""

    def test_pd_merge_is_classmethod(self) -> None:
        """Verify pd_merge is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(PandasMergeEngine, "pd_merge")
        assert inspect.ismethod(method), "pd_merge should be a classmethod (bound method on class)"
        assert method.__self__ is PandasMergeEngine, "pd_merge should be bound to the PandasMergeEngine class"

    def test_pd_concat_is_classmethod(self) -> None:
        """Verify pd_concat is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(PandasMergeEngine, "pd_concat")
        assert inspect.ismethod(method), "pd_concat should be a classmethod (bound method on class)"
        assert method.__self__ is PandasMergeEngine, "pd_concat should be bound to the PandasMergeEngine class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        import pandas as pd

        assert PandasMergeEngine.pd_merge() == pd.merge
        assert PandasMergeEngine.pd_concat() == pd.concat


@pytest.mark.skipif(_pl is None, reason="Polars is not installed")
class TestPolarsDataFrameClassMethods:
    """Test that PolarsDataFrame helper methods are classmethods."""

    def test_pl_dataframe_is_classmethod(self) -> None:
        """Verify pl_dataframe is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_PolarsDataFrame, "pl_dataframe")
        assert inspect.ismethod(method), "pl_dataframe should be a classmethod (bound method on class)"
        assert method.__self__ is _PolarsDataFrame, "pl_dataframe should be bound to the PolarsDataFrame class"

    def test_pl_series_is_classmethod(self) -> None:
        """Verify pl_series is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_PolarsDataFrame, "pl_series")
        assert inspect.ismethod(method), "pl_series should be a classmethod (bound method on class)"
        assert method.__self__ is _PolarsDataFrame, "pl_series should be bound to the PolarsDataFrame class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        assert _PolarsDataFrame is not None
        assert _PolarsDataFrame.pl_dataframe() == _pl.DataFrame
        assert _PolarsDataFrame.pl_series() == _pl.Series

    def test_methods_callable_on_instance(self) -> None:
        """Verify methods can be called on instances."""
        from mloda.user import ParallelizationMode

        assert _PolarsDataFrame is not None
        instance = _PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.pl_dataframe() == _pl.DataFrame
        assert instance.pl_series() == _pl.Series


@pytest.mark.skipif(_pl is None, reason="Polars is not installed")
class TestPolarsLazyDataFrameClassMethods:
    """Test that PolarsLazyDataFrame helper methods are classmethods."""

    def test_pl_lazy_frame_is_classmethod(self) -> None:
        """Verify pl_lazy_frame is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_PolarsLazyDataFrame, "pl_lazy_frame")
        assert inspect.ismethod(method), "pl_lazy_frame should be a classmethod (bound method on class)"
        assert method.__self__ is _PolarsLazyDataFrame, "pl_lazy_frame should be bound to PolarsLazyDataFrame class"

    def test_pl_dataframe_is_classmethod(self) -> None:
        """Verify pl_dataframe is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_PolarsLazyDataFrame, "pl_dataframe")
        assert inspect.ismethod(method), "pl_dataframe should be a classmethod (bound method on class)"
        assert method.__self__ is _PolarsLazyDataFrame, "pl_dataframe should be bound to PolarsLazyDataFrame class"

    def test_pl_series_is_classmethod(self) -> None:
        """Verify pl_series is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_PolarsLazyDataFrame, "pl_series")
        assert inspect.ismethod(method), "pl_series should be a classmethod (bound method on class)"
        assert method.__self__ is _PolarsLazyDataFrame, "pl_series should be bound to PolarsLazyDataFrame class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        assert _PolarsLazyDataFrame is not None
        assert _PolarsLazyDataFrame.pl_lazy_frame() == _pl.LazyFrame
        assert _PolarsLazyDataFrame.pl_dataframe() == _pl.DataFrame
        assert _PolarsLazyDataFrame.pl_series() == _pl.Series

    def test_inheritance_from_polars_dataframe(self) -> None:
        """Verify that PolarsLazyDataFrame inherits from PolarsDataFrame and methods work correctly."""
        assert _PolarsDataFrame is not None
        assert _PolarsLazyDataFrame is not None
        assert issubclass(_PolarsLazyDataFrame, _PolarsDataFrame), (
            "PolarsLazyDataFrame should inherit from PolarsDataFrame"
        )

        # The inherited methods should be bound to PolarsLazyDataFrame, not the parent
        method = getattr(_PolarsLazyDataFrame, "pl_dataframe")
        assert method.__self__ is _PolarsLazyDataFrame, "Inherited classmethod should be bound to the subclass"


@pytest.mark.skipif(_duckdb is None, reason="DuckDB is not installed")
class TestDuckDBFrameworkClassMethods:
    """Test that DuckDBFramework helper methods are classmethods."""

    def test_duckdb_relation_is_classmethod(self) -> None:
        """Verify duckdb_relation is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_DuckDBFramework, "duckdb_relation")
        assert inspect.ismethod(method), "duckdb_relation should be a classmethod (bound method on class)"
        assert method.__self__ is _DuckDBFramework, "duckdb_relation should be bound to the DuckDBFramework class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        assert _DuckDBFramework is not None
        assert _DuckDBFramework.duckdb_relation() == _duckdb.DuckDBPyRelation

    def test_methods_callable_on_instance(self) -> None:
        """Verify methods can be called on instances."""
        from mloda.user import ParallelizationMode

        assert _DuckDBFramework is not None
        instance = _DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.duckdb_relation() == _duckdb.DuckDBPyRelation


@pytest.mark.skipif(_SparkSession is None, reason="PySpark is not installed")
class TestSparkFrameworkClassMethods:
    """Test that SparkFramework helper methods are classmethods."""

    def test_spark_dataframe_is_classmethod(self) -> None:
        """Verify spark_dataframe is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_SparkFramework, "spark_dataframe")
        assert inspect.ismethod(method), "spark_dataframe should be a classmethod (bound method on class)"
        assert method.__self__ is _SparkFramework, "spark_dataframe should be bound to the SparkFramework class"

    def test_spark_session_is_classmethod(self) -> None:
        """Verify spark_session is a classmethod by checking it's bound when accessed from the class."""
        method = getattr(_SparkFramework, "spark_session")
        assert inspect.ismethod(method), "spark_session should be a classmethod (bound method on class)"
        assert method.__self__ is _SparkFramework, "spark_session should be bound to the SparkFramework class"

    def test_methods_callable_on_class(self) -> None:
        """Verify methods can be called on the class."""
        from pyspark.sql import DataFrame

        assert _SparkFramework is not None
        assert _SparkFramework.spark_dataframe() == DataFrame
        assert _SparkFramework.spark_session() == _SparkSession

    def test_methods_callable_on_instance(self) -> None:
        """Verify methods can be called on instances."""
        from mloda.user import ParallelizationMode
        from pyspark.sql import DataFrame

        assert _SparkFramework is not None
        instance = _SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.spark_dataframe() == DataFrame
        assert instance.spark_session() == _SparkSession


class TestPandasDataFrameEngineClassMethods:
    """Tests for engine factory methods that should be classmethods."""

    def test_merge_engine_is_classmethod(self) -> None:
        """merge_engine should be a classmethod since it doesn't use self."""
        method = getattr(PandasDataFrame, "merge_engine")
        assert inspect.ismethod(method), "merge_engine should be a classmethod"
        assert method.__self__ is PandasDataFrame, "merge_engine should be bound to the PandasDataFrame class"

    def test_filter_engine_is_classmethod(self) -> None:
        """filter_engine should be a classmethod since it doesn't use self."""
        method = getattr(PandasDataFrame, "filter_engine")
        assert inspect.ismethod(method), "filter_engine should be a classmethod"
        assert method.__self__ is PandasDataFrame, "filter_engine should be bound to the PandasDataFrame class"

    def test_expected_data_framework_is_classmethod(self) -> None:
        """expected_data_framework should be a classmethod for consistency."""
        method = getattr(PandasDataFrame, "expected_data_framework")
        assert inspect.ismethod(method), "expected_data_framework should be a classmethod"
        assert method.__self__ is PandasDataFrame, (
            "expected_data_framework should be bound to the PandasDataFrame class"
        )

    def test_engine_methods_callable_on_class(self) -> None:
        """Verify engine methods work when called on the class."""
        from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

        # These should work without instantiation
        assert PandasDataFrame.merge_engine() == PandasMergeEngine
        assert PandasDataFrame.filter_engine() == PandasFilterEngine

    def test_engine_methods_callable_on_instance(self) -> None:
        """Verify engine methods work when called on instances."""
        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.pandas.pandas_filter_engine import PandasFilterEngine

        instance = PandasDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.merge_engine() == PandasMergeEngine
        assert instance.filter_engine() == PandasFilterEngine


@pytest.mark.skipif(_pl is None, reason="Polars is not installed")
class TestPolarsDataFrameEngineClassMethods:
    """Tests for Polars engine factory methods that should be classmethods."""

    def test_merge_engine_is_classmethod(self) -> None:
        """merge_engine should be a classmethod since it doesn't use self."""
        method = getattr(_PolarsDataFrame, "merge_engine")
        assert inspect.ismethod(method), "merge_engine should be a classmethod"
        assert method.__self__ is _PolarsDataFrame, "merge_engine should be bound to the PolarsDataFrame class"

    def test_filter_engine_is_classmethod(self) -> None:
        """filter_engine should be a classmethod since it doesn't use self."""
        method = getattr(_PolarsDataFrame, "filter_engine")
        assert inspect.ismethod(method), "filter_engine should be a classmethod"
        assert method.__self__ is _PolarsDataFrame, "filter_engine should be bound to the PolarsDataFrame class"

    def test_expected_data_framework_is_classmethod(self) -> None:
        """expected_data_framework should be a classmethod for consistency."""
        method = getattr(_PolarsDataFrame, "expected_data_framework")
        assert inspect.ismethod(method), "expected_data_framework should be a classmethod"
        assert method.__self__ is _PolarsDataFrame, (
            "expected_data_framework should be bound to the PolarsDataFrame class"
        )

    def test_engine_methods_callable_on_class(self) -> None:
        """Verify engine methods work when called on the class."""
        from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
        from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

        assert _PolarsDataFrame is not None
        assert _PolarsDataFrame.merge_engine() == PolarsMergeEngine
        assert _PolarsDataFrame.filter_engine() == PolarsFilterEngine

    def test_engine_methods_callable_on_instance(self) -> None:
        """Verify engine methods work when called on instances."""
        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.polars.polars_merge_engine import PolarsMergeEngine
        from mloda_plugins.compute_framework.base_implementations.polars.polars_filter_engine import PolarsFilterEngine

        assert _PolarsDataFrame is not None
        instance = _PolarsDataFrame(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.merge_engine() == PolarsMergeEngine
        assert instance.filter_engine() == PolarsFilterEngine


@pytest.mark.skipif(_duckdb is None, reason="DuckDB is not installed")
class TestDuckDBFrameworkEngineClassMethods:
    """Tests for DuckDB engine factory methods that should be classmethods."""

    def test_merge_engine_is_classmethod(self) -> None:
        """merge_engine should be a classmethod since it doesn't use self."""
        method = getattr(_DuckDBFramework, "merge_engine")
        assert inspect.ismethod(method), "merge_engine should be a classmethod"
        assert method.__self__ is _DuckDBFramework, "merge_engine should be bound to the DuckDBFramework class"

    def test_filter_engine_is_classmethod(self) -> None:
        """filter_engine should be a classmethod since it doesn't use self."""
        method = getattr(_DuckDBFramework, "filter_engine")
        assert inspect.ismethod(method), "filter_engine should be a classmethod"
        assert method.__self__ is _DuckDBFramework, "filter_engine should be bound to the DuckDBFramework class"

    def test_expected_data_framework_is_classmethod(self) -> None:
        """expected_data_framework should be a classmethod for consistency."""
        method = getattr(_DuckDBFramework, "expected_data_framework")
        assert inspect.ismethod(method), "expected_data_framework should be a classmethod"
        assert method.__self__ is _DuckDBFramework, (
            "expected_data_framework should be bound to the DuckDBFramework class"
        )

    def test_engine_methods_callable_on_class(self) -> None:
        """Verify engine methods work when called on the class."""
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine

        assert _DuckDBFramework is not None
        assert _DuckDBFramework.merge_engine() == DuckDBMergeEngine
        assert _DuckDBFramework.filter_engine() == DuckDBFilterEngine

    def test_engine_methods_callable_on_instance(self) -> None:
        """Verify engine methods work when called on instances."""
        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_merge_engine import DuckDBMergeEngine
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_filter_engine import DuckDBFilterEngine

        assert _DuckDBFramework is not None
        instance = _DuckDBFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.merge_engine() == DuckDBMergeEngine
        assert instance.filter_engine() == DuckDBFilterEngine


@pytest.mark.skipif(_SparkSession is None, reason="PySpark is not installed")
class TestSparkFrameworkEngineClassMethods:
    """Tests for Spark engine factory methods that should be classmethods."""

    def test_merge_engine_is_classmethod(self) -> None:
        """merge_engine should be a classmethod since it doesn't use self."""
        method = getattr(_SparkFramework, "merge_engine")
        assert inspect.ismethod(method), "merge_engine should be a classmethod"
        assert method.__self__ is _SparkFramework, "merge_engine should be bound to the SparkFramework class"

    def test_filter_engine_is_classmethod(self) -> None:
        """filter_engine should be a classmethod since it doesn't use self."""
        method = getattr(_SparkFramework, "filter_engine")
        assert inspect.ismethod(method), "filter_engine should be a classmethod"
        assert method.__self__ is _SparkFramework, "filter_engine should be bound to the SparkFramework class"

    def test_expected_data_framework_is_classmethod(self) -> None:
        """expected_data_framework should be a classmethod for consistency."""
        method = getattr(_SparkFramework, "expected_data_framework")
        assert inspect.ismethod(method), "expected_data_framework should be a classmethod"
        assert method.__self__ is _SparkFramework, "expected_data_framework should be bound to SparkFramework class"

    def test_engine_methods_callable_on_class(self) -> None:
        """Verify engine methods work when called on the class."""
        from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import SparkMergeEngine
        from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine

        assert _SparkFramework is not None
        assert _SparkFramework.merge_engine() == SparkMergeEngine
        assert _SparkFramework.filter_engine() == SparkFilterEngine

    def test_engine_methods_callable_on_instance(self) -> None:
        """Verify engine methods work when called on instances."""
        from mloda.user import ParallelizationMode
        from mloda_plugins.compute_framework.base_implementations.spark.spark_merge_engine import SparkMergeEngine
        from mloda_plugins.compute_framework.base_implementations.spark.spark_filter_engine import SparkFilterEngine

        assert _SparkFramework is not None
        instance = _SparkFramework(mode=ParallelizationMode.SYNC, children_if_root=frozenset())
        assert instance.merge_engine() == SparkMergeEngine
        assert instance.filter_engine() == SparkFilterEngine
