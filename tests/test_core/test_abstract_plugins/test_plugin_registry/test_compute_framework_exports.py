"""Compute-framework export matrix for mloda.user (issue #649).

Contract: the 9 concrete ComputeFramework classes must be importable directly
from mloda.user (e.g. ``from mloda.user import PythonDictFramework``) while
``import mloda.user`` stays dependency-free. Each symbol is listed in
mloda.user.__all__ and is the identical object defined in its canonical
mloda_plugins source module; the old deep-path imports keep working; and an
unknown attribute on mloda.user still raises AttributeError.
"""

import importlib
from types import ModuleType

import pytest

import mloda.user

# Source modules are resolved lazily so a missing optional backend fails only its own matrix row.
# (symbol exported by mloda.user, canonical source module holding the class)
EXPORT_MATRIX: list[tuple[str, str]] = [
    ("PythonDictFramework", "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework"),
    ("PandasDataFrame", "mloda_plugins.compute_framework.base_implementations.pandas.dataframe"),
    ("PolarsDataFrame", "mloda_plugins.compute_framework.base_implementations.polars.dataframe"),
    ("PolarsLazyDataFrame", "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe"),
    ("PyArrowTable", "mloda_plugins.compute_framework.base_implementations.pyarrow.table"),
    ("SqliteFramework", "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework"),
    ("DuckDBFramework", "mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework"),
    ("SparkFramework", "mloda_plugins.compute_framework.base_implementations.spark.spark_framework"),
    ("IcebergFramework", "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework"),
]

_MATRIX_IDS = [symbol for symbol, _source in EXPORT_MATRIX]


def _source_module(source_module: str) -> ModuleType:
    return importlib.import_module(source_module)


class TestComputeFrameworkExportMatrix:
    @pytest.mark.parametrize(("symbol", "source_module"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_listed_in_user_all(self, symbol: str, source_module: str) -> None:
        assert symbol in mloda.user.__all__, f"mloda.user must list '{symbol}' in __all__"

    @pytest.mark.parametrize(("symbol", "source_module"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_is_identical_to_source_object(self, symbol: str, source_module: str) -> None:
        source = _source_module(source_module)
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert hasattr(mloda.user, symbol), f"mloda.user must expose '{symbol}'"
        assert getattr(mloda.user, symbol) is getattr(source, symbol), (
            f"mloda.user.{symbol} must be the identical object from {source.__name__}"
        )

    @pytest.mark.parametrize(("symbol", "source_module"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_deep_path_import_still_works(self, symbol: str, source_module: str) -> None:
        source = _source_module(source_module)
        assert getattr(source, symbol) is getattr(mloda.user, symbol), (
            f"deep-path {source.__name__}.{symbol} must stay the identical object as mloda.user.{symbol}"
        )


class TestUnknownAttributeStillRaises:
    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            mloda.user.DefinitelyNotAnExportedSymbol  # type: ignore[attr-defined]
