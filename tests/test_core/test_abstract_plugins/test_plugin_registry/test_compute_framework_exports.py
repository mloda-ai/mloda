"""Compute-framework export matrix for mloda.user (issue #649).

Contract: the 9 concrete ComputeFramework classes are importable directly from
mloda.user (e.g. ``from mloda.user import PythonDictFramework``) via PEP 562
lazy ``__getattr__``, while ``import mloda.user`` stays dependency-free. They are
deliberately kept OUT of mloda.user.__all__ so ``from mloda.user import *`` stays
dependency-free in minimal installs, but they ARE surfaced by ``__dir__`` for
discoverability. Each resolves to the identical object defined in its canonical
mloda_plugins source module; the deep-path imports keep working; and an unknown
attribute on mloda.user still raises AttributeError.
"""

import importlib
from types import ModuleType

import pytest

import mloda.provider
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
    def test_symbol_excluded_from_all_but_discoverable_via_dir(self, symbol: str, source_module: str) -> None:
        # Kept out of __all__ so wildcard import stays dependency-free, but surfaced by __dir__.
        assert symbol not in mloda.user.__all__, (
            f"mloda.user.__all__ must NOT list '{symbol}' (keeps wildcard import dependency-free)"
        )
        assert symbol in dir(mloda.user), f"mloda.user.__dir__ must surface '{symbol}' for discoverability"

    @pytest.mark.parametrize(("symbol", "source_module"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_symbol_is_identical_to_source_object(self, symbol: str, source_module: str) -> None:
        source = _source_module(source_module)
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert hasattr(mloda.user, symbol), f"mloda.user must expose '{symbol}'"
        assert getattr(mloda.user, symbol) is getattr(source, symbol), (
            f"mloda.user.{symbol} must be the identical object from {source.__name__}"
        )

    @pytest.mark.parametrize(("symbol", "source_module"), EXPORT_MATRIX, ids=_MATRIX_IDS)
    def test_deep_path_import_yields_same_object(self, symbol: str, source_module: str) -> None:
        # Independently import via the real deep path (mloda_plugins...<module>) and
        # assert identity with the lazily-resolved mloda.user attribute (no breaking change).
        deep_path_module = importlib.import_module(source_module)
        deep_path_object = getattr(deep_path_module, symbol)
        assert deep_path_object is getattr(mloda.user, symbol), (
            f"deep-path {source_module}.{symbol} must be the identical object as mloda.user.{symbol}"
        )


class TestUnknownAttributeStillRaises:
    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            mloda.user.DefinitelyNotAnExportedSymbol


# python_dict helper functions follow the same lazy-export contract as the framework
# classes: importable from mloda.user, identical to the canonical source object,
# out of __all__, surfaced by __dir__.
_PYTHON_DICT_UTILS_MODULE = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils"

HELPER_EXPORT_MATRIX: list[tuple[str, str]] = [
    ("columnar_to_rows", _PYTHON_DICT_UTILS_MODULE),
    ("homogenize_rows", _PYTHON_DICT_UTILS_MODULE),
    ("is_columnar", _PYTHON_DICT_UTILS_MODULE),
    ("rows_to_columnar", _PYTHON_DICT_UTILS_MODULE),
]

_HELPER_MATRIX_IDS = [symbol for symbol, _source in HELPER_EXPORT_MATRIX]


class TestPythonDictHelperExportMatrix:
    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_helper_excluded_from_all_but_discoverable_via_dir(self, symbol: str, source_module: str) -> None:
        assert symbol not in mloda.user.__all__, (
            f"mloda.user.__all__ must NOT list '{symbol}' (keeps wildcard import dependency-free)"
        )
        assert symbol in dir(mloda.user), f"mloda.user.__dir__ must surface '{symbol}' for discoverability"

    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_helper_is_identical_to_source_object(self, symbol: str, source_module: str) -> None:
        source = _source_module(source_module)
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert hasattr(mloda.user, symbol), f"mloda.user must expose '{symbol}'"
        assert getattr(mloda.user, symbol) is getattr(source, symbol), (
            f"mloda.user.{symbol} must be the identical object from {source.__name__}"
        )


# Provider side (issue #707): pivoting a columnar frame back to rows is a provider-side
# concern, so the same four helpers are ALSO exported from mloda.provider. Unlike the
# lazy mloda.user exports, python_dict_utils is stdlib-only and mloda.provider already
# imports eagerly from mloda_plugins, so these are eager exports listed in __all__.
class TestPythonDictHelperProviderExportMatrix:
    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_helper_importable_from_provider(self, symbol: str, source_module: str) -> None:
        assert hasattr(mloda.provider, symbol), f"mloda.provider must expose '{symbol}'"

    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_helper_listed_in_provider_all(self, symbol: str, source_module: str) -> None:
        assert symbol in mloda.provider.__all__, f"mloda.provider.__all__ must list '{symbol}' (eager export)"

    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_provider_helper_is_identical_to_source_object(self, symbol: str, source_module: str) -> None:
        source = _source_module(source_module)
        assert hasattr(source, symbol), f"{source.__name__} must define '{symbol}'"
        assert getattr(mloda.provider, symbol) is getattr(source, symbol), (
            f"mloda.provider.{symbol} must be the identical object from {source.__name__}"
        )

    @pytest.mark.parametrize(("symbol", "source_module"), HELPER_EXPORT_MATRIX, ids=_HELPER_MATRIX_IDS)
    def test_provider_and_user_surfaces_agree(self, symbol: str, source_module: str) -> None:
        assert getattr(mloda.provider, symbol) is getattr(mloda.user, symbol), (
            f"mloda.provider.{symbol} and mloda.user.{symbol} must be the identical object"
        )
