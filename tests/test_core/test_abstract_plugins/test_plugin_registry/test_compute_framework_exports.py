"""Export contract for the mloda.user and mloda.provider facades (issues #649, #707, #716).

mloda.user resolves the 9 concrete ComputeFramework classes and the python_dict helpers via
PEP 562 lazy ``__getattr__``: they stay OUT of ``__all__`` so ``from mloda.user import *``
remains dependency-free in minimal installs, but ``__dir__`` surfaces them.

mloda.provider exports the python_dict symbols (PythonDictFramework plus the columnar
helpers) EAGERLY and lists them in ``__all__``. The plugin modules import their base classes
from the defining modules instead of the facades, so there is no import cycle to work around
and mloda.provider must not define a module-level ``__getattr__``.

Every symbol resolves to the identical object defined in its canonical mloda_plugins source
module, deep-path imports keep working, and unknown attributes still raise AttributeError.
"""

import importlib
import subprocess  # nosec B404
import sys
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
    ("result_rows", _PYTHON_DICT_UTILS_MODULE),
    ("rows_to_columnar", _PYTHON_DICT_UTILS_MODULE),
    # Issue #716: the columnar surface is only complete with the shape check and the row count.
    ("row_count", _PYTHON_DICT_UTILS_MODULE),
    ("validate_columnar_dict", _PYTHON_DICT_UTILS_MODULE),
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
# concern, so the same helpers are ALSO exported from mloda.provider. Unlike the lazy
# mloda.user exports, python_dict_utils is stdlib-only and mloda.provider already imports
# eagerly from mloda_plugins, so these are eager exports listed in __all__.
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


# PythonDictFramework on the provider surface (issue #716): plugin authors reach the framework
# class from the short namespace instead of the deep mloda_plugins path. The python_dict plugin
# modules import their base classes from the DEFINING modules (not from the mloda.provider /
# mloda.user facades), so there is no cycle and the export is eager, exactly like the helpers.
_PYTHON_DICT_FRAMEWORK_MODULE = "mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework"
_UNKNOWN_SYMBOL = "DefinitelyNotAnExportedSymbol"


class TestPythonDictFrameworkProviderExport:
    def test_framework_is_identical_to_source_object(self) -> None:
        source = _source_module(_PYTHON_DICT_FRAMEWORK_MODULE)
        assert hasattr(mloda.provider, "PythonDictFramework"), "mloda.provider must expose 'PythonDictFramework'"
        assert getattr(mloda.provider, "PythonDictFramework") is source.PythonDictFramework, (
            f"mloda.provider.PythonDictFramework must be the identical object from {_PYTHON_DICT_FRAMEWORK_MODULE}"
        )

    def test_framework_listed_in_provider_all(self) -> None:
        assert "PythonDictFramework" in mloda.provider.__all__, (
            "mloda.provider.__all__ must list 'PythonDictFramework' (eager export, stdlib-only plugin module)"
        )

    def test_provider_and_user_framework_surfaces_agree(self) -> None:
        assert getattr(mloda.provider, "PythonDictFramework") is getattr(mloda.user, "PythonDictFramework"), (
            "mloda.provider.PythonDictFramework and mloda.user.PythonDictFramework must be the identical object"
        )

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            getattr(mloda.provider, _UNKNOWN_SYMBOL)


class TestProviderHasNoModuleLevelGetattr:
    def test_provider_does_not_define_module_getattr(self) -> None:
        """A module-level __getattr__ on mloda.provider would type-check every typo as Any."""
        assert "__getattr__" not in vars(mloda.provider), (
            "mloda.provider must NOT define a module-level __getattr__: mypy honors it as a catch-all, so "
            "every unknown attribute on the module type-checks as Any and the typo guard silently disappears "
            "for the whole provider surface ('from mloda.provider import FeatureGropu' would pass mypy --strict "
            "and only fail at runtime). Export eagerly instead; the import cycle is fixed at its source, where "
            "the python_dict plugin modules import their base classes from the defining modules, not the facades."
        )


# (id, statements executed in order in a fresh interpreter)
_IMPORT_ORDERS: list[tuple[str, str]] = [
    (
        "plugin_module_first",
        f"import {_PYTHON_DICT_FRAMEWORK_MODULE} as deep\nimport mloda.provider\n",
    ),
    (
        "provider_first",
        f"import mloda.provider\nimport {_PYTHON_DICT_FRAMEWORK_MODULE} as deep\n",
    ),
]

_IMPORT_ORDER_IDS = [order_id for order_id, _source in _IMPORT_ORDERS]


class TestProviderImportOrderCycleGuard:
    @pytest.mark.parametrize(("order_id", "imports"), _IMPORT_ORDERS, ids=_IMPORT_ORDER_IDS)
    def test_both_import_orders_resolve_the_same_class(self, order_id: str, imports: str) -> None:
        """Neither import order may raise ImportError on a mloda.provider <-> python_dict cycle."""
        script = (
            f"{imports}"
            "assert mloda.provider.PythonDictFramework is deep.PythonDictFramework, 'not the same class object'\n"
            "print('ok')\n"
        )
        # Safe: fixed argv (sys.executable + a script built from module-level constants), no shell, no user input.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Import order '{order_id}' failed in a fresh interpreter. mloda.provider imports "
            f"PythonDictFramework eagerly, so the python_dict plugin modules must import their base classes "
            f"from the defining modules; a facade import there re-enters a partially initialized "
            f"mloda.provider and raises ImportError.\nstderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"
