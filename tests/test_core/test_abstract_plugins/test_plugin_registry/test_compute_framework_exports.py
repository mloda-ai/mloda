"""Facade export contract for mloda.user, mloda.provider and mloda.steward (issue #726: facades must
not import plugins).

The facades are plugin-free: none of them imports anything from mloda_plugins and none defines a
module-level ``__getattr__``. Importing one pulls in no mloda_plugins module and no optional
backend library.

The concrete ComputeFramework classes are published from one module per backend under
``mloda.user`` (``mloda.user.pandas``, ``mloda.user.python_dict``, ...). Each backend module
re-exports the identical objects from their canonical mloda_plugins source modules and lists
them in its own ``__all__``, every shipped framework is covered by exactly one such module, and
the plugin modules never import back from them. The eager provider exports of #707/#713/#716/#719/#721
and the lazy ``mloda.user`` exports of #649 are gone: the symbols are no longer attributes of any facade.

ApiInputDataFeature is core, not a plugin: it lives in mloda.core.abstract_plugins.components and
stays exported from mloda.provider. The old plugin module is deleted without a back-compat shim.
"""

import ast
import importlib
import inspect
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import mloda.provider
import mloda.user
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.user import PluginLoader

_PYTHON_DICT_PACKAGE = "mloda_plugins.compute_framework.base_implementations.python_dict"
_PYTHON_DICT_FRAMEWORK_MODULE = f"{_PYTHON_DICT_PACKAGE}.python_dict_framework"
_PYTHON_DICT_UTILS_MODULE = f"{_PYTHON_DICT_PACKAGE}.python_dict_utils"
_PANDAS_DATAFRAME_MODULE = "mloda_plugins.compute_framework.base_implementations.pandas.dataframe"

_FACADES: list[str] = ["mloda.provider", "mloda.steward", "mloda.user"]

_UNKNOWN_SYMBOL = "DefinitelyNotAnExportedSymbol"

# (backend module under mloda.user, exported symbol, canonical mloda_plugins source module)
#
# No row is skip-gated on its backend library (issue #736): every plugin module guards its
# module-level backend import, so importing the mloda.user module and asserting the export
# identity works with the library absent. A framework whose library is missing says so through
# is_available(). Skip-gating the rows would make them vacuous in envs that ship no pyspark.
BACKEND_EXPORT_MATRIX: list[tuple[str, str, str]] = [
    ("mloda.user.python_dict", "PythonDictFramework", _PYTHON_DICT_FRAMEWORK_MODULE),
    ("mloda.user.python_dict", "columnar_to_rows", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "homogenize_rows", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "is_columnar", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "result_rows", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "row_count", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "rows_to_columnar", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.python_dict", "validate_columnar_dict", _PYTHON_DICT_UTILS_MODULE),
    ("mloda.user.pandas", "PandasDataFrame", _PANDAS_DATAFRAME_MODULE),
    (
        "mloda.user.polars",
        "PolarsDataFrame",
        "mloda_plugins.compute_framework.base_implementations.polars.dataframe",
    ),
    (
        "mloda.user.polars",
        "PolarsLazyDataFrame",
        "mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe",
    ),
    (
        "mloda.user.pyarrow",
        "PyArrowTable",
        "mloda_plugins.compute_framework.base_implementations.pyarrow.table",
    ),
    (
        "mloda.user.duckdb",
        "DuckDBFramework",
        "mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework",
    ),
    (
        "mloda.user.sqlite",
        "SqliteFramework",
        "mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework",
    ),
    (
        "mloda.user.spark",
        "SparkFramework",
        "mloda_plugins.compute_framework.base_implementations.spark.spark_framework",
    ),
    (
        "mloda.user.iceberg",
        "IcebergFramework",
        "mloda_plugins.compute_framework.base_implementations.iceberg.iceberg_framework",
    ),
]

_BACKEND_MATRIX_IDS = [f"{backend}:{symbol}" for backend, symbol, _source in BACKEND_EXPORT_MATRIX]

# Every symbol that used to live on the facades themselves (#649 lazy user exports, #707/#716
# eager provider exports). None of them may be a facade attribute any more.
_FORMER_FACADE_SYMBOLS: list[str] = sorted({symbol for _backend, symbol, _source in BACKEND_EXPORT_MATRIX})


def _import_backend(backend_module: str) -> ModuleType:
    """Import a mloda.user backend module. No backend library may be needed for that."""
    return importlib.import_module(backend_module)


def _facade_source(facade: str) -> Path:
    """Resolve the facade's __init__.py through the imported module, not a hardcoded repo path."""
    module = importlib.import_module(facade)
    source = module.__file__
    assert source is not None, f"{facade} must be a file-backed module"
    return Path(source)


def _is_plugin_module(module_name: str) -> bool:
    return module_name == "mloda_plugins" or module_name.startswith("mloda_plugins.")


def _is_user_backend_module(module_name: str) -> bool:
    """True for a mloda.user.<backend> module; the bare mloda.user facade is not a backend module."""
    return module_name.startswith("mloda.user.")


def _imported_modules(source: Path) -> list[str]:
    """Every module imported anywhere in the source, TYPE_CHECKING blocks included."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported.append(node.module)
    return imported


def _plugin_imports(source: Path) -> list[str]:
    return [name for name in _imported_modules(source) if _is_plugin_module(name)]


def _user_backend_imports(source: Path) -> list[str]:
    return [name for name in _imported_modules(source) if _is_user_backend_module(name)]


class TestFacadesArePluginFree:
    @pytest.mark.parametrize("facade", _FACADES)
    def test_facade_source_has_no_mloda_plugins_import(self, facade: str) -> None:
        """AST check: a facade that imports a plugin drags the plugin tree into every core install."""
        offenders = _plugin_imports(_facade_source(facade))
        assert offenders == [], (
            f"{facade}/__init__.py must not import from mloda_plugins, but imports {sorted(set(offenders))}. "
            f"The facades are the public core surface; plugin classes are published from the per-backend "
            f"modules under mloda.user instead."
        )

    @pytest.mark.parametrize("facade", _FACADES)
    def test_facade_does_not_define_module_getattr(self, facade: str) -> None:
        """A module-level __getattr__ makes mypy type-check every typo on the facade as Any."""
        module = importlib.import_module(facade)
        assert "__getattr__" not in vars(module), (
            f"{facade} must NOT define a module-level __getattr__: mypy honors it as a catch-all, so every "
            f"unknown attribute type-checks as Any and the typo guard disappears for the whole surface. "
            f"Optional backends are reached through their own module (import mloda.user.pandas), which "
            f"imports whether or not the library is installed and reports it via is_available()."
        )

    @pytest.mark.parametrize("facade", _FACADES)
    def test_unknown_attribute_raises_attribute_error(self, facade: str) -> None:
        module = importlib.import_module(facade)
        with pytest.raises(AttributeError):
            getattr(module, _UNKNOWN_SYMBOL)


class TestBackendExportMatrix:
    @pytest.mark.parametrize(
        ("backend_module", "symbol", "source_module"), BACKEND_EXPORT_MATRIX, ids=_BACKEND_MATRIX_IDS
    )
    def test_symbol_is_identical_to_source_object(self, backend_module: str, symbol: str, source_module: str) -> None:
        backend = _import_backend(backend_module)
        source = importlib.import_module(source_module)
        assert hasattr(source, symbol), f"{source_module} must define '{symbol}'"
        assert hasattr(backend, symbol), f"{backend_module} must export '{symbol}'"
        assert getattr(backend, symbol) is getattr(source, symbol), (
            f"{backend_module}.{symbol} must be the identical object from {source_module}"
        )

    @pytest.mark.parametrize(
        ("backend_module", "symbol", "source_module"), BACKEND_EXPORT_MATRIX, ids=_BACKEND_MATRIX_IDS
    )
    def test_symbol_listed_in_backend_all(self, backend_module: str, symbol: str, source_module: str) -> None:
        backend = _import_backend(backend_module)
        assert hasattr(backend, "__all__"), f"{backend_module} must define __all__"
        assert symbol in backend.__all__, f"{backend_module}.__all__ must list '{symbol}'"


class TestFrameworkSymbolsAreNotOnTheFacades:
    @pytest.mark.parametrize("facade", _FACADES)
    @pytest.mark.parametrize("symbol", _FORMER_FACADE_SYMBOLS)
    def test_symbol_is_not_a_facade_attribute(self, facade: str, symbol: str) -> None:
        """The framework classes and python_dict helpers moved to the per-backend modules."""
        module = importlib.import_module(facade)
        assert not hasattr(module, symbol), (
            f"{facade} must NOT expose '{symbol}' any more: it is published from its mloda.user backend module "
            f"(the facades stay free of plugin imports)."
        )

    @pytest.mark.parametrize("facade", _FACADES)
    @pytest.mark.parametrize("symbol", _FORMER_FACADE_SYMBOLS)
    def test_symbol_is_not_listed_in_facade_all(self, facade: str, symbol: str) -> None:
        module = importlib.import_module(facade)
        assert symbol not in module.__all__, f"{facade}.__all__ must NOT list '{symbol}'"


_API_INPUT_DATA_FEATURE_MODULE = "mloda.core.abstract_plugins.components.input_data.api.api_input_data_feature"
_OLD_API_DATA_MODULE = "mloda_plugins.feature_group.input_data.api_data.api_data"


class TestApiInputDataFeatureIsCore:
    def test_provider_export_is_defined_in_core(self) -> None:
        """ApiInputDataFeature is what makes mloda.provider import a plugin today; it belongs to core."""
        assert hasattr(mloda.provider, "ApiInputDataFeature"), "mloda.provider must keep exporting ApiInputDataFeature"
        assert mloda.provider.ApiInputDataFeature.__module__.startswith("mloda.core"), (
            f"mloda.provider.ApiInputDataFeature must be defined under mloda.core, but its __module__ is "
            f"'{mloda.provider.ApiInputDataFeature.__module__}'"
        )

    def test_provider_export_is_identical_to_core_object(self) -> None:
        source = importlib.import_module(_API_INPUT_DATA_FEATURE_MODULE)
        assert mloda.provider.ApiInputDataFeature is source.ApiInputDataFeature, (
            f"mloda.provider.ApiInputDataFeature must be the identical object from {_API_INPUT_DATA_FEATURE_MODULE}"
        )

    def test_export_stays_listed_in_provider_all(self) -> None:
        assert "ApiInputDataFeature" in mloda.provider.__all__, (
            "mloda.provider.__all__ must keep listing 'ApiInputDataFeature' (moved, not removed)"
        )

    def test_old_plugin_module_is_gone(self) -> None:
        """The plugin module is deleted outright: no back-compat shim re-exporting the core class."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(_OLD_API_DATA_MODULE)


# Fresh-interpreter guard: importing a facade must pull in neither plugins nor optional backends.
# pyarrow is exempt: core's mloda/core/abstract_plugins/components/data_types.py imports it at module
# level when installed, so tests/test_core/test_optional_pyarrow/ (pyarrow blocked in a subprocess) is
# what proves the import stays optional.
_FORBIDDEN_LIBRARIES: list[str] = ["pandas", "polars", "duckdb"]

_NO_PLUGIN_SCRIPT = """
import sys

import {facade}

leaked = sorted(m for m in sys.modules if m == "mloda_plugins" or m.startswith("mloda_plugins."))
assert not leaked, f"import {facade} pulled in mloda_plugins modules: {{leaked}}"

print("ok")
"""

_NO_LIBRARY_SCRIPT = """
import sys

import {facade}

assert "{library}" not in sys.modules, "import {facade} pulled in the optional backend library {library}"

print("ok")
"""

_LIBRARY_PARAMS: list[tuple[str, str]] = [(facade, library) for facade in _FACADES for library in _FORBIDDEN_LIBRARIES]

_LIBRARY_IDS = [f"{facade}:{library}" for facade, library in _LIBRARY_PARAMS]


def _run_isolation_script(script: str) -> subprocess.CompletedProcess[str]:
    # Safe: fixed argv (sys.executable + a script built from module-level constants), no shell, no user input.
    return subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestFacadeImportIsDependencyFree:
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("facade", _FACADES)
    def test_facade_import_pulls_in_no_plugin(self, facade: str) -> None:
        """The point of #726: a core-only install must never load mloda_plugins through a facade."""
        result = _run_isolation_script(_NO_PLUGIN_SCRIPT.format(facade=facade))

        assert result.returncode == 0, (
            f"import {facade} must not import any mloda_plugins module in a fresh interpreter.\n"
            f"stderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(("facade", "library"), _LIBRARY_PARAMS, ids=_LIBRARY_IDS)
    def test_facade_import_pulls_in_no_backend_library(self, facade: str, library: str) -> None:
        """Importing a facade must not cost an optional backend import, whether or not it is installed."""
        result = _run_isolation_script(_NO_LIBRARY_SCRIPT.format(facade=facade, library=library))

        assert result.returncode == 0, (
            f"import {facade} must not import '{library}' in a fresh interpreter: the facades are the core "
            f"surface, optional backends load only through their mloda.user backend module.\n"
            f"stderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"


# (backend module under mloda.user, canonical plugin module, symbol)
_CYCLE_CASES: list[tuple[str, str, str]] = [
    ("mloda.user.python_dict", _PYTHON_DICT_FRAMEWORK_MODULE, "PythonDictFramework"),
    ("mloda.user.pandas", _PANDAS_DATAFRAME_MODULE, "PandasDataFrame"),
]

_IMPORT_ORDERS: list[str] = ["plugin_module_first", "backend_module_first"]

_CYCLE_PARAMS: list[tuple[str, str, str, str]] = [
    (backend, deep, symbol, order) for backend, deep, symbol in _CYCLE_CASES for order in _IMPORT_ORDERS
]

_CYCLE_IDS = [f"{backend}:{order}" for backend, _deep, _symbol, order in _CYCLE_PARAMS]


class TestImportOrderCycleGuard:
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize(("backend_module", "deep_module", "symbol", "order"), _CYCLE_PARAMS, ids=_CYCLE_IDS)
    def test_both_import_orders_resolve_the_same_class(
        self, backend_module: str, deep_module: str, symbol: str, order: str
    ) -> None:
        """Neither import order may raise ImportError on a mloda.user.<backend> <-> plugin-module cycle."""
        deep_import = f"import {deep_module} as deep\n"
        backend_import = f"import {backend_module} as backend\n"
        imports = deep_import + backend_import if order == "plugin_module_first" else backend_import + deep_import
        script = f"{imports}assert backend.{symbol} is deep.{symbol}, 'not the same class object'\nprint('ok')\n"

        # Safe: fixed argv (sys.executable + a script built from module-level constants), no shell, no user input.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Import order '{order}' failed in a fresh interpreter for {backend_module}. The plugin modules must "
            f"import their base classes from the defining modules, not from the facades; otherwise the backend "
            f"module re-enters a partially initialized plugin module and raises ImportError.\n"
            f"stderr:\n{result.stderr}"
        )
        assert result.stdout.strip() == "ok"


_COMPUTE_FRAMEWORK_PACKAGE = "mloda_plugins.compute_framework"


def _package_directory(package: str) -> Path:
    """Resolve a package directory through the imported package, not a hardcoded repo path."""
    module = importlib.import_module(package)
    source = module.__file__
    assert source is not None, f"{package} must be a file-backed package"
    return Path(source).parent


_COMPUTE_FRAMEWORK_DIRECTORY = _package_directory(_COMPUTE_FRAMEWORK_PACKAGE)

_COMPUTE_FRAMEWORK_SOURCES: list[Path] = sorted(_COMPUTE_FRAMEWORK_DIRECTORY.rglob("*.py"))

_COMPUTE_FRAMEWORK_SOURCE_IDS = [
    str(source.relative_to(_COMPUTE_FRAMEWORK_DIRECTORY)) for source in _COMPUTE_FRAMEWORK_SOURCES
]


class TestComputeFrameworkPluginsDoNotImportBackendModules:
    @pytest.mark.parametrize("source", _COMPUTE_FRAMEWORK_SOURCES, ids=_COMPUTE_FRAMEWORK_SOURCE_IDS)
    def test_plugin_module_does_not_import_a_user_backend_module(self, source: Path) -> None:
        """Rot guard: the mloda.user backend modules import the plugins, never the other way round."""
        offenders = _user_backend_imports(source)
        assert offenders == [], (
            f"{source} must not import from {sorted(set(offenders))}: a compute-framework internal importing "
            f"its own facade module (polars/lazy_dataframe.py importing mloda.user.polars, say) re-enters a "
            f"partially initialized module, because that module is what imports the plugin in the first place. "
            f"Import the class from its canonical mloda_plugins module instead."
        )


def _shipped_concrete_frameworks() -> list[type[Any]]:
    """Concrete ComputeFramework subclasses defined under mloda_plugins/compute_framework."""
    return sorted(
        (
            cls
            for cls in get_all_subclasses(ComputeFramework)
            if cls.__module__.startswith(f"{_COMPUTE_FRAMEWORK_PACKAGE}.") and not inspect.isabstract(cls)
        ),
        key=lambda cls: f"{cls.__module__}:{cls.__qualname__}",
    )


def _backend_module_exports() -> dict[str, list[str]]:
    """Map symbol -> mloda.user backend modules listing it in __all__, read via AST so nothing is imported."""
    exports: dict[str, list[str]] = {}
    for source in sorted(_package_directory("mloda.user").glob("*.py")):
        if source.name == "__init__.py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            for element in node.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    exports.setdefault(element.value, []).append(f"mloda.user.{source.stem}")
    return exports


class TestBackendModulesCoverEveryShippedFramework:
    def test_every_concrete_framework_is_published_from_exactly_one_backend_module(self) -> None:
        """Completeness: a new framework without a mloda.user backend module is unreachable for users."""
        # Every compute-framework plugin imports regardless of its backend library (issue #736), so the
        # group exposes all shipped frameworks here. The conftest fixture restores the registry.
        PluginLoader().load_group("compute_framework")

        frameworks = _shipped_concrete_frameworks()
        assert frameworks, "sanity: loading the compute_framework group must expose concrete frameworks"

        exports = _backend_module_exports()
        for framework in frameworks:
            backend_modules = exports.get(framework.__name__, [])
            assert len(backend_modules) == 1, (
                f"{framework.__module__}:{framework.__qualname__} must be re-exported from exactly one "
                f"mloda.user backend module, but is listed in the __all__ of {backend_modules}. Every shipped "
                f"framework needs its own mloda.user.<backend> module; that module is the only public import path."
            )
            backend = importlib.import_module(backend_modules[0])
            assert getattr(backend, framework.__name__) is framework, (
                f"{backend_modules[0]}.{framework.__name__} must be the identical class object from "
                f"{framework.__module__}"
            )
