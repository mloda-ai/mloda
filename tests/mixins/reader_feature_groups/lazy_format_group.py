"""Class-attribute descriptor that imports a format group on first access, so test files collect without it."""

import importlib
from typing import Any, cast

from mloda.provider import ReadFileFG

FILE_FORMATS_PACKAGE = "mloda_plugins.feature_group.input_data.file_formats"


class _LazyGroup:
    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name

    def __get__(self, obj: Any, owner: Any = None) -> type[ReadFileFG]:
        return cast(type[ReadFileFG], getattr(importlib.import_module(self._module), self._name))


def lazy_group(module_name: str, class_name: str) -> type[ReadFileFG]:
    """Stand-in for a stock group class, resolved on attribute access; module_name is relative to file_formats."""
    return cast(type[ReadFileFG], _LazyGroup(f"{FILE_FORMATS_PACKAGE}.{module_name}", class_name))


def load_group(module_name: str, class_name: str) -> type[ReadFileFG]:
    """The stock group class, imported now (call inside a test body or fixture)."""
    return cast(type[ReadFileFG], getattr(importlib.import_module(f"{FILE_FORMATS_PACKAGE}.{module_name}"), class_name))
