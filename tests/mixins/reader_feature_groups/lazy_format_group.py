"""Class-attribute descriptor that imports a format group on first access, so test files collect without it."""

import importlib
from typing import TYPE_CHECKING, Any, cast

from mloda.provider import ReadFileFG

if TYPE_CHECKING:
    from mloda.provider import ReadDocumentFG

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


DOCUMENT_FORMATS_PACKAGE = "mloda_plugins.feature_group.input_data.document_formats"


class _LazyDocumentGroup:
    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name

    def __get__(self, obj: Any, owner: Any = None) -> "type[ReadDocumentFG]":
        return cast("type[ReadDocumentFG]", getattr(importlib.import_module(self._module), self._name))


def lazy_document_group(module_name: str, class_name: str) -> "type[ReadDocumentFG]":
    """Like lazy_group for the document formats package (module_name is relative to document_formats)."""
    return cast("type[ReadDocumentFG]", _LazyDocumentGroup(f"{DOCUMENT_FORMATS_PACKAGE}.{module_name}", class_name))


def load_document_group(module_name: str, class_name: str) -> "type[ReadDocumentFG]":
    """The document group class, imported now (call inside a test body or fixture)."""
    module = importlib.import_module(f"{DOCUMENT_FORMATS_PACKAGE}.{module_name}")
    return cast("type[ReadDocumentFG]", getattr(module, class_name))
