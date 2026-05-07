from __future__ import annotations

import ast
import importlib.metadata
import inspect
import linecache
import hashlib
from typing import Any
from abc import ABC


class BaseFeatureGroupVersion(ABC):
    @classmethod
    def mloda_version(cls) -> str:
        """
        Retrieves the version of the 'mloda' package using importlib.metadata.
        If retrieval fails, it returns "0.0.0" as a fallback.
        """
        try:
            return importlib.metadata.version("mloda")
        except Exception:
            return "0.0.0"

    @classmethod
    def class_source_hash(cls, target_class: type[Any]) -> str:
        """
        Returns a SHA-256 hash of the target class's source code.

        Falls back to a linecache-based lookup via the class's methods'
        ``__code__.co_filename`` when ``inspect.getsource`` cannot resolve the
        class (common for classes defined in long-lived namespaces such as
        Jupyter cells where ``__module__ == '__main__'``).
        """

        # Import FeatureGroup locally to avoid circular import.
        from mloda.core.abstract_plugins.feature_group import FeatureGroup

        if not issubclass(target_class, FeatureGroup):
            raise ValueError(f"target_class must be a subclass of FeatureGroup: {target_class}")

        try:
            source: str = inspect.getsource(target_class)
        except (OSError, TypeError):
            fallback = _linecache_source_for_class(target_class)
            if fallback is None:
                raise
            source = fallback
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @classmethod
    def module_name(cls, target_class: type[Any]) -> str:
        """
        Returns the module name of the target class.
        """
        return target_class.__module__

    @classmethod
    def version(cls, target_class: type[Any]) -> str:
        """
        Returns a composite version string.

        The version string is composed of:
          - the package version (from installed metadata),
          - the module name of the target class, and
          - a SHA-256 hash of the target class's source code.
        """

        # Import FeatureGroup locally to avoid circular import.
        from mloda.core.abstract_plugins.feature_group import FeatureGroup

        if not issubclass(target_class, FeatureGroup):
            raise ValueError(f"target_class must be a subclass of FeatureGroup: {target_class}")

        return f"{cls.mloda_version()}-{cls.module_name(target_class)}-{cls.class_source_hash(target_class)}"


def _linecache_source_for_class(target_class: type[Any]) -> str | None:
    """Returns the source for the target class's ``ClassDef`` AST node from the linecache, not the whole file.

    This keeps the hash stable when unrelated content in the same Jupyter cell changes.

    Only consults linecache for synthetic filenames of the form ``<...>`` (e.g.,
    Jupyter cells: ``<ipython-input-N-...>``). Real source files are skipped
    because ``inspect.getsource`` already had its chance and any text we'd read
    here would not be specific to the target class.
    """
    filenames: set[str] = set()
    for value in target_class.__dict__.values():
        func = value.__func__ if hasattr(value, "__func__") else value
        code = getattr(func, "__code__", None)
        if code is None:
            continue
        co_filename = code.co_filename
        if co_filename.startswith("<") and co_filename.endswith(">"):
            filenames.add(co_filename)
    if not filenames:
        return None

    target_name = target_class.__name__
    for filename in sorted(filenames):
        text = "".join(linecache.getlines(filename))
        if not text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == target_name:
                segment = ast.get_source_segment(text, node)
                if segment is not None:
                    return segment
    return None
