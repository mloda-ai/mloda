from __future__ import annotations

import ast
import importlib.metadata
import inspect
import linecache
import hashlib
import weakref
from typing import Any, Final
from abc import ABC

# Exceptions inspect.getsource raises for dynamically built classes (no source backing / built-in).
SOURCE_INTROSPECTION_ERRORS: Final = (OSError, TypeError)


# Hash cache keyed by the class OBJECT (never by name): a redefined same-name
# class is a new object and must be hashed fresh. The hash is deterministic per
# class object within a process; docs enumeration calls class_source_hash for
# every FeatureGroup subclass per call, and this cache keeps that walk off the
# pytest 10s timeout as subclass counts grow. The cache is weak-keyed so
# temporary classes (Jupyter cells, test-local, synthetic dedup classes) are
# not pinned for the process lifetime and can still be garbage-collected.
_class_source_hash_cache: "weakref.WeakKeyDictionary[type[Any], str]" = weakref.WeakKeyDictionary()

# Package metadata cannot change within a process; docs enumeration calls
# mloda_version per FeatureGroup subclass, and repeated importlib.metadata
# parsing was a measured hot path in large test sessions.
_mloda_version_cache: str | None = None


class BaseFeatureGroupVersion(ABC):
    @classmethod
    def mloda_version(cls) -> str:
        """
        Retrieves the version of the 'mloda' package using importlib.metadata.
        If retrieval fails, it returns "0.0.0" as a fallback.
        The result (including the fallback) is memoized for the process lifetime.
        """
        global _mloda_version_cache
        if _mloda_version_cache is not None:
            return _mloda_version_cache
        try:
            _mloda_version_cache = importlib.metadata.version("mloda")
        except Exception:
            _mloda_version_cache = "0.0.0"
        return _mloda_version_cache

    @classmethod
    def class_source_hash(cls, target_class: type[Any]) -> str:
        """
        Returns a SHA-256 hash of the target class's source code.

        Text returned by ``inspect.getsource`` is validated: it must actually
        define the target class (Python 3.13+ can return unrelated file content
        for exec-defined classes via ``__firstlineno__``). Invalid text is
        treated like a getsource failure.

        Falls back to a linecache-based lookup via the class's methods'
        ``__code__.co_filename`` when ``inspect.getsource`` cannot resolve the
        class (common for classes defined in long-lived namespaces such as
        Jupyter cells where ``__module__ == '__main__'``).
        """

        # Import FeatureGroup locally to avoid circular import.
        from mloda.core.abstract_plugins.feature_group import FeatureGroup

        if not issubclass(target_class, FeatureGroup):
            raise ValueError(f"target_class must be a subclass of FeatureGroup: {target_class}")

        cached = _class_source_hash_cache.get(target_class)
        if cached is not None:
            return cached

        try:
            source: str = inspect.getsource(target_class)
        except SOURCE_INTROSPECTION_ERRORS:
            fallback = _linecache_source_for_class(target_class)
            if fallback is None:
                raise
            source = fallback
        else:
            if not _source_defines_class(source, target_class.__name__):
                fallback = _linecache_source_for_class(target_class)
                if fallback is None:
                    raise OSError(
                        f"inspect.getsource returned text that does not define {target_class.__name__!r} "
                        "and no linecache fallback was available"
                    )
                source = fallback
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        _class_source_hash_cache[target_class] = source_hash
        return source_hash

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


def _source_defines_class(source: str, class_name: str) -> bool:
    """Returns True when the source text contains a ``ClassDef`` named ``class_name``.

    On Python 3.13+, ``inspect.getsource`` slices lines out of whatever file the
    class's module resolves to (via ``__firstlineno__``), so it can succeed with
    text that does not define the class at all.

    Nested classes come back indented, so the text is parsed twice: first as-is
    (top-level classes), then wrapped in an ``if True:`` block so the indented
    text becomes a valid block body. ``textwrap.dedent`` is NOT used here: it
    computes the common indent over all nonblank lines including lines inside
    string literals, so a flush-left line inside a multiline string makes dedent
    a no-op and the still-indented text would be wrongly rejected. If neither
    parse succeeds, the source does not define the class.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        try:
            tree = ast.parse("if True:\n" + source)
        except SyntaxError:
            return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
    return False


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
