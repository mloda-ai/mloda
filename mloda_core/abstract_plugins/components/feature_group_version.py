from __future__ import annotations

import importlib.metadata
import inspect
import hashlib
from typing import Type
from abc import ABC


class FeatureGroupVersion(ABC):
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
    def class_source_hash(cls, target_class: Type) -> str:  # type: ignore
        """
        Returns a SHA-256 hash of the target class's source code.
        """

        # Import AbstractFeatureGroup locally to avoid circular import.
        from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

        if not issubclass(target_class, AbstractFeatureGroup):
            raise ValueError(f"target_class must be a subclass of AbstractFeatureGroup: {target_class}")

        source = inspect.getsource(target_class)
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @classmethod
    def module_name(cls, target_class: Type) -> str:  # type: ignore
        """
        Returns the module name of the target class.
        """
        return target_class.__module__

    @classmethod
    def version(cls, target_class: Type) -> str:  # type: ignore
        """
        Returns a composite version string.

        The version string is composed of:
          - the package version (from installed metadata),
          - the module name of the target class, and
          - a SHA-256 hash of the target class's source code.
        """

        # Import AbstractFeatureGroup locally to avoid circular import.
        from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup

        if not issubclass(target_class, AbstractFeatureGroup):
            raise ValueError(f"target_class must be a subclass of AbstractFeatureGroup: {target_class}")

        return f"{cls.mloda_version()}-{cls.module_name(target_class)}-{cls.class_source_hash(target_class)}"
