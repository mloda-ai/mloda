"""Tagged-union return contract for ``FeatureGroup.return_data_type_rule``.

``DataTypeDeclaration`` (``DataType | None | Deferred``) is what a plugin may
return: a concrete type is expressed as a bare ``DataType``, ``None`` means the
feature group is polymorphic / has no fixed type, and ``Deferred`` is the
compute-time sentinel. ``Broken`` is the engine-internal wrapper carrying the
exception; it is produced only when a rule raises and is never returned by a
plugin.
"""

from __future__ import annotations

from dataclasses import dataclass

from mloda.core.abstract_plugins.components.data_types import DataType


@dataclass(frozen=True)
class Deferred:
    pass


@dataclass(frozen=True)
class Broken:
    error: BaseException


DataTypeDeclaration = DataType | None | Deferred
