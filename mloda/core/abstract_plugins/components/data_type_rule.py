"""Tagged-union return contract for ``FeatureGroup.return_data_type_rule``.

``DataTypeDeclaration`` (``DataType | None | Deferred``) is what a plugin may
return: a concrete type is expressed as a bare ``DataType``, ``None`` means the
feature group is polymorphic / has no fixed type, and ``Deferred`` is the
compute-time sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass

from mloda.core.abstract_plugins.components.data_types import DataType


@dataclass(frozen=True)
class Deferred:
    pass


DataTypeDeclaration = DataType | None | Deferred
