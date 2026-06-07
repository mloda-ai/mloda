"""Tagged-union return contract for ``FeatureGroup.return_data_type_rule``.

``RuleResult`` (``DataType | Open | Deferred``) is what plugins may return: a
concrete type is expressed as a bare ``DataType``, while ``Open`` and ``Deferred``
are sentinels. ``RuleOutcome`` additionally includes ``Broken``, which is
engine-internal and produced only when a rule raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from mloda.core.abstract_plugins.components.data_types import DataType


@dataclass(frozen=True)
class Open:
    pass


@dataclass(frozen=True)
class Deferred:
    pass


@dataclass(frozen=True)
class Broken:
    error: BaseException


RuleResult = DataType | Open | Deferred
RuleOutcome = DataType | Open | Deferred | Broken
