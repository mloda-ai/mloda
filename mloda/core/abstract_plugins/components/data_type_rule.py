"""Tagged-union return contract for ``FeatureGroup.return_data_type_rule``.

``RuleResult`` (``Fixed | Open | Deferred``) is what plugins may return.
``RuleOutcome`` additionally includes ``Broken``, which is engine-internal and
produced only when a rule raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from mloda.core.abstract_plugins.components.data_types import DataType


@dataclass(frozen=True)
class Fixed:
    data_type: DataType


@dataclass(frozen=True)
class Open:
    pass


@dataclass(frozen=True)
class Deferred:
    pass


@dataclass(frozen=True)
class Broken:
    error: BaseException


RuleResult = Fixed | Open | Deferred
RuleOutcome = Fixed | Open | Deferred | Broken
