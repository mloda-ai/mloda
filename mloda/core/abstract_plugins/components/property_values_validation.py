"""Validate a plain dict of values against a PropertySpec mapping without echoing the values."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    FeatureChainParser,
    option_key_is_present,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec, element_admitted, is_no_default

logger = logging.getLogger(__name__)


class PropertyValidationError(ValueError):
    """A value dict the mapping rejects; the message names the key and reason, never the value."""

    def __init__(self, key: str, reason: str) -> None:
        super().__init__(f"'{key}' {reason}")
        self.key = key
        self.reason = reason


def _log_raise(key: str, what: str, exc: Exception) -> None:
    logger.warning("%s for '%s' raised %s; treating as rejected.", what, key, type(exc).__name__)


def _required_by_predicate(spec: PropertySpec, key: str, options: Options) -> bool | None:
    """Predicate verdict, or None when it raised (logged by type name only)."""
    predicate = spec.required_when
    assert predicate is not None
    try:
        return bool(predicate(options))
    except Exception as exc:  # Swallows: a predicate that raises cannot judge, so the key is rejected.
        _log_raise(key, "required_when", exc)
        return None


def _rejection(key: str, spec: PropertySpec, value: Any, options: Options) -> str | None:
    """The rejection reason for one declared key, or None when it is fine."""
    if not option_key_is_present(spec, key, options):
        if spec.required_when is not None:
            verdict = _required_by_predicate(spec, key, options)
            if verdict is None:
                return "required_when raised, so the key is rejected"
            return "is required but absent" if verdict else None
        return "is required but absent" if is_no_default(spec.default) else None

    if not spec.strict_validation:
        return None
    if spec.scalar_only and isinstance(value, (list, tuple, set, frozenset)):
        return "must be a scalar, not a collection"

    def log_validator_raise(exc: Exception) -> None:
        _log_raise(key, "element_validator", exc)

    for element in FeatureChainParser._unpack_property_value(value):
        if not element_admitted(spec, element, log_validator_raise):
            return "has a value that is not accepted"
    return None


def validate_property_values(
    values: Mapping[str, Any], mapping: Mapping[str, PropertySpec], *, closed_world: bool
) -> None:
    """Raise PropertyValidationError when values break the mapping; defaults are never applied."""
    for key, spec in mapping.items():
        if not isinstance(spec, PropertySpec):
            raise TypeError(f"mapping entry '{key}' must be a PropertySpec, got {type(spec).__name__}")

    if closed_world:
        for key in sorted(values, key=str):
            if key not in mapping:
                raise PropertyValidationError(str(key), "is not declared")

    options = Options(group=dict(values))
    for key, spec in mapping.items():
        reason = _rejection(key, spec, values.get(key), options)
        if reason is not None:
            raise PropertyValidationError(key, reason)
