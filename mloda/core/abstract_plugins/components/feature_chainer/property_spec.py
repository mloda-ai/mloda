"""Authoring helper that builds a conventional PROPERTY_MAPPING spec dict.

The contract stays a plain dict; this only changes authoring, validating the
invariants up front instead of relying on hand-written spec dicts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
) -> dict[str, Any]:
    if allowed_values is not None and not isinstance(allowed_values, Mapping):
        allowed_values = tuple(allowed_values)

    if strict and not allowed_values:
        raise ValueError(f"property_spec({explanation!r}): strict=True needs a non-empty allowed_values.")

    if not strict and allowed_values is not None:
        raise ValueError(f"property_spec({explanation!r}): allowed_values is never enforced without strict=True.")

    if strict and default is not None and allowed_values is not None:
        accepted = set(allowed_values.keys()) if isinstance(allowed_values, Mapping) else set(allowed_values)
        if default not in accepted:
            raise ValueError(
                f"property_spec({explanation!r}): default {default!r} is not in the accepted set {accepted}."
            )

    spec: dict[str, Any] = {"explanation": explanation}
    if allowed_values is not None:
        spec[DefaultOptionKeys.allowed_values] = allowed_values
    spec[DefaultOptionKeys.context] = context
    spec[DefaultOptionKeys.strict_validation] = strict
    spec[DefaultOptionKeys.default] = default
    return spec
