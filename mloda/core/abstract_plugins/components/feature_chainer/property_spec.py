"""Authoring helpers for PROPERTY_MAPPING specs.

``property_spec`` builds the conventional spec dict; that contract stays a plain dict.
``PropertySpec`` is the typed, frozen spec carrying the same invariants, enforced at
construction (issue #694).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mloda.core.abstract_plugins.components.default_options_key import DefaultOptionKeys
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import FeatureChainParser


def property_spec(
    explanation: str,
    *,
    strict: bool = False,
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None,
    default: Any = None,
    context: bool = True,
    element_validator: Callable[..., Any] | None = None,
    required_when: Callable[..., Any] | None = None,
    match_guard: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    # A str/bytes is iterable, so tuple("add") would silently build ('a', 'd', 'd'): reject the
    # shape before materializing it. Holds whether or not the spec is strict.
    if isinstance(allowed_values, (str, bytes)):
        raise ValueError(
            f"property_spec({explanation!r}): allowed_values is a {type(allowed_values).__name__} "
            f"({allowed_values!r}), which would make membership a substring test. Wrap it in a "
            f"container, for example a one-element tuple: ({allowed_values!r},)."
        )

    if allowed_values is not None and not isinstance(allowed_values, Mapping):
        allowed_values = tuple(allowed_values)

    if element_validator is not None and not callable(element_validator):
        raise ValueError(f"property_spec({explanation!r}): element_validator must be callable.")

    if required_when is not None and not callable(required_when):
        raise ValueError(f"property_spec({explanation!r}): required_when must be callable.")

    if match_guard is not None and not callable(match_guard):
        raise ValueError(f"property_spec({explanation!r}): match_guard must be callable.")

    # Dead without strict: _validate_property_value returns early on a non-strict spec. allowed_values
    # is NOT dead there, so it has no such rule: a non-strict value space still maps a name-parsed
    # value back onto its PROPERTY_MAPPING key.
    if element_validator is not None and not strict:
        raise ValueError(f"property_spec({explanation!r}): element_validator is never enforced without strict=True.")

    if strict and allowed_values is not None and not allowed_values:
        raise ValueError(f"property_spec({explanation!r}): an empty allowed_values would reject every value.")

    if strict and allowed_values is None and element_validator is None:
        raise ValueError(
            f"property_spec({explanation!r}): strict=True needs a non-empty allowed_values or an element_validator."
        )

    spec: dict[str, Any] = {"explanation": explanation}
    if allowed_values is not None:
        spec[DefaultOptionKeys.allowed_values] = allowed_values
    if element_validator is not None:
        spec[DefaultOptionKeys.element_validator] = element_validator
    if required_when is not None:
        spec[DefaultOptionKeys.required_when] = required_when
    if match_guard is not None:
        spec[DefaultOptionKeys.match_guard] = match_guard
    spec[DefaultOptionKeys.context] = context
    spec[DefaultOptionKeys.strict_validation] = strict
    if default is not None:
        spec[DefaultOptionKeys.default] = default

    # The declared-default semantics live in core, so the builder and the
    # class-definition check cannot drift apart.
    FeatureChainParser.check_declared_default(f"property_spec({explanation!r})", explanation, spec)

    return spec


@dataclass(frozen=True)
class PropertySpec:
    """Typed, frozen PROPERTY_MAPPING spec. Construction enforces the ``property_spec`` invariants."""

    explanation: str
    # Accepts any iterable (tuple, list, set, frozenset, generator); normalized to a tuple. A
    # Mapping is the value-space-with-descriptions form and is kept as given.
    allowed_values: Mapping[Any, str] | Iterable[Any] | None = None
    default: Any = None
    context: bool = True
    strict_validation: bool = False
    element_validator: Callable[[Any], Any] | None = None
    match_guard: Callable[[Any], Any] | None = None
    required_when: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        prefix = f"PropertySpec({self.explanation!r})"

        # A str/bytes is iterable, so tuple("add") would silently build ('a', 'd', 'd'): reject the
        # shape before materializing it.
        if isinstance(self.allowed_values, (str, bytes)):
            raise ValueError(
                f"{prefix}: allowed_values is a {type(self.allowed_values).__name__} "
                f"({self.allowed_values!r}), which would make membership a substring test. Wrap it in a "
                f"container, for example a one-element tuple: ({self.allowed_values!r},)."
            )

        if self.allowed_values is not None and not isinstance(self.allowed_values, Mapping):
            if not isinstance(self.allowed_values, Iterable):
                raise ValueError(
                    f"{prefix}: allowed_values is a {type(self.allowed_values).__name__} "
                    f"({self.allowed_values!r}), not a Mapping or an iterable of accepted values."
                )
            object.__setattr__(self, "allowed_values", tuple(self.allowed_values))

        if not isinstance(self.strict_validation, bool):
            raise ValueError(
                f"{prefix}: strict_validation must be a bool, got {type(self.strict_validation).__name__}."
            )

        if self.element_validator is not None and not callable(self.element_validator):
            raise ValueError(f"{prefix}: element_validator must be callable.")

        if self.required_when is not None and not callable(self.required_when):
            raise ValueError(f"{prefix}: required_when must be callable.")

        if self.match_guard is not None and not callable(self.match_guard):
            raise ValueError(f"{prefix}: match_guard must be callable.")

        if self.element_validator is not None and not self.strict_validation:
            raise ValueError(f"{prefix}: element_validator is never enforced without strict_validation=True.")

        if self.strict_validation and self.allowed_values is not None and not self.allowed_values:
            raise ValueError(f"{prefix}: an empty allowed_values would reject every value.")

        if self.strict_validation and self.allowed_values is None and self.element_validator is None:
            raise ValueError(
                f"{prefix}: strict_validation=True needs a non-empty allowed_values or an element_validator."
            )

        self._check_declared_default(prefix)

    def _check_declared_default(self, prefix: str) -> None:
        """A strict spec's declared default must be within its own value space (issue #530 semantics)."""
        if self.default is None or not self.strict_validation:
            return

        if self.element_validator is not None:
            try:
                verdict = self.element_validator(self.default)
            except Exception as exc:
                raise ValueError(
                    f"{prefix}: the element_validator raised an error when called with default {self.default!r}."
                ) from exc
            if not verdict:
                raise ValueError(f"{prefix}: default {self.default!r} is rejected by the element_validator.")
            return

        if self.allowed_values is None:
            return

        # Mapping membership tests keys; an unhashable default can never be a key.
        if isinstance(self.allowed_values, Mapping) and not isinstance(self.default, Hashable):
            raise ValueError(f"{prefix}: default {self.default!r} is not within the declared allowed_values.")

        if self.default not in self.allowed_values:
            raise ValueError(f"{prefix}: default {self.default!r} is not within the declared allowed_values.")
