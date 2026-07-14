"""The PROPERTY_MAPPING spec type.

``PropertySpec`` is the typed, frozen spec and IS the contract: construction enforces every
invariant (issue #694). ``property_spec`` is a thin builder kept for its authoring surface;
its ``strict=`` keyword maps to the ``strict_validation=`` field.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PropertySpec:
    """Typed, frozen PROPERTY_MAPPING spec. Construction enforces the spec invariants."""

    explanation: str
    # The declared type lists the container shapes so a bare str/bytes (the forgotten-comma bug) is
    # an author-time type error rather than a silent substring value space. The runtime is
    # deliberately more lenient: __post_init__ materializes ANY non-Mapping iterable (a generator,
    # a range) to a tuple, and keeps the str/bytes ValueError as the backstop for unchecked callers.
    # A Mapping is the value-space-with-descriptions form and is kept as given.
    allowed_values: Mapping[Any, str] | tuple[Any, ...] | list[Any] | set[Any] | frozenset[Any] | None = None
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
) -> PropertySpec:
    """Build a ``PropertySpec``; the ``strict=`` keyword sets the ``strict_validation`` field."""
    # The builder's authoring surface stays any-iterable (a generator, a range) because
    # PropertySpec.__post_init__ materializes those to a tuple; the field's declared type only lists
    # the container shapes. An untyped seam forwards the wider value to the narrower field.
    values: Any = allowed_values
    return PropertySpec(
        explanation,
        allowed_values=values,
        default=default,
        context=context,
        strict_validation=strict,
        element_validator=element_validator,
        match_guard=match_guard,
        required_when=required_when,
    )
