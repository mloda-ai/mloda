from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from mloda.core.abstract_plugins.components.options import Options


@dataclass(frozen=True, kw_only=True)
class SubtypeDeclaration:
    """Declarative subtype dimension of a FeatureGroup family.

    Exactly two legal shapes: shape A names a ``PROPERTY_MAPPING`` key with an
    enumerable value space (``key``); shape B declares an explicit ``universe``
    together with a ``resolver``. ``parametric_families`` and ``supported`` are
    legal in both shapes. Shape-A checks that depend on ``PROPERTY_MAPPING``
    run in ``FeatureGroup.__init_subclass__``.
    """

    key: str | None = None
    universe: Iterable[str] | None = None
    resolver: Callable[[str, Options], str | None] | None = None
    parametric_families: Mapping[str, str] | None = None
    supported: Mapping[str, Iterable[str]] | None = None

    def __post_init__(self) -> None:
        if self.universe is not None:
            object.__setattr__(self, "universe", frozenset(self.universe))
        if self.parametric_families is not None:
            object.__setattr__(self, "parametric_families", dict(self.parametric_families))
        if self.supported is not None:
            object.__setattr__(self, "supported", {name: frozenset(values) for name, values in self.supported.items()})

        if self.key is not None and (self.universe is not None or self.resolver is not None):
            raise ValueError("SubtypeDeclaration: 'key' cannot be combined with 'universe' or 'resolver'.")
        if self.universe is not None and self.resolver is None:
            raise ValueError("SubtypeDeclaration: 'universe' requires a 'resolver'.")
        if self.resolver is not None and self.universe is None:
            raise ValueError("SubtypeDeclaration: 'resolver' requires a 'universe'.")
        if self.key is None and self.universe is None:
            raise ValueError(
                "SubtypeDeclaration: declare either 'key' (shape A) or 'universe' with 'resolver' (shape B)."
            )

        if self.universe is None:
            return

        literals = frozenset(self.universe)
        families = self.family_names()
        colliding = families & literals
        if colliding:
            raise ValueError(
                f"SubtypeDeclaration: parametric families {sorted(colliding)} collide with declared universe members."
            )

        if self.supported is not None:
            full_universe = literals | families
            for framework_name, values in self.supported.items():
                overreach = frozenset(values) - full_universe
                if overreach:
                    raise ValueError(
                        f"SubtypeDeclaration: supported subtypes {sorted(overreach)} on "
                        f"{framework_name} are outside the declared universe."
                    )

    def family_names(self) -> frozenset[str]:
        return frozenset(self.parametric_families or ())
