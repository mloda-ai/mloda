"""Shared FeatureGroup stubs: a test declares a plugin by keyword instead of writing a class body."""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar, cast

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup


class StubFeatureGroup(FeatureGroup):
    """A FeatureGroup that reads every answer off its own ClassVars."""

    MATCHED_NAMES: ClassVar[frozenset[str]] = frozenset()
    DOMAIN_NAME: ClassVar[str | None] = None
    FRAMEWORK_RULE: ClassVar[set[type[ComputeFramework]] | None] = None
    SUPPORTED_NAMES: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        # Replaces the FeatureGroup default, which matches the class name, its prefix and the input
        # data, so an unconfigured stub adds no matcher to the suite.
        return str(feature_name) in cls.MATCHED_NAMES

    @classmethod
    def get_domain(cls) -> Domain:
        if cls.DOMAIN_NAME is None:
            return super().get_domain()
        return Domain(cls.DOMAIN_NAME)

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return cls.FRAMEWORK_RULE

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.SUPPORTED_NAMES)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """None is the root-feature protocol signal that ``FeatureGroup.is_root`` reads."""
        return None


def _stub_abstract_hook(self: FeatureGroup) -> None:
    """Unimplemented member that makes an ``abstract=True`` stub abstract to ABCMeta."""


# A call, not a decorator: mypy rejects @abstractmethod outside a class body. Both spellings only set
# __isabstractmethod__ on this same function, which is what ABCMeta reads at class creation.
_STUB_ABSTRACT_HOOK = abstractmethod(_stub_abstract_hook)


# Names only, never the classes: a retained class object would outlive its own test and trip the
# registry-isolation fixture on the next one.
_MINTED_NAMES: set[tuple[str, str]] = set()


def _claim(module: str, name: str) -> None:
    """Reserve one (module, name) pair per process."""
    if (module, name) in _MINTED_NAMES:
        raise ValueError(f"A stub named {name} was already minted in {module}. Give this one its own name.")
    _MINTED_NAMES.add((module, name))


def _name_set(names: str | Iterable[str]) -> frozenset[str]:
    """One bare name, or an iterable of them; a str is never spread over its characters."""
    if isinstance(names, str):
        return frozenset({names})
    return frozenset(names)


def make_fg(
    name: str,
    *,
    matches: str | Iterable[str] | None = None,
    domain: str | None = None,
    frameworks: set[type[ComputeFramework]] | None = None,
    supported_names: str | Iterable[str] | None = None,
    abstract: bool = False,
    base: type[StubFeatureGroup] = StubFeatureGroup,
    doc: str | None = None,
) -> type[StubFeatureGroup]:
    """Mint a FeatureGroup subclass named ``name`` in the calling module, configured by keyword.

    Bind the result at module level under exactly ``name``: that binding is what makes the class
    picklable and what ``_is_live_in_module`` and the registry-isolation fixture read.
    One ``(module, name)`` pair per process, so this cannot be called from a fixture or from a
    parametrized test.
    A minted class has no retrievable source, so ``version()`` is unavailable and the docs path
    reports it as such; a stub needing a real version must be written as a class statement.
    """
    if not issubclass(base, StubFeatureGroup):
        # A foreign base reads none of the stub ClassVars, so every keyword is silently dropped and
        # FeatureGroup's default matcher, which matches the class's own name, is back in the suite.
        raise ValueError(f"base must be a StubFeatureGroup subclass, got {base.__name__}.")

    # The caller's module, so (module, qualname) dedup, _is_live_in_module and the isolation fixture
    # read a minted stub exactly like a hand-written class.
    module: str = sys._getframe(1).f_globals["__name__"]
    _claim(module, name)

    namespace: dict[str, Any] = {"__module__": module, "__qualname__": name}
    # An omitted keyword is None and inherits the base's ClassVar; an explicitly empty one is a
    # value, and shadows it with the empty declaration.
    if matches is not None:
        namespace["MATCHED_NAMES"] = _name_set(matches)
    if domain is not None:
        namespace["DOMAIN_NAME"] = domain
    if frameworks is not None:
        # A copy: mutating the caller's set afterwards must not rewrite the minted class's rule.
        namespace["FRAMEWORK_RULE"] = set(frameworks)
    if supported_names is not None:
        namespace["SUPPORTED_NAMES"] = _name_set(supported_names)
    if doc is not None:
        namespace["__doc__"] = doc
    if abstract:
        namespace[_STUB_ABSTRACT_HOOK.__name__] = _STUB_ABSTRACT_HOOK

    # The base's metaclass, not type: FeatureGroup is an ABC, and only ABCMeta turns the injected
    # abstract member into a real instantiation error.
    metaclass: type[Any] = type(base)
    return cast("type[StubFeatureGroup]", metaclass(name, (base,), namespace))
