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
from mloda.core.abstract_plugins.components.index.index import Index
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


class StubHookError(RuntimeError):
    """The single error an armed stub hook raises."""


class HookCounter:
    """The tally a consuming test module owns: hook calls by class, plus candidate-framework pairs."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.pairs: dict[tuple[str, str], int] = {}

    def record(self, class_name: str, hook: str) -> None:
        key = f"{class_name}.{hook}"
        self.calls[key] = self.calls.get(key, 0) + 1

    def record_pair(self, class_name: str, framework_name: str) -> None:
        """A hook-name tally cannot see one pair being asked twice, which is what this one is for."""
        pair = (class_name, framework_name)
        self.pairs[pair] = self.pairs.get(pair, 0) + 1

    def clear(self) -> None:
        # In place, never rebound: a caller aliases counter.calls at import and asserts on that alias
        # for the rest of its module.
        self.calls.clear()
        self.pairs.clear()


class CountingStubFeatureGroup(StubFeatureGroup):
    """A stub that tallies every provider-overridable hook it answers, and may raise from a named one."""

    COUNTER: ClassVar[HookCounter | None] = None
    SUPPORTED_FRAMEWORKS: ClassVar[frozenset[str] | None] = None
    INDEX_COLUMNS: ClassVar[list[Index] | None] = None
    SUPPORTS_INDEX_RESULT: ClassVar[bool | None] = None
    RAISING_HOOKS: ClassVar[frozenset[str]] = frozenset()
    ARMED: ClassVar[bool] = True

    @classmethod
    def _enter_hook(cls, hook: str, compute_framework: type[ComputeFramework] | None = None) -> None:
        """Count before raising: a caller asserts on the count of the very hook that raised."""
        # Read off the ClassVar at call time, so a child counts into whatever its base currently
        # declares. An unset counter counts nowhere and still answers: this is a globally visible
        # FeatureGroup subclass, so other tests' registry sweeps call these hooks.
        counter = cls.COUNTER
        if counter is not None:
            counter.record(cls.get_class_name(), hook)
            if compute_framework is not None:
                counter.record_pair(cls.get_class_name(), compute_framework.get_class_name())
        if cls.ARMED and hook in cls.RAISING_HOOKS:
            raise StubHookError(f"{cls.get_class_name()}.{hook} is armed to raise.")

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        cls._enter_hook("match_feature_group_criteria")
        return super().match_feature_group_criteria(feature_name, options, data_access_collection)

    @classmethod
    def get_domain(cls) -> Domain:
        cls._enter_hook("get_domain")
        return super().get_domain()

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        cls._enter_hook("compute_framework_rule")
        return super().compute_framework_rule()

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        cls._enter_hook("supports_compute_framework", compute_framework)
        if cls.SUPPORTED_FRAMEWORKS is None:
            return super().supports_compute_framework(feature_name, options, compute_framework)
        return compute_framework.get_class_name() in cls.SUPPORTED_FRAMEWORKS

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        cls._enter_hook("index_columns")
        if cls.INDEX_COLUMNS is None:
            return super().index_columns()
        return cls.INDEX_COLUMNS

    @classmethod
    def supports_index(cls, index: Index) -> bool | None:
        cls._enter_hook("supports_index")
        if cls.SUPPORTS_INDEX_RESULT is not None:
            return cls.SUPPORTS_INDEX_RESULT
        # Reads the ClassVar, not index_columns(), which is what super() would call: delegating would
        # count that hook a second time and would make an armed index_columns reachable through here.
        if cls.INDEX_COLUMNS is None:
            return None
        return any(index.is_a_part_of(column) for column in cls.INDEX_COLUMNS)

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        cls._enter_hook("feature_names_supported")
        return super().feature_names_supported()

    @classmethod
    def prefix(cls) -> str:
        cls._enter_hook("prefix")
        return super().prefix()


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
    counter: HookCounter | None = None,
    supported_frameworks: str | Iterable[str] | None = None,
    index_columns: list[Index] | None = None,
    supports_index: bool | None = None,
    raising_hooks: str | Iterable[str] | None = None,
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

    counting_keywords = {
        keyword: value
        for keyword, value in (
            ("counter", counter),
            ("supported_frameworks", supported_frameworks),
            ("index_columns", index_columns),
            ("supports_index", supports_index),
            ("raising_hooks", raising_hooks),
        )
        if value is not None
    }
    if counting_keywords and not issubclass(base, CountingStubFeatureGroup):
        # Only a counting base reads these ClassVars, so on any other base the keyword would be
        # silently dropped.
        raise ValueError(f"{', '.join(counting_keywords)} needs a CountingStubFeatureGroup base, got {base.__name__}.")

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
    if counter is not None:
        namespace["COUNTER"] = counter
    if supported_frameworks is not None:
        namespace["SUPPORTED_FRAMEWORKS"] = _name_set(supported_frameworks)
    if index_columns is not None:
        # A copy, for the same reason as frameworks above.
        namespace["INDEX_COLUMNS"] = list(index_columns)
    if supports_index is not None:
        namespace["SUPPORTS_INDEX_RESULT"] = supports_index
    if raising_hooks is not None:
        namespace["RAISING_HOOKS"] = _name_set(raising_hooks)
    if doc is not None:
        namespace["__doc__"] = doc
    if abstract:
        namespace[_STUB_ABSTRACT_HOOK.__name__] = _STUB_ABSTRACT_HOOK

    # The base's metaclass, not type: FeatureGroup is an ABC, and only ABCMeta turns the injected
    # abstract member into a real instantiation error.
    metaclass: type[Any] = type(base)
    return cast("type[StubFeatureGroup]", metaclass(name, (base,), namespace))


def make_raising_fg(
    name: str,
    hook: str,
    *,
    doc: str | None = None,
    base: type[FeatureGroup] = FeatureGroup,
) -> type[FeatureGroup]:
    """Mint a transient double in the calling module whose ``hook`` classmethod raises for any arguments.

    Claims no ``(module, name)`` and retains nothing: the caller reaps its class inside the test
    instead of binding it at module level, so the same name can be minted again.
    """

    def raise_from_hook(cls: type[FeatureGroup], *args: Any, **kwargs: Any) -> Any:
        # A plain RuntimeError, not StubHookError: this double stands in for an arbitrary broken plugin.
        raise RuntimeError(f"{name}.{hook} is a deliberately broken stub.")

    module: str = sys._getframe(1).f_globals["__name__"]
    namespace: dict[str, Any] = {"__module__": module, "__qualname__": name, hook: classmethod(raise_from_hook)}
    if doc is not None:
        namespace["__doc__"] = doc

    metaclass: type[Any] = type(base)
    return cast("type[FeatureGroup]", metaclass(name, (base,), namespace))
