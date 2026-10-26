"""Run-scoped match and dependency facts shared by candidate attempts and graph construction."""

from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_collection import Features
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import COLUMN_SEPARATOR
from mloda.core.abstract_plugins.components.input_data.match_cache import run_cached
from mloda.core.abstract_plugins.components.input_data.reader_probe import current_reader_probe_scope
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.implementation import current_implementations
from mloda.core.prepare.accessible_plugins import FeatureGroupEnvironmentMapping


T = TypeVar("T")
EnvironmentKey = tuple[Any, ...]
_CACHE = object()
_ENVIRONMENT = object()
_CYCLE = object()
_RUN: ContextVar[Any] = ContextVar("input_chain_run", default=None)


@contextmanager
def input_resolution_scope(run: Any) -> Iterator[None]:
    """A nested Engine has its own collector and extenders even inside an outer match cache."""
    token = _RUN.set(run)
    try:
        yield
    finally:
        _RUN.reset(token)


@dataclass(frozen=True, eq=False)
class _Identity:
    """Keep an unhashable run resource alive as well as distinguishing it by identity."""

    value: Any

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Identity) and self.value is other.value


def resolution_key(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    """Snapshot resolution inputs; Feature/Options equality intentionally omits some of them."""
    if isinstance(value, (Feature, Options, dict, list, tuple, set, frozenset)):
        if id(value) in seen:
            return _CYCLE
        seen = seen | {id(value)}
    if isinstance(value, Feature):
        return (
            Feature,
            str(value.name),
            resolution_key(value.options, seen),
            value.feature_group_scope,
            resolution_key(value.required_declarations, seen),
            value.domain,
            None if value.compute_frameworks is None else frozenset(value.compute_frameworks),
            value.implementation_pin,
            value.data_type,
            value.forward_group,
            value.forward_group_exclude,
            value.inherit_context_keys,
            value.link,
        )
    if isinstance(value, Options):
        # Includes context, forwarding/provenance flags, and provider-owned Options subclass fields.
        return type(value), resolution_key(value.__dict__, seen)
    if isinstance(value, dict):
        return type(value), frozenset((resolution_key(k, seen), resolution_key(v, seen)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value), tuple(resolution_key(v, seen) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value), frozenset(resolution_key(v, seen) for v in value)
    try:
        hash(value)
    except Exception:  # Swallows: an optional memo key must not reject an otherwise valid provider value.
        # repr() is only a hash fallback elsewhere; it cannot establish value equality for a cache.
        return type(value), _Identity(value)
    return type(value), value


def resolution_environment(
    accessible_plugins: FeatureGroupEnvironmentMapping,
    links: set[Link] | None,
    data_access_collection: DataAccessCollection | None,
    excluded: Mapping[type[FeatureGroup], str] | None = None,
) -> EnvironmentKey:
    """A nested or standalone resolution must not borrow another run's grants or plugin universe."""
    environment = (
        _Identity(_RUN.get()),
        frozenset((group, frozenset(frameworks)) for group, frameworks in accessible_plugins.items()),
        None if links is None else frozenset(links),
        _Identity(data_access_collection),
        _Identity(current_reader_probe_scope()),
        frozenset(current_implementations()),
        frozenset((excluded or {}).items()),
    )
    # Reuse the complete snapshot so each candidate lookup does not compare the whole plugin universe.
    return run_cached((_ENVIRONMENT, environment), lambda: environment)


@dataclass(frozen=True)
class _InputFactFailure:
    error: Exception


def cached_input_fact(environment: EnvironmentKey, kind: str, key: Hashable, compute: Callable[[], T]) -> T:
    """Memoize successes and exact exceptions without choosing their containment policy."""

    def capture() -> T | _InputFactFailure:
        try:
            return compute()
        except Exception as exc:  # Swallows: store the exact exception until this helper immediately re-raises it.
            return _InputFactFailure(exc.with_traceback(None))

    outcome = run_cached((_CACHE, environment, kind, key), capture)
    if isinstance(outcome, _InputFactFailure):
        raise outcome.error.with_traceback(None)
    return outcome


@dataclass(frozen=True)
class _NameOutcome:
    name: FeatureName
    part: str | None
    options: Options
    error: Exception | None = None


def normalize_input_name(
    environment: EnvironmentKey,
    feature: Feature,
    feature_group: FeatureGroup,
    output_part: tuple[str, str] | None,
) -> None:
    """Apply the same provider rename and output-part handling during a probe and at intake."""
    requested = feature.name
    key = type(feature_group), str(requested), resolution_key(feature.options), output_part

    def compute() -> _NameOutcome:
        options = feature.options._fork()
        name = FeatureName(output_part[0]) if output_part is not None else requested
        try:
            name = feature_group.set_feature_name(options, name)
        except Exception as exc:  # noqa: BLE001  (capture an outcome; the caller retains the raise policy)
            return _NameOutcome(name, None, options, exc.with_traceback(None))
        part = output_part[1] if output_part is not None else None
        if output_part is None and name != requested and requested.startswith(f"{name}{COLUMN_SEPARATOR}"):
            part = requested[len(name) + 1 :]
        return _NameOutcome(name, part, options)

    outcome = cached_input_fact(environment, "name", key, compute)
    feature.options._adopt(outcome.options)
    if outcome.error is not None:
        raise outcome.error.with_traceback(None)
    feature.name = outcome.name
    if outcome.part is not None:
        feature.record_part(outcome.part)


@dataclass
class _InputTemplate:
    feature: Feature
    source: Feature
    materialized: bool = False


@dataclass(frozen=True)
class _InputDeclaration:
    entries: tuple[_InputTemplate | str, ...] | None
    options: Options
    error: Exception | None = None


def _copy_input(feature: Feature, *, fresh_identity: bool = False) -> Feature:
    fresh = copy(feature)
    # Feature.__copy__ rebuilds plain Options for intake equality. A provider's declared
    # input must retain its Options subclass and fields, just as the matcher's fork does.
    fresh.options = feature.options._fork()
    if feature.child_options is not None:
        fresh.child_options = feature.child_options._fork()
    if fresh_identity:
        fresh.uuid = uuid4()
    return fresh


def declared_inputs(
    environment: EnvironmentKey,
    feature_group: type[FeatureGroup],
    options: Options,
    feature_name: FeatureName,
    *,
    materialize: bool = False,
) -> tuple[Feature | str, ...] | None:
    """Snapshot declarations for preflight; real expansion retains the provider's live input references."""
    key = feature_group, str(feature_name), resolution_key(options)

    def compute() -> _InputDeclaration:
        forked = options._fork()
        try:
            returned = feature_group().input_features(forked, feature_name)
            if returned is None:
                return _InputDeclaration(None, forked)
            entries: list[Feature | str] = list(returned)
            if isinstance(returned, (set, frozenset)):
                entries.sort(key=lambda item: str(item.name) if isinstance(item, Feature) else str(item))
            return _InputDeclaration(
                tuple(
                    _InputTemplate(_copy_input(item), item) if isinstance(item, Feature) else item for item in entries
                ),
                forked,
            )
        except Exception as exc:  # noqa: BLE001  (also cache raises, without changing sole-candidate behavior)
            return _InputDeclaration(None, forked, exc.with_traceback(None))

    outcome = cached_input_fact(environment, "declaration", key, compute)
    options._adopt(outcome.options)
    if outcome.error is not None:
        raise outcome.error.with_traceback(None)
    if outcome.entries is None:
        return None
    entries: list[Feature | str] = []
    for item in outcome.entries:
        if isinstance(item, _InputTemplate):
            if materialize and not item.materialized:
                item.materialized = True
                feature = item.source
                if feature.options is outcome.options:
                    feature.options = options
            else:
                # Replaying a cached declaration for another parent must not reuse its graph UUID or domain.
                feature = _copy_input(item.feature, fresh_identity=True)
            entries.append(feature)
        else:
            entries.append(item)
    return tuple(entries)


def input_features_collection(
    feature_group: type[FeatureGroup],
    entries: tuple[Feature | str, ...],
    parent: Feature,
) -> Features:
    """Use Features' real inheritance, including context directives and parent-domain propagation."""
    for entry in entries:
        if isinstance(entry, Feature) and entry.connection is not None:
            raise ValueError(
                f"{feature_group.get_class_name()}.input_features returned '{entry.name}' with a "
                "connection; register it in the DataAccessCollection and set the 'connection_handle' option."
            )
    return Features(
        list(entries),
        child_options=parent.options,
        child_uuid=parent.uuid,
        parent_domain=parent.domain.name if parent.domain is not None else None,
    )


@dataclass(frozen=True)
class InputChainOutcome:
    viable: bool
    computing: bool
    reason: str = ""
    refusals: tuple[Exception, ...] = ()
    # A back edge is provisionally viable, but cannot seed a completed-success memo.
    pending: frozenset[Hashable] = frozenset()


@dataclass
class InputChainState:
    active: set[Hashable] = field(default_factory=set)
    active_features: dict[Hashable, set[Hashable]] = field(default_factory=dict)
    completed: dict[Hashable, InputChainOutcome] = field(default_factory=dict)


def input_chain_state(environment: EnvironmentKey) -> InputChainState:
    return cached_input_fact(environment, "chains", (), InputChainState)
