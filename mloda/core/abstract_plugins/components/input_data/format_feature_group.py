"""Shared base of one-FeatureGroup-per-format: claim routes decide matching, hooks discover and load sources."""

import inspect
from abc import abstractmethod
from collections.abc import Callable, Collection
from typing import Any, ClassVar

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.base_input_data import (
    RESERVED_READER_OPTION_KEY,
    dispatch_input_data_load,
)
from mloda.core.abstract_plugins.components.input_data.claim_route import (
    ClaimRoute,
    NamePolicy,
    SourceMatch,
    aborts_are_contained,
    current_feature_group_scope,
)
from mloda.core.abstract_plugins.components.match_rejection import INPUT_DATA_STAGE, record_match_rejection
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import defer_match_abort, escalate_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.data_types import DataType

_MAX_LISTED_COLUMNS = 20

HANDLE_OPTION = "data_access_handle"
HANDLE_SPEC = PropertySpec(
    "Name of the DataAccessCollection handle to read from.",
    default=None,
    strict_validation=True,
    context=True,
    element_validator=lambda value: isinstance(value, str),
    match_guard=lambda value: isinstance(value, str),
    expected="a str handle name",
)


def _listed(columns: Collection[str] | None) -> str:
    if columns is None:
        return "unknown"
    names = sorted(columns)
    text = ", ".join(names[:_MAX_LISTED_COLUMNS])
    if len(names) > _MAX_LISTED_COLUMNS:
        text += f", ... and {len(names) - _MAX_LISTED_COLUMNS} more"
    return f"[{text}]"


class FormatFeatureGroup(FeatureGroup):
    """A FeatureGroup that claims features of one source format through its CLAIM_ROUTES."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = ()
    _LOADERS: ClassVar[dict[type[ComputeFramework], Callable[[SourceMatch, Any], Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for route in cls.__dict__.get("CLAIM_ROUTES", ()):
            if route.names is NamePolicy.OPEN and route.searched:
                raise TypeError(
                    f"{cls.__name__} declares an OPEN route that is searched; an open route would claim every "
                    "name, so it must be pointed-only (searched=False)."
                )

    @classmethod
    @abstractmethod
    def find_sources(
        cls,
        route: ClaimRoute,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> list[SourceMatch]:
        """Sources of the route's kind; must not mutate the collection or options. `source` is unique per access."""

    @classmethod
    @abstractmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        """The neutral form of the source, used when the current framework has no loader."""

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        """Column names of the source, None when it cannot enumerate."""
        return None

    @classmethod
    def unknown_columns_reason(cls, match: SourceMatch) -> str | None:
        """Why the source cannot enumerate its columns, None when no reason is known."""
        return None

    @classmethod
    def has_column(cls, match: SourceMatch, column: str) -> bool | None:
        """Whether the source has the column, None when unknown."""
        columns = cls.columns(match)
        return None if columns is None else column in columns

    @classmethod
    def ambiguity_fix(
        cls, feature_name: str, matches: list[SourceMatch], data_access_collection: DataAccessCollection | None
    ) -> str:
        """Advice appended to the several-sources abort."""
        return "narrow with a data_access_handle or column_to_file."

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return frozenset()

    @classmethod
    def declared_name_prefixes(cls) -> tuple[str, ...]:
        return ()

    @classmethod
    def is_pointed(
        cls, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> bool:
        """True when the user pointed at this group: class-name key or feature_group= scope."""
        return cls._scope_points_here(current_feature_group_scope()) or any(
            key in options for key in cls.pointer_keys()
        )

    @classmethod
    def pointer_keys(cls) -> tuple[str, ...]:
        """Own class name, then concrete format ancestors, most-derived first."""
        return tuple(
            klass.__name__
            for klass in cls.__mro__
            if issubclass(klass, FormatFeatureGroup)
            and klass is not FormatFeatureGroup
            and (klass is cls or not inspect.isabstract(klass))
        )

    @classmethod
    def _scope_points_here(cls, scope: Any) -> bool:
        """A scope points at this group when it names the group or a non-abstract format ancestor."""
        if scope is None:
            return False
        for klass in cls.__mro__:
            if not (isinstance(klass, type) and issubclass(klass, FormatFeatureGroup)):
                continue
            if klass is not cls and inspect.isabstract(klass):
                continue
            if scope is klass or scope == klass.__name__:
                return True
        return False

    @classmethod
    def register_loader(cls, framework: type[ComputeFramework], loader: Callable[[SourceMatch, Any], Any]) -> None:
        """Register loader(match, features) for framework on this class only."""
        loaders = cls.__dict__.get("_LOADERS")
        if loaders is None:
            loaders = {}
            cls._LOADERS = loaders
        if framework in loaders:
            raise ValueError(f"{cls.__name__} already has a loader registered for {framework.__name__}.")
        loaders[framework] = loader

    @classmethod
    def _loader_for(cls, framework: type[ComputeFramework]) -> Callable[[SourceMatch, Any], Any] | None:
        if not framework.is_available():
            return None
        for klass in cls.__mro__:
            for framework_class in framework.__mro__:
                loader: Callable[[SourceMatch, Any], Any] | None = klass.__dict__.get("_LOADERS", {}).get(
                    framework_class
                )
                if loader is not None:
                    return loader
            if "load_neutral" in klass.__dict__:
                return None
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Load the matched source with the current framework's loader, else its neutral form, via the hook."""
        match = features.input_data_match
        if match is None:
            raise ValueError(f"{cls.__name__}.calculate_feature found no input_data_match on the feature set.")
        source = match[1]
        framework = features.get_sorted_features()[0].get_compute_framework()
        loader = cls._loader_for(framework)
        load = loader if loader is not None else cls.load_neutral
        loader_name = framework.__name__ if loader is not None else "neutral"
        return dispatch_input_data_load(
            cls,
            cls,
            load,
            source,
            features,
            identity=lambda: cls.data_access_identity(source),
            data_access_format=cls.data_access_name,
            loader_name=loader_name,
        )

    @classmethod
    def describe_columns(cls, match: SourceMatch) -> dict[str, DataType | None]:
        """Column name to None for each column the source lists; raises when it cannot enumerate."""
        columns = cls.columns(match)
        if columns is None:
            raise NotImplementedError(f"{cls.__name__} cannot enumerate the columns of {match.source}.")
        return {column: None for column in columns}

    @classmethod
    def count_rows(cls, match: SourceMatch, compute_framework: type[ComputeFramework]) -> int | None:
        """Rows of the source without loading it; None when only a read can tell."""
        return None

    @classmethod
    def data_access_name(cls) -> str:
        return cls.get_class_name()

    @classmethod
    def data_access_identity(cls, match: SourceMatch) -> str:
        return match.source

    @classmethod
    def _declared_name(cls, base_name: str) -> bool:
        return (
            base_name in cls.declared_names()
            or any(base_name.startswith(prefix) for prefix in cls.declared_name_prefixes())
            or cls.feature_name_equal_to_class_name(base_name)
            or cls.feature_name_contains_class_name_as_prefix(base_name)
            or base_name in cls.feature_names_supported()
        )

    @classmethod
    def _listed_columns(cls, match: SourceMatch) -> str:
        columns = cls.columns(match)
        if columns is not None:
            return _listed(columns)
        reason = cls.unknown_columns_reason(match)
        return "unknown" if reason is None else f"unknown ({reason})"

    @classmethod
    def _abort(cls, error: ValueError, deferrable: bool = False) -> bool:
        """Raise the match abort, or decline with a rejection when the global filter probe contains it."""
        if aborts_are_contained():
            record_match_rejection(cls.get_class_name(), str(error), stage=INPUT_DATA_STAGE)
            return False
        raise defer_match_abort(error) if deferrable else escalate_match_abort(error)

    @classmethod
    def _matches_by_default_rules(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> bool:
        name = str(feature_name)
        if inspect.isabstract(cls):
            if cls.is_pointed(name, options, data_access_collection):
                reason = f"{cls.get_class_name()} is abstract; missing {sorted(cls.__abstractmethods__)}"
                record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
            return False
        base_name = cls.get_column_base_feature(name)
        pointed = cls.is_pointed(name, options, data_access_collection)
        fitting: dict[str, SourceMatch] = {}
        seen: dict[str, SourceMatch] = {}
        undeclared = False
        for route in cls.CLAIM_ROUTES:
            if not all(key in options for key in route.required_options):
                continue
            unlocked = pointed or bool(route.required_options)
            if not (route.searched or unlocked):
                continue
            for match in cls.find_sources(route, name, options, data_access_collection):
                if route.names is NamePolicy.CHECKED:
                    seen[match.source] = match
                    if cls.has_column(match, base_name):
                        fitting[match.source] = match
                elif route.names is NamePolicy.OPEN or cls._declared_name(base_name):
                    fitting[match.source] = match
                else:
                    undeclared = True
        if (len(fitting) > 1 or (not fitting and seen and pointed)) and not cls._passes_option_declarations(options):
            return False
        if len(fitting) > 1:
            return cls._abort(
                ValueError(
                    f"{cls.get_class_name()} found feature '{name}' in several sources: "
                    f"{', '.join(sorted(fitting))}; "
                    f"{cls.ambiguity_fix(name, list(fitting.values()), data_access_collection)}"
                )
            )
        if len(fitting) == 1:
            options.add_to_group(RESERVED_READER_OPTION_KEY, (cls, next(iter(fitting.values()))))
            return True
        if seen:
            described = "; ".join(
                f"{source} has columns {cls._listed_columns(match)}" for source, match in sorted(seen.items())
            )
            reason = f"column '{base_name}' is in none of the sources of {cls.get_class_name()}: {described}"
            if pointed:
                return cls._abort(
                    ValueError(f"{reason}; request a column a source has or point at another source."),
                    deferrable=True,
                )
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        elif pointed and undeclared:
            reason = f"'{base_name}' is not a name {cls.get_class_name()} declares"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        elif pointed:
            reason = f"{cls.get_class_name()} was pointed at but found no source for '{name}'"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        return False
