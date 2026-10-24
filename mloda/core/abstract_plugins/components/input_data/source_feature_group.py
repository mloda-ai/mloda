"""Shared base of one FeatureGroup per kind of source: claim routes decide matching, hooks discover and load sources."""

import difflib
import inspect
import types
from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from mloda.core.abstract_plugins.components.credential_scrub import scrub_credentials
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
    computed_elsewhere,
    current_feature_group_scope,
    feature_group_scope,
    note_open_claim,
)
from mloda.core.abstract_plugins.components.input_data.reader_probe import SourceProbeOutcome, probe_source_group
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_STAGE,
    MATCH_REJECTION_REASONS,
    collect_match_rejections,
    drop_match_rejections_since,
    match_rejection_owners,
    record_match_rejection,
    replace_probe_match_rejection,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.property_spec import PropertySpec
from mloda.core.abstract_plugins.components.utils import defer_match_abort, escalate_match_abort
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda.core.abstract_plugins.components.data_types import DataType

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.implementation import Implementation

_MAX_LISTED_COLUMNS = 20
_MAX_DESCRIBED = 5
_MAX_COLUMN_NAME = 64
_MAX_REASON = 200
FOLDER_ESCAPE_MARK = "links outside its folder"
REASON_DETAIL = "\x1f"  # splits a reason lead from detail (reader error text) that only pointed users see

QUERY_OPTION = "query_text"
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


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _suggestion(column: str, known: list[str]) -> str:
    close = difflib.get_close_matches(column, sorted(set(known)), n=3, cutoff=0.6)
    names = ", ".join(repr(_clip(scrub_credentials(name), _MAX_COLUMN_NAME)) for name in close)
    return f"; did you mean {names}?" if close else ""


def _listed(columns: Collection[str] | None) -> str:
    if columns is None:
        return "unknown"
    names = sorted(_clip(scrub_credentials(column), _MAX_COLUMN_NAME) for column in columns)
    text = ", ".join(names[:_MAX_LISTED_COLUMNS])
    if len(names) > _MAX_LISTED_COLUMNS:
        text += f", ... and {len(names) - _MAX_LISTED_COLUMNS} more"
    return f"[{text}]"


def implementation_path(feature_group: type, implementation: Any, match: Any) -> str | None:
    """module:qualname of the implementation, "neutral" for a source group with a match and none, else None."""
    if implementation is not None:
        return f"{implementation.__module__}:{implementation.__qualname__}"
    if match is not None and issubclass(feature_group, SourceFeatureGroup):
        return "neutral"
    return None


class SourcePointerError(ValueError):
    """A class-name pointer or feature_group= scope that cannot be served; validated before matching."""


def concrete_source_groups() -> dict[str, type["SourceFeatureGroup"]]:
    """Loaded concrete source groups by class name, including ones a run's filters dropped."""
    found: dict[str, type[SourceFeatureGroup]] = {}
    stack: list[Any] = list(SourceFeatureGroup.__subclasses__())
    while stack:
        current = stack.pop()
        stack.extend(current.__subclasses__())
        if not inspect.isabstract(current):
            found[current.__name__] = current
    return found


class SourceFeatureGroup(FeatureGroup):
    """A FeatureGroup that claims features of one kind of source through its CLAIM_ROUTES."""

    CLAIM_ROUTES: ClassVar[tuple[ClaimRoute, ...]] = ()
    _DECLARED_PROPERTY_MAPPING: ClassVar[Mapping[Any, Any] | None] = None  # raw own declaration, pre-merge

    @classmethod
    def _merge_property_mapping(cls) -> None:
        has_own = "PROPERTY_MAPPING" in cls.__dict__
        own = cls.__dict__.get("PROPERTY_MAPPING")
        if has_own and not isinstance(own, Mapping):
            raise TypeError(
                f"{cls.__name__}.PROPERTY_MAPPING must be a mapping of PropertySpec, got {type(own).__name__}."
            )
        merged: dict[Any, Any] = {}
        for klass in reversed(cls.__mro__):
            if klass is cls and has_own:
                mapping = own
            elif "_DECLARED_PROPERTY_MAPPING" in klass.__dict__:
                mapping = klass.__dict__["_DECLARED_PROPERTY_MAPPING"]
            else:
                mapping = klass.__dict__.get("PROPERTY_MAPPING")
            if mapping is None:
                continue
            if not isinstance(mapping, Mapping):
                raise TypeError(
                    f"{cls.__name__}.PROPERTY_MAPPING must be a mapping of PropertySpec, got {type(mapping).__name__}."
                )
            merged.update(mapping)
        cls._DECLARED_PROPERTY_MAPPING = own if has_own else None
        if has_own or (merged and merged != cls.PROPERTY_MAPPING):
            cls.PROPERTY_MAPPING = merged

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls._merge_property_mapping()
        super().__init_subclass__(**kwargs)
        for route in cls.__dict__.get("CLAIM_ROUTES", ()):
            if route.names is NamePolicy.OPEN and route.searched:
                raise TypeError(
                    f"{cls.__name__} declares an OPEN route that is searched; an open route would claim every "
                    "name, so it must be pointed-only (searched=False)."
                )

    @classmethod
    def serves_query_text(cls) -> bool:
        """True when a claim route requires query_text."""
        return any(QUERY_OPTION in route.required_options for route in cls.CLAIM_ROUTES)

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
        """The neutral form of the source, used when no Implementation serves the current framework."""

    @classmethod
    def columns(cls, match: SourceMatch) -> Collection[str] | None:
        """Column names of the source, None when it cannot enumerate."""
        return None

    @classmethod
    def match_connection(cls, match: SourceMatch) -> Any | None:
        """The connection object the match reads from, None when the source is not a live connection."""
        return None

    @classmethod
    def unknown_columns_reason(cls, match: SourceMatch) -> str | None:
        """Why the source cannot enumerate its columns, None when no reason is known."""
        return None

    @classmethod
    def _hidden_reason(cls, match: SourceMatch) -> str | None:
        """The unknown_columns_reason safe to show a user who did not point at the group."""
        return cls.unknown_columns_reason(match)

    @classmethod
    def _hidden_source(cls, match: SourceMatch) -> str:
        """The source label shown to a user who did not point at the group."""
        return match.source

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
        """Own class name, then concrete source group ancestors, most-derived first."""
        return tuple(
            klass.__name__
            for klass in cls.__mro__
            if issubclass(klass, SourceFeatureGroup)
            and klass is not SourceFeatureGroup
            and (klass is cls or not inspect.isabstract(klass))
        )

    @classmethod
    def has_query_text(cls, options: Options) -> bool:
        """True when the options carry a query_text."""
        return QUERY_OPTION in options

    @classmethod
    def foreign_pointer(cls, options: Options) -> str | None:
        """First option key naming a concrete source group that is not one of this group's pointer keys."""
        keys = sorted(key for key in options.keys() if isinstance(key, str))
        if not keys:
            return None
        groups = concrete_source_groups()
        own = cls.pointer_keys()
        return next((key for key in keys if key in groups and key not in own), None)

    @classmethod
    def _scope_points_here(cls, scope: Any) -> bool:
        """A scope points at this group when it names the group or a non-abstract source group ancestor."""
        if scope is None:
            return False
        for klass in cls.__mro__:
            if not (isinstance(klass, type) and issubclass(klass, SourceFeatureGroup)):
                continue
            if klass is not cls and inspect.isabstract(klass):
                continue
            if scope is klass or scope == klass.__name__:
                return True
        return False

    @classmethod
    def select_implementation(
        cls,
        compute_framework: type[ComputeFramework],
        match: SourceMatch,
        implementations: Iterable[type["Implementation[Any]"]],
    ) -> "type[Implementation[Any]] | None":
        """The implementation serving this group and framework: group MRO outer, framework MRO inner."""
        if not compute_framework.is_available():
            return None
        from mloda.core.abstract_plugins.implementation import serving_implementation

        return serving_implementation(cls, compute_framework, implementations, stop_after="load_neutral")

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Load the matched source with the plan's implementation, else its neutral form, via the hook."""
        match = features.input_data_match
        if match is None:
            raise ValueError(f"{cls.__name__}.calculate_feature found no input_data_match on the feature set.")
        source = match[1]
        implementation = features.implementation
        load: Callable[[SourceMatch, Any], Any]
        if implementation is not None:
            marker = features._load_marker
            framework = marker.framework if marker is not None else None
            if (
                framework is not None
                and framework.framework_connection_object is None
                and framework.fallback_connection_object is not None
            ):
                framework.set_framework_connection_object(framework.fallback_connection_object)

            def calculate(group: Any, match: SourceMatch, features: Any) -> Any:
                return implementation.calculate(group, match, features, framework)

            load = types.MethodType(calculate, cls)
        else:
            load = cls.load_neutral
        loader_name = implementation_path(cls, implementation, match)
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
    def _abort(
        cls,
        error: ValueError,
        deferrable: bool = False,
        candidates: tuple[str, ...] = (),
        refusal: Exception | None = None,
    ) -> bool:
        """Raise the match abort, or decline with a rejection when the global filter probe contains it."""
        if aborts_are_contained():
            record_match_rejection(
                cls.get_class_name(), str(error), stage=INPUT_DATA_STAGE, candidates=candidates, refusal=refusal
            )
            return False
        if refusal is not None:
            error.__cause__ = refusal
        raise defer_match_abort(error) if deferrable else escalate_match_abort(error)

    @classmethod
    def _probe_routes(
        cls,
        routes: list[ClaimRoute],
        name: str,
        base_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> tuple[dict[str, SourceMatch], dict[str, SourceMatch], bool, frozenset[str]]:
        fitting: dict[str, SourceMatch] = {}
        seen: dict[str, SourceMatch] = {}
        firm: set[str] = set()
        undeclared = False
        for route in routes:
            for match in cls.find_sources(route, name, options, data_access_collection):
                if route.names is NamePolicy.CHECKED:
                    seen[match.source] = match
                    if cls.has_column(match, base_name):
                        fitting[match.source] = match
                        firm.add(match.source)
                elif route.names is NamePolicy.OPEN:
                    fitting[match.source] = match
                    if cls._declared_name(base_name):
                        firm.add(match.source)
                elif cls._declared_name(base_name):
                    fitting[match.source] = match
                    firm.add(match.source)
                else:
                    undeclared = True
        return fitting, seen, undeclared, frozenset(fitting) - firm

    @classmethod
    def _decline_refused_probe(cls, outcome: SourceProbeOutcome, pointed: bool, known_owners: frozenset[str]) -> bool:
        drop_match_rejections_since(known_owners)
        message = f"refused at INPUT_DATA_MATCH: {outcome.refused}"
        if pointed:
            return cls._abort(ValueError(f"{cls.get_class_name()} {message}"), deferrable=True, refusal=outcome.error)
        record_match_rejection(cls.get_class_name(), message, stage=INPUT_DATA_STAGE, refusal=outcome.error)
        return False

    @classmethod
    def _reject_abstract(cls, name: str, options: Options, data_access_collection: DataAccessCollection | None) -> str:
        """Record the abstract-class rejection when pointed; returns its reason."""
        reason = f"{cls.get_class_name()} is abstract; missing {sorted(cls.__abstractmethods__)}"
        if cls.is_pointed(name, options, data_access_collection):
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        return reason

    @classmethod
    def gated_probe(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
        probe: Callable[[], Any],
    ) -> SourceProbeOutcome:
        """Run probe inside the INPUT_DATA_MATCH hook; a refusal declines or aborts like core's own probe."""
        name = str(feature_name)
        if inspect.isabstract(cls):
            return SourceProbeOutcome(cls._reject_abstract(name, options, data_access_collection), None)
        pointed = cls.is_pointed(name, options, data_access_collection)
        known_owners = match_rejection_owners()
        outcome = probe_source_group(cls, probe, data_access_collection, cls.get_column_base_feature(name), options)
        if outcome.refused is not None:
            cls._decline_refused_probe(outcome, pointed, known_owners)
        return outcome

    @classmethod
    def has_source(
        cls,
        options: Options,
        data_access_collection: DataAccessCollection | None,
        feature_name: FeatureName | str = "",
    ) -> bool:
        """Whether a claim route that matching would select finds a source, without claiming or recording rejections.
        Runs behind the INPUT_DATA_MATCH gate when in scope; a refusal returns False."""
        if inspect.isabstract(cls):
            return False
        name = str(feature_name)
        with collect_match_rejections(), feature_group_scope(current_feature_group_scope(), contain_aborts=True):
            pointed = cls.is_pointed(name, options, data_access_collection)
            if not pointed and cls.foreign_pointer(options) is not None:
                return False
            if QUERY_OPTION in options and not cls.serves_query_text():
                return False
            routes = [
                route
                for route in cls.CLAIM_ROUTES
                if all(key in options for key in route.required_options)
                and (route.searched or pointed or bool(route.required_options))
            ]
            if not routes:
                return False
            outcome = probe_source_group(
                cls,
                lambda: any(cls.find_sources(route, name, options, data_access_collection) for route in routes),
                data_access_collection,
                cls.get_column_base_feature(name),
                options,
            )
        return outcome.refused is None and bool(outcome.result)

    @classmethod
    def _matches_by_default_rules(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> bool:
        name = str(feature_name)
        if inspect.isabstract(cls):
            cls._reject_abstract(name, options, data_access_collection)
            return False
        base_name = cls.get_column_base_feature(name)
        pointed = cls.is_pointed(name, options, data_access_collection)
        foreign = None if pointed else cls.foreign_pointer(options)
        query_declined = foreign is None and QUERY_OPTION in options and not cls.serves_query_text()
        query_reason = (
            f"{QUERY_OPTION} is set, but {cls.get_class_name()} lists no query route "
            "(database groups opt in with ReadDBFG.QUERY_ROUTE)"
        )
        if pointed and query_declined:
            return cls._abort(ValueError(query_reason), deferrable=True)
        skipped = False
        selected: list[ClaimRoute] = []
        for route in cls.CLAIM_ROUTES:
            if not all(key in options for key in route.required_options):
                continue
            unlocked = pointed or bool(route.required_options)
            if not (route.searched or unlocked):
                continue
            if foreign is not None:
                skipped = True
                continue
            if not unlocked and computed_elsewhere(name):
                continue
            selected.append(route)
        fitting: dict[str, SourceMatch] = {}
        seen: dict[str, SourceMatch] = {}
        undeclared = False
        open_only: frozenset[str] = frozenset()
        known_owners = match_rejection_owners()
        if selected:
            outcome = probe_source_group(
                cls,
                lambda: cls._probe_routes(selected, name, base_name, options, data_access_collection),
                data_access_collection,
                base_name,
                options,
            )
            if outcome.refused is not None:
                return cls._decline_refused_probe(outcome, pointed, known_owners)
            fitting, seen, undeclared, open_only = outcome.result
        if query_declined:
            if fitting or seen:
                replace_probe_match_rejection(cls.get_class_name(), query_reason, known_owners, stage=INPUT_DATA_STAGE)
            return False
        if (len(fitting) > 1 or (not fitting and seen and pointed)) and not cls.passes_option_declarations(options):
            return False
        show_columns = pointed or isinstance(options.get(HANDLE_OPTION), str)
        if len(fitting) > 1:
            if show_columns:
                listed = sorted(fitting)
            else:
                counts: dict[str, int] = {}
                for match in fitting.values():
                    label = cls._hidden_source(match)
                    counts[label] = counts.get(label, 0) + 1
                listed = sorted(label if n == 1 else f"{label} ({n} sources)" for label, n in counts.items())
            return cls._abort(
                ValueError(
                    scrub_credentials(
                        f"{cls.get_class_name()} found feature '{name}' in several sources: "
                        f"{', '.join(listed)}; "
                        f"{cls.ambiguity_fix(name, list(fitting.values()), data_access_collection)}"
                    )
                ),
                candidates=tuple(sorted(fitting)),
            )
        if len(fitting) == 1:
            reader_pair = (cls, next(iter(fitting.values())))
            # Contained: only a plugin's own matcher can write this key.
            if RESERVED_READER_OPTION_KEY in options.context or (
                RESERVED_READER_OPTION_KEY in options.group and reader_pair != options.group[RESERVED_READER_OPTION_KEY]
            ):
                held_value = options.context.get(
                    RESERVED_READER_OPTION_KEY, options.group.get(RESERVED_READER_OPTION_KEY)
                )
                held = (
                    held_value[0].__name__
                    if isinstance(held_value, tuple) and held_value and isinstance(held_value[0], type)
                    else type(held_value).__name__
                )
                raise ValueError(
                    f"{cls.get_class_name()} cannot record its reader: the reserved option already holds {held}"
                )
            options.add_to_group(RESERVED_READER_OPTION_KEY, reader_pair)
            if open_only:
                note_open_claim()
            return True
        if seen:
            items = sorted(seen.items())
            known = {source: cls.columns(match) for source, match in items}
            parts: list[str] = []
            if show_columns:
                described = [(source, source, match) for source, match in items]
            else:
                labelled: dict[str, tuple[str, SourceMatch]] = {}
                for source, match in items:
                    label = cls._hidden_source(match)
                    if label not in labelled or (known[labelled[label][0]] is None and known[source] is not None):
                        labelled[label] = (source, match)
                described = [(label, source, match) for label, (source, match) in labelled.items()]
            for label, source, match in described[:_MAX_DESCRIBED]:
                if known[source] is not None:
                    parts.append(f"{label} has columns {_listed(known[source])}" if show_columns else label)
                else:
                    why = cls.unknown_columns_reason(match) if show_columns else cls._hidden_reason(match)
                    parts.append(label if why is None else f"{label}: {_clip(scrub_credentials(why), _MAX_REASON)}")
            if len(described) > _MAX_DESCRIBED:
                parts.append(f"and {len(described) - _MAX_DESCRIBED} more")
            name = cls.get_class_name()
            head = (
                f"column '{base_name}' is in none of the sources of {name}"
                if any(columns is not None for columns in known.values())
                else f"{name} could not list the columns of any of its sources for column '{base_name}'"
            )
            text = f"{head}: {'; '.join(parts)}"
            if show_columns:
                text += _suggestion(base_name, [c for columns in known.values() for c in columns or ()])
            reason = scrub_credentials(text)
            probed = (MATCH_REJECTION_REASONS.get() or {}).get(cls.get_class_name())
            if probed is not None and FOLDER_ESCAPE_MARK in probed.reason and cls.get_class_name() not in known_owners:
                reason += "; skipped folder entries that link outside their folder"
            if pointed:
                return cls._abort(
                    ValueError(f"{reason}; request a column a source has or point at another source."),
                    deferrable=True,
                )
            replace_probe_match_rejection(cls.get_class_name(), reason, known_owners, stage=INPUT_DATA_STAGE)
        elif skipped:
            reason = f"options point at {foreign}, so {cls.get_class_name()} does not search the collection"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        elif pointed and undeclared:
            reason = f"'{base_name}' is not a name {cls.get_class_name()} declares"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        elif pointed:
            reason = f"{cls.get_class_name()} was pointed at but found no source for '{name}'"
            record_match_rejection(cls.get_class_name(), reason, stage=INPUT_DATA_STAGE)
        return False
