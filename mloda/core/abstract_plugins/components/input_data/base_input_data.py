import inspect
import logging
import os
import weakref
from abc import ABC
from collections.abc import Callable, Iterable, Mapping
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, ClassVar, cast

from mloda.core.abstract_plugins.components.credential_scrub import _URI_PATTERN, _uri_projection
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.data_types import DataType
from mloda.core.abstract_plugins.components.property_spec import PropertySpec, element_admitted, is_no_default
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.function_extender import ExtenderHook, GateBypassError, _invoke_extender
from mloda.core.abstract_plugins.hook_context import HookContext, input_data_load_gate_scopes_active, instrument
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    context_forwarding_remedy,
    drop_match_rejections_since,
    match_rejection_owners,
    record_match_rejection,
    restamp_match_rejections_since,
)
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.declaration_surface import (
    DeclarationSurface,
    merged_declaration,
    reject_merge_cache_assignment,
    validate_declaration,
)


from mloda.core.abstract_plugins.components.utils import (
    contained_raise_log_level,
    contained_raise_reason,
    escalate_match_abort,
    get_all_subclasses,
    safe_field,
    safe_value_text,
)
from mloda.core.abstract_plugins.components.declared_attributes import (
    current_declaration_requirement,
    read_declared_attributes,
)

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework

logger = logging.getLogger(__name__)

RESERVED_READER_OPTION_KEY = "BaseInputData"


def _format_keys(keys: Iterable[str]) -> str:
    return "{" + ", ".join(sorted(set(keys))) + "}"


_fallback_identity_warned: "weakref.WeakSet[type]" = weakref.WeakSet()


def _is_fallback_identity(data_access: Any, identity: str) -> bool:
    """True when identity is the access's type name or its bare mapping key names."""
    if identity == type(data_access).__name__:
        return True
    return isinstance(data_access, Mapping) and identity == _format_keys(str(key) for key in data_access)


def dispatch_input_data_load(
    owner: Any,
    owner_type: Any,
    load: Callable[[Any, FeatureSet], Any],
    data_access: Any,
    features: FeatureSet,
    *,
    identity: Callable[[], str],
    data_access_format: Callable[[], str],
    loader_name: str | None = None,
) -> Any:
    """Run load through the INPUT_DATA_LOAD extender when one is registered, instrumenting it with a HookContext
    that inherits identity fields from the active calculate-phase HookContext."""
    from mloda.core.abstract_plugins.compute_framework import ComputeFramework

    cfw = ComputeFramework.current()
    if cfw is None:
        if input_data_load_gate_scopes_active() > 0:
            raise GateBypassError(
                f"{owner_type.__qualname__}.load_data ran outside the calculation context while an "
                "INPUT_DATA_LOAD gate is active (a thread hop lost it); run the thread's work with "
                "contextvars.copy_context().run."
            )
        return load(data_access, features)

    extender = cfw.get_function_extender(ExtenderHook.INPUT_DATA_LOAD)
    if extender is None:
        return load(data_access, features)

    calc_context = HookContext.current()
    if calc_context is None:
        if extender.never_fall_back:
            raise GateBypassError(
                f"{owner_type.__qualname__}.load_data ran without a calculate-phase HookContext under an "
                "INPUT_DATA_LOAD gate; run the thread's work with contextvars.copy_context().run."
            )
        return load(data_access, features)

    access_identity = identity()
    is_fallback = _is_fallback_identity(data_access, access_identity)
    if is_fallback and owner_type not in _fallback_identity_warned:
        _fallback_identity_warned.add(owner_type)
        logger.warning(
            "%s.data_access_identity fell back to %r, which names no source, so INPUT_DATA_LOAD extenders "
            "cannot tell its sources apart; override data_access_identity on %s.",
            owner_type.__qualname__,
            access_identity,
            owner_type.__qualname__,
        )

    context = HookContext(
        hook=ExtenderHook.INPUT_DATA_LOAD,
        feature_group_class=calc_context.feature_group_class,
        feature_group_version=calc_context.feature_group_version,
        plugin_version=calc_context.plugin_version,
        feature_names=calc_context.feature_names,
        specialized_from=calc_context.specialized_from,
        input_features=calc_context.input_features,
        input_feature_edges=calc_context.input_feature_edges,
        compute_framework_name=cfw.get_class_name(),
        run_id=calc_context.run_id,
        carrier=calc_context.carrier,
        tenant_id=calc_context.tenant_id,
        project_id=calc_context.project_id,
        principal=calc_context.principal,
        worker_index=calc_context.worker_index,
        data_access_identity=access_identity,
        data_access_identity_is_fallback=is_fallback,
        data_access_format=data_access_format(),
        data_access_dataset_version=None,
        data_access_loader=loader_name,
        declared_attributes=safe_field(
            lambda: read_declared_attributes(owner, features),
            None,
            field=f"{owner_type.__qualname__}.declared_attributes",
            warn_once_for=owner_type,
        ),
        reader_class=owner_type,
    )
    with context.activate():
        return _invoke_extender(extender, instrument(context, load, row_count=cfw._row_count), data_access, features)


class BaseInputData(ABC):
    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        RESERVED_READER_OPTION_KEY: PropertySpec(
            "The matched (ReaderClass, data_access) pair, written transiently while matching and "
            "never kept in options; it lives on Feature.input_data_match.",
            default=None,
            framework_set=True,
        ),
    }

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guards run at class definition only: mutating READER_OPTIONS afterwards or overriding
        __init_subclass__ without calling super() defeats them."""
        # Checked before super(): a cooperative later-in-MRO hook may warm the cache during the
        # super() chain; here cls.__dict__ still holds only what the class body wrote.
        reject_merge_cache_assignment(cls, DeclarationSurface.READER)
        super().__init_subclass__(**kwargs)
        cls._validate_reader_options()

    @classmethod
    def _validate_reader_options(cls) -> None:
        cls._validate_reserved_reader_option_key()
        validate_declaration(cls, DeclarationSurface.READER, BaseInputData)

    @classmethod
    def _validate_reserved_reader_option_key(cls) -> None:
        """The reserved key must survive the MRO MERGE, not just cls's own dict: a plain mixin is never
        validated itself, yet its declaration outranks the base's and is what selection reads."""
        for klass in cls.__mro__:
            declared = klass.__dict__.get("READER_OPTIONS", {})
            # Shape is not this check's concern; own_declaration rejects it once validate_declaration reaches it.
            if not isinstance(declared, dict) or RESERVED_READER_OPTION_KEY not in declared:
                continue
            spec = declared[RESERVED_READER_OPTION_KEY]
            if not isinstance(spec, PropertySpec) or not spec.framework_set:
                raise ValueError(
                    f"{cls.__name__} merges READER_OPTIONS['{RESERVED_READER_OPTION_KEY}'] to the declaration on "
                    f"{klass.__name__}, which is not a PropertySpec with framework_set=True; the framework writes "
                    f"this reserved key itself, so the admit path would judge a value no user ever supplies."
                )
            return

    @classmethod
    def reader_option_specs(cls) -> dict[str, PropertySpec]:
        """The declarations of this reader, most-derived winning; a fresh dict per call."""
        return dict(merged_declaration(cls, DeclarationSurface.READER))

    @classmethod
    def declared_reader_option_keys(cls) -> frozenset[str]:
        """Every option key this reader declares."""
        return frozenset(merged_declaration(cls, DeclarationSurface.READER))

    @classmethod
    def _declared_reader_option_spec(cls, key: str) -> PropertySpec:
        """The declaration of key; an undeclared key is a typo, not a silent None."""
        specs = merged_declaration(cls, DeclarationSurface.READER)
        if key not in specs:
            raise ValueError(f"Reader option '{key}' is not declared in READER_OPTIONS of {cls.__name__}.")
        return specs[key]

    @classmethod
    def reader_option_default(cls, key: str) -> Any:
        """The declared default of key, without consulting any Options."""
        spec = cls._declared_reader_option_spec(key)
        if is_no_default(spec.default):
            raise ValueError(f"Reader option '{key}' of {cls.__name__} declares no default.")
        return spec.default

    @classmethod
    def reader_option(cls, key: str, options: Options | None) -> Any:
        """The supplied value of key when present, else the declared default; NO_DEFAULT raises.
        allow_explicit_none=True reads presence as ``key in options``; options=None reads all-absent."""
        spec = cls._declared_reader_option_spec(key)
        if options is not None:
            value = options.get(key)
            present = key in options if spec.allow_explicit_none else value is not None
            if present:
                return value
        if is_no_default(spec.default):
            raise ValueError(f"Reader option '{key}' of {cls.__name__} declares no default and no value was supplied.")
        return spec.default

    @classmethod
    def _reader_options_admit(cls, options: Options | None, record_absence: bool) -> bool:
        """Check this candidate's merged declarations BEFORE its probe runs; a veto is its own non-match.
        record_absence doubles as the ownership signal: it gates absence recordings and stages present-value ones."""
        for key, spec in merged_declaration(cls, DeclarationSurface.READER).items():
            if spec.framework_set:
                continue
            if options is None:
                if not cls._absent_reader_option_admits(key, spec, options, record_absence):
                    return False
                continue
            value = options.get(key)
            present = key in options if spec.allow_explicit_none else value is not None
            if not present:
                if not cls._absent_reader_option_admits(key, spec, options, record_absence):
                    return False
            elif spec.strict_validation:
                if not cls._present_reader_option_admits(key, spec, value, owned=record_absence):
                    return False
        return True

    @classmethod
    def _unmet_current_declaration(cls) -> str | None:
        """Reason this reader's declarations miss the consumer requirement in scope, else None."""
        requirement = current_declaration_requirement()
        if requirement is None:
            return None
        return requirement.unmet_reason(cls.get_class_name(), cls)

    @classmethod
    def _absent_reader_option_admits(
        cls, key: str, spec: PropertySpec, options: Options | None, record_absence: bool
    ) -> bool:
        """Requiredness of an ABSENT key: required_when decides when declared, else NO_DEFAULT rejects.
        record_absence says whether the veto is recorded."""
        owner = cls.get_class_name()
        predicate = spec.required_when
        if predicate is not None:
            try:
                is_required = bool(predicate(options if options is not None else Options()))
            # Swallows: a predicate that raises cannot judge, so the reader is a non-match, not the run.
            except Exception as exc:
                # Text, not exc: a retained record must not pin the traceback, its frames and the plugin class.
                logger.log(
                    contained_raise_log_level(exc),
                    "required_when predicate %s for reader option '%s' %s; treating reader %s as a non-match.",
                    getattr(predicate, "__name__", repr(predicate)),
                    key,
                    contained_raise_reason(exc),
                    owner,
                )
                return False
            if is_required:
                if record_absence:
                    record_match_rejection(
                        owner,
                        f"required reader option '{key}' is absent, but {owner} declares it required "
                        f"(required_when predicate {getattr(predicate, '__name__', repr(predicate))} is satisfied)"
                        f"{context_forwarding_remedy(spec.context)}",
                        stage=INPUT_DATA_OWNED_STAGE,
                    )
                return False
            return True
        if is_no_default(spec.default):
            if record_absence:
                record_match_rejection(
                    owner,
                    f"required reader option '{key}' is absent, but {owner} declares it required (no default declared)"
                    f"{context_forwarding_remedy(spec.context)}",
                    stage=INPUT_DATA_OWNED_STAGE,
                )
            return False
        return True

    @classmethod
    def _present_reader_option_admits(cls, key: str, spec: PropertySpec, value: Any, owned: bool) -> bool:
        """Strict validation of a PRESENT value: list/tuple/set/frozenset unpack element-wise,
        a str is one scalar, a dict one composite value; owned stages the recorded rejection.
        scalar_only=True short-circuits: a list/tuple/set/frozenset value is rejected outright, never unpacked."""
        if spec.scalar_only and isinstance(value, (list, tuple, set, frozenset)):
            owner = cls.get_class_name()
            record_match_rejection(
                owner,
                f"reader option '{key}' value is a {type(value).__name__} of {len(value)} elements, but the "
                f"declaration of {owner} marks it scalar_only and rejects a collection outright",
                stage=INPUT_DATA_OWNED_STAGE if owned else INPUT_DATA_STAGE,
            )
            return False
        elements = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
        for element in elements:
            if cls._reader_option_element_admits(key, spec, element):
                continue
            owner = cls.get_class_name()
            record_match_rejection(
                owner,
                f"reader option '{key}' value {safe_value_text(element)} is rejected by the declaration of {owner}",
                stage=INPUT_DATA_OWNED_STAGE if owned else INPUT_DATA_STAGE,
            )
            return False
        return True

    @classmethod
    def _reader_option_element_admits(cls, key: str, spec: PropertySpec, element: Any) -> bool:
        """One element's verdict: a declared element_validator REPLACES membership."""

        def on_raise(exc: Exception) -> None:
            logger.log(
                contained_raise_log_level(exc),
                "element_validator for reader option '%s' of %s %s; treating value as rejected.",
                key,
                cls.get_class_name(),
                contained_raise_reason(exc),
            )

        return element_admitted(spec, element, on_raise)

    @classmethod
    def data_access_name(cls) -> str:
        """This function should return the name of the data access."""
        return cls.__name__

    @classmethod
    def wrap_feature_scoped_access(cls, data_access: Any) -> Any:
        """Normalize a feature-scoped data access before matching; default is identity."""
        return data_access

    @classmethod
    def data_access_identity(cls, data_access: Any) -> str:
        """Mapping keys, a parsed URI's projection, an existing local path, else the type name."""
        if isinstance(data_access, Mapping):
            return _format_keys(str(key) for key in data_access)
        if isinstance(data_access, str):
            match = _URI_PATTERN.fullmatch(data_access)
            if match is not None:
                return _uri_projection(match) or type(data_access).__name__
        if isinstance(data_access, (str, PurePath)) and os.path.exists(data_access):
            return str(data_access)
        return type(data_access).__name__

    def matches(
        self,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        """
        We look if feature scope data access or global scope access is set.

        Feature scope access are set via options per feature,
        whereas global scope access is set via data_access_collection.
        """
        if self.feature_scope_data_access(options, feature_name) is True:
            return True

        if self.global_scope_data_access(feature_name, options, data_access_collection) is True:
            return True
        return False

    @classmethod
    def feature_scope_data_access(cls, options: Options, feature_name: str) -> bool:
        """True when an option keyed by this class's name carries an access this class accepts."""
        entry_owners = match_rejection_owners()
        for key, value in options.items():
            if cls.deal_with_base_input_data_name_as_cls_or_str(key) != cls.data_access_name():
                continue
            # The user addressed this class by name (ownership), so vetoes record as owned.
            before_checks = match_rejection_owners()
            if cls._reader_options_admit(options, record_absence=True):
                wrapped = cls.wrap_feature_scoped_access(value)
                if wrapped is not value:
                    options.set(key, wrapped)
                    value = wrapped
                known_owners = match_rejection_owners()
                matched_data_access = cls.match_subclass_data_access(  # type: ignore[attr-defined]
                    value, [feature_name], options=options
                )
                if matched_data_access:
                    unmet = cls._unmet_current_declaration()
                    if unmet is None:
                        # Records from this call must not mask a later failure reason.
                        drop_match_rejections_since(entry_owners)
                        cls.add_base_input_data_to_options(cls, matched_data_access, options)
                        return True
                    record_match_rejection(cls.get_class_name(), unmet, stage=INPUT_DATA_OWNED_STAGE)
                else:
                    # The addressed probe matched nothing, so whatever content decline it recorded becomes owned.
                    restamp_match_rejections_since(known_owners, INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)
            if match_rejection_owners() <= before_checks:
                name = cls.get_class_name()
                record_match_rejection(
                    name,
                    f"{name} is pinned for feature '{feature_name}' but matched nothing; a pinned source is final",
                    stage=INPUT_DATA_OWNED_STAGE,
                )
            return False
        return False

    @classmethod
    def deal_with_base_input_data_name_as_cls_or_str(cls, key: Any) -> str:
        if hasattr(key, "get_class_name"):
            if not issubclass(key, BaseInputData):
                # Contained: this runs per candidate over every option key, so an odd key is a non-match (#845).
                raise ValueError(f"Key {key} is not a subclass of BaseInputData.")
            # Options normalizes a class key the same way, so an overridden alias stays the one identity.
            key = key.data_access_name()

        if not isinstance(key, str):
            # Contained: this runs per candidate over every option key, so one odd key must not abort the run (#845).
            raise ValueError(f"Key {key} is not a string.")
        return key

    @classmethod
    def global_scope_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: DataAccessCollection | None,
    ) -> bool:
        if data_access_collection is None:
            return False

        if options.get(cls.data_access_name()):
            return False

        data_access_cls, matched_data_access = cls.match_data_access(
            [feature_name], data_access_collection, options=options
        )
        if data_access_cls is None:
            return False

        cls.add_base_input_data_to_options(data_access_cls, matched_data_access, options)
        return True

    @classmethod
    def match_data_access(
        cls,
        feature_names: list[str],
        data_access_collection: DataAccessCollection,
        options: Options | None = None,
    ) -> tuple[Any, Any]:
        """
        We check for data access collection if any child classes match the data access.
        """
        # A global probe never established ownership, so a silent absence veto cannot displace a real near-miss.
        if cls._reader_options_admit(options, record_absence=False):
            matched_data_access = cls.match_subclass_data_access(  # type: ignore[attr-defined]
                data_access_collection, feature_names, options=options
            )
            if matched_data_access:
                unmet = cls._unmet_current_declaration()
                if unmet is None:
                    return cls, matched_data_access
                record_match_rejection(cls.get_class_name(), unmet, stage=INPUT_DATA_OWNED_STAGE)

        cls._record_unowned_pin(data_access_collection, feature_names)
        return None, None

    @classmethod
    def _record_unowned_pin(cls, data_access_collection: DataAccessCollection, feature_names: list[str]) -> None:
        """Records one attributable elimination when no file format group owns the pinned
        file's suffix. Recorded at the owned stage since a plain-stage recording is never harvested once a
        name rule matches. Keyed apart from the candidate's own data_access_name() so an earlier plain
        rejection recorded under that same key in this window cannot silently absorb this one.
        """
        column_to_file = data_access_collection.column_to_file
        if column_to_file is None or not all(name in column_to_file for name in feature_names):
            return
        pinned_paths = {data_access_collection.files[column_to_file[name]] for name in feature_names}
        if len(pinned_paths) != 1:
            return
        pinned_path = next(iter(pinned_paths))
        from mloda.core.abstract_plugins.components.input_data.read_file_fg import ReadFileFG

        if any(
            not inspect.isabstract(group) and pinned_path.endswith(group.suffixes())
            for group in get_all_subclasses(ReadFileFG)
        ):
            return
        record_match_rejection(
            f"{cls.data_access_name()} (unowned pin)",
            f"pinned file {pinned_path} has a suffix no registered reader owns",
            stage=INPUT_DATA_OWNED_STAGE,
        )

    @classmethod
    def add_base_input_data_to_options(
        cls, cls_to_be_added: type["BaseInputData"], matched_data_access: Any, options: Options
    ) -> None:
        """
        Adding the found data access class to the options.
        """

        if RESERVED_READER_OPTION_KEY in options:
            existing_data = options.get(RESERVED_READER_OPTION_KEY)
            # `is True`, not a truth test: a non-bool __eq__ result (numpy array) must not raise unmarked here.
            if (existing_data == (cls_to_be_added, matched_data_access)) is True:
                return

            if isinstance(existing_data, tuple) and len(existing_data) == 2:
                existing_label = f"{existing_data[0]} (access type {type(existing_data[1]).__name__})"
            else:
                existing_label = type(existing_data).__name__

            # Marked: two conflicting readers for one feature is a user misconfiguration.
            # Keyed on presence so add_to_group cannot raise it unmarked; access named by type, it may hold secrets.
            raise escalate_match_abort(
                ValueError(
                    f"BaseInputData already set with different values. "
                    f"incoming={cls_to_be_added} (access type {type(matched_data_access).__name__}), "
                    f"existing={existing_label}"
                )
            )
        options.add_to_group(RESERVED_READER_OPTION_KEY, (cls_to_be_added, matched_data_access))

    def init_reader(self, match: tuple[type["BaseInputData"], Any]) -> tuple["BaseInputData", Any]:
        reader, data_access = match
        return reader(), data_access

    def load(self, features: FeatureSet) -> Any:
        match = features.input_data_match
        if match is None:
            raise ValueError(
                f"{self.__class__.__name__}.load() found no input_data_match on the feature set; "
                "the reader match is set while the feature group is identified."
            )

        reader, data_access = self.init_reader(cast("tuple[type[BaseInputData], Any]", match))
        data = self._load_data_via_hook(reader, data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    @staticmethod
    def _load_data_via_hook(reader: "BaseInputData", data_access: Any, features: FeatureSet) -> Any:
        """Dispatch reader.load_data through the INPUT_DATA_LOAD extender when one is registered."""
        return dispatch_input_data_load(
            reader,
            type(reader),
            reader.load_data,
            data_access,
            features,
            identity=lambda: reader.data_access_identity(data_access),
            data_access_format=reader.data_access_name,
        )

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        """Scalar attributes this reader declares on its extender hooks; empty by default."""
        return {}

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """
        This function should be implemented in final child classes, which use scoped data access.
        """
        raise NotImplementedError

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def describe_columns(cls, data_access: Any) -> dict[str, DataType | None]:
        """Maps column name to DataType (None if unknown; a duplicate name collapses to one entry). Raises
        NotImplementedError (cannot enumerate), ImportError (backend missing), or OSError/ValueError (unreadable)."""
        raise NotImplementedError

    @classmethod
    def count_rows(cls, data_access: Any, compute_framework: "type[ComputeFramework]") -> int | None:
        """Rows of data_access without loading its data; None when only a read can tell.
        Raises ImportError (backend missing), or OSError/ValueError (non-path, missing or unreadable source)."""
        return None
