import logging
from abc import ABC
from typing import Any, ClassVar, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_chainer.property_spec import PropertySpec, is_no_default
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.match_rejection import (
    INPUT_DATA_OWNED_STAGE,
    INPUT_DATA_STAGE,
    record_match_rejection,
    restamp_match_rejection,
)
from mloda.core.abstract_plugins.components.options import Options


from mloda.core.abstract_plugins.components.utils import (
    contained_raise_log_level,
    contained_raise_reason,
    escalate_match_abort,
    get_all_subclasses,
)

logger = logging.getLogger(__name__)


class BaseInputData(ABC):
    READER_OPTIONS: ClassVar[dict[str, PropertySpec]] = {
        "BaseInputData": PropertySpec(
            "The matched (ReaderClass, data_access) pair, written by add_base_input_data_to_options "
            "and read back by init_reader.",
            default=None,
            framework_set=True,
        ),
    }

    _reader_option_specs_cache: ClassVar[Optional[dict[str, PropertySpec]]] = None
    """Merged declarations of ONE class, written into that class's own __dict__ by _merged_reader_option_specs."""

    def __init__(self) -> None:
        pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._validate_reader_options()

    @classmethod
    def _validate_reader_options(cls) -> None:
        """Reject a reader-invalid declaration where it is written, not later deep in reader matching;
        checks only this class's own declaration so a bad child under a valid parent still raises."""
        for key, spec in cls.__dict__.get("READER_OPTIONS", {}).items():
            if not isinstance(spec, PropertySpec):
                raise ValueError(
                    f"{cls.__name__}.READER_OPTIONS['{key}'] is a {type(spec).__name__}, not a PropertySpec. "
                    f"Construct PropertySpec(...) instead."
                )
            if key == "BaseInputData" and not spec.framework_set:
                raise ValueError(
                    f"{cls.__name__}.READER_OPTIONS['{key}'] redeclares the reserved key without "
                    f"framework_set=True; the framework writes this key itself, so the admit path would "
                    f"judge a value no user ever supplies."
                )
            if spec.match_guard is not None:
                raise ValueError(
                    f"{cls.__name__}.READER_OPTIONS['{key}'] declares a match_guard, which is name-matching "
                    f"machinery and silently inert on a reader."
                )
            if spec.deferred_binding:
                raise ValueError(
                    f"{cls.__name__}.READER_OPTIONS['{key}'] declares deferred_binding=True, which is the "
                    f"name-capture exemption; a reader key has no name path."
                )
            if spec.context is False:
                raise ValueError(
                    f"{cls.__name__}.READER_OPTIONS['{key}'] declares context=False, which places a "
                    f"materialized value; readers place none."
                )
            if spec.framework_set:
                if spec.strict_validation:
                    raise ValueError(
                        f"{cls.__name__}.READER_OPTIONS['{key}'] combines framework_set=True with "
                        f"strict_validation=True; the framework-written key is exempt from user-value "
                        f"validation, so strictness would be silently inert."
                    )
                if spec.required_when is not None:
                    raise ValueError(
                        f"{cls.__name__}.READER_OPTIONS['{key}'] combines framework_set=True with a "
                        f"required_when predicate; the framework-written key is exempt from user-value "
                        f"validation, so the predicate would be silently inert."
                    )
                if spec.allow_explicit_none:
                    raise ValueError(
                        f"{cls.__name__}.READER_OPTIONS['{key}'] combines framework_set=True with "
                        f"allow_explicit_none=True; the admit path skips the framework-written key, "
                        f"so the flag cannot affect reader selection."
                    )
                if is_no_default(spec.default):
                    raise ValueError(
                        f"{cls.__name__}.READER_OPTIONS['{key}'] declares framework_set=True without a "
                        f"declared default; the framework-written key must declare its absent-state "
                        f"default explicitly, None included."
                    )

    @classmethod
    def _merged_reader_option_specs(cls) -> dict[str, PropertySpec]:
        """The merged declarations of cls, cached in cls's OWN __dict__; internal, never handed out.

        Reader selection is a hot path, so the MRO walk runs once per class. The cache is read back
        with cls.__dict__.get so a subclass never answers from a warm parent cache, and it lives on
        the class itself rather than in a class-keyed registry so it holds no reference to any class.
        """
        cached: Optional[dict[str, PropertySpec]] = cls.__dict__.get("_reader_option_specs_cache")
        if cached is not None:
            return cached

        merged: dict[str, PropertySpec] = {}
        for klass in reversed(cls.__mro__):
            merged.update(klass.__dict__.get("READER_OPTIONS", {}))
        cls._reader_option_specs_cache = merged
        return merged

    @classmethod
    def reader_option_specs(cls) -> dict[str, PropertySpec]:
        """The declarations of this reader family, most-derived winning; a fresh dict per call.

        Merged across cls.__mro__ from each class's own __dict__ so an inherited attribute is not
        merged twice, walking the MRO in reverse so a derived declaration wins on a key collision.
        """
        return dict(cls._merged_reader_option_specs())

    @classmethod
    def declared_reader_option_keys(cls) -> frozenset[str]:
        """Every option key this reader family declares."""
        return frozenset(cls._merged_reader_option_specs())

    @classmethod
    def _declared_reader_option_spec(cls, key: str) -> PropertySpec:
        """The declaration of key; an undeclared key is a typo, not a silent None."""
        specs = cls._merged_reader_option_specs()
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
    def reader_option(cls, key: str, options: Optional[Options]) -> Any:
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
    def _reader_options_admit(cls, options: Optional[Options], record_absence: bool) -> bool:
        """Check this candidate's merged declarations BEFORE its probe runs; a veto is its own non-match.
        record_absence doubles as the ownership signal: it gates absence recordings and stages present-value ones."""
        for key, spec in cls._merged_reader_option_specs().items():
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
    def _absent_reader_option_admits(
        cls, key: str, spec: PropertySpec, options: Optional[Options], record_absence: bool
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
                        f"(required_when predicate {getattr(predicate, '__name__', repr(predicate))} is satisfied)",
                        stage=INPUT_DATA_OWNED_STAGE,
                    )
                return False
            return True
        if is_no_default(spec.default):
            if record_absence:
                record_match_rejection(
                    owner,
                    f"required reader option '{key}' is absent, but {owner} declares it required (no default declared)",
                    stage=INPUT_DATA_OWNED_STAGE,
                )
            return False
        return True

    @classmethod
    def _present_reader_option_admits(cls, key: str, spec: PropertySpec, value: Any, owned: bool) -> bool:
        """Strict validation of a PRESENT value: list/tuple/set/frozenset unpack element-wise,
        a str is one scalar, a dict one composite value; owned stages the recorded rejection."""
        elements = list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]
        for element in elements:
            if cls._reader_option_element_admits(key, spec, element):
                continue
            owner = cls.get_class_name()
            record_match_rejection(
                owner,
                f"reader option '{key}' value {element!r} is rejected by the declaration of {owner}",
                stage=INPUT_DATA_OWNED_STAGE if owned else INPUT_DATA_STAGE,
            )
            return False
        return True

    @classmethod
    def _reader_option_element_admits(cls, key: str, spec: PropertySpec, element: Any) -> bool:
        """One element's verdict: a declared element_validator REPLACES membership."""
        validator = spec.element_validator
        if validator is not None:
            try:
                return bool(validator(element))
            except Exception as exc:  # Swallows: a validator that raises cannot judge the value, so it is rejected.
                logger.log(
                    contained_raise_log_level(exc),
                    "element_validator for reader option '%s' of %s %s; treating value as rejected.",
                    key,
                    cls.get_class_name(),
                    contained_raise_reason(exc),
                )
                return False
        try:
            return spec.allowed_values is not None and element in spec.allowed_values
        # Swallows: an unhashable element can never be a member, so the TypeError is a clean rejection.
        except TypeError:
            return False

    @classmethod
    def data_access_name(cls) -> str:
        """This function should return the name of the data access."""
        return cls.__name__

    @staticmethod
    def _underlying(member: Any) -> Any:
        """Underlying function of a classmethod/staticmethod/plain override, for identity comparison."""
        return getattr(member, "__func__", member)

    @classmethod
    def _is_overridden(cls, base: type, method_name: str) -> bool:
        """Structurally check whether cls overrides method_name relative to base."""
        return cls._underlying(getattr(cls, method_name)) is not cls._underlying(getattr(base, method_name))

    def matches(
        self,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
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
        """
        We check for the feature scope data access if any child classes match the data access.
        """
        subclasses = get_all_filtered_subclasses(BaseInputData, cls)
        for subclass in subclasses:
            for key, value in options.items():
                _key = cls.deal_with_base_input_data_name_as_cls_or_str(key)

                if _key == subclass.data_access_name():
                    # The user addressed this reader family by name (ownership), so vetoes record as owned.
                    if not subclass._reader_options_admit(options, record_absence=True):
                        break
                    matched_data_access = subclass.match_subclass_data_access(value, [feature_name], options=options)  # type: ignore[attr-defined]
                    if matched_data_access:
                        cls.add_base_input_data_to_options(subclass, matched_data_access, options)
                        return True
                    # The addressed reader matched nothing, so a recorded content decline becomes an owned veto.
                    restamp_match_rejection(subclass.get_class_name(), INPUT_DATA_STAGE, INPUT_DATA_OWNED_STAGE)
                    break  # This case is if a feature requests an input feature, which should have scoped access.
        return False

    @classmethod
    def deal_with_base_input_data_name_as_cls_or_str(cls, key: Any) -> str:
        if hasattr(key, "get_class_name"):
            if not issubclass(key, BaseInputData):
                # Contained: this runs per candidate over every option key, so an odd key is a non-match (#845).
                raise ValueError(f"Key {key} is not a subclass of BaseInputData.")
            key = key.get_class_name()

        if not isinstance(key, str):
            # Contained: this runs per candidate over every option key, so one odd key must not abort the run (#845).
            raise ValueError(f"Key {key} is not a string.")
        return key

    @classmethod
    def global_scope_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection],
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
        options: Optional[Options] = None,
    ) -> tuple[Any, Any]:
        """
        We check for data access collection if any child classes match the data access.
        """
        subclasses = get_all_filtered_subclasses(BaseInputData, cls)

        for subclass in subclasses:
            # A global probe never established ownership, so a silent absence veto cannot displace a real near-miss.
            if not subclass._reader_options_admit(options, record_absence=False):
                continue
            matched_data_access = subclass.match_subclass_data_access(  # type: ignore[attr-defined]
                data_access_collection, feature_names, options=options
            )
            if matched_data_access:
                return (subclass, matched_data_access)
        return None, None

    @classmethod
    def add_base_input_data_to_options(
        cls, cls_to_be_added: type["BaseInputData"], matched_data_access: Any, options: Options
    ) -> None:
        """
        Adding the found data access class to the options.
        """

        if "BaseInputData" in options:
            existing_data = options.get("BaseInputData")
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
        options.add_to_group("BaseInputData", (cls_to_be_added, matched_data_access))

    def init_reader(self, options: Optional[Options]) -> tuple["BaseInputData", Any]:
        if options is None:
            raise ValueError(
                f"Options were not set for {self.__class__.__name__}.init_reader().\n"
                "Provide an Options object with a 'BaseInputData' key mapping to a tuple of "
                "(ReaderClass, data_access).\n"
                "Example:\n"
                "  options = Options(context={\n"
                "      'BaseInputData': (ReaderClass, data_access)\n"
                "  })"
            )

        reader_data_access = options.get("BaseInputData")

        if reader_data_access is None:
            raise ValueError(
                f"'BaseInputData' key is missing in the provided Options for {self.__class__.__name__}.\n"
                "The 'BaseInputData' key in Options must map to a tuple of "
                "(ReaderClass, data_access).\n"
                "Example:\n"
                "  options = Options(context={\n"
                "      'BaseInputData': (ReaderClass, data_access)\n"
                "  })"
            )

        reader, data_access = reader_data_access
        return reader(), data_access

    def load(self, features: FeatureSet) -> Any:
        _options = None
        for feature in features.features:
            if _options:
                if _options != feature.options:
                    raise ValueError("All features must have the same options.")
            _options = feature.options

        reader, data_access = self.init_reader(_options)
        data = reader.load_data(data_access, features)

        if data is None:
            raise ValueError(f"Loading data failed for feature {features.get_name_of_one_feature()}.")

        return data

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """
        This function should be implemented in final child classes, which use scoped data access.
        """
        raise NotImplementedError

    @classmethod
    def _final_reader_requires(cls) -> tuple[str, ...]:
        """
        A family base redeclares this to name the hooks a subclass must override (relative to
        that family base) to classify as a final reader; the load_data wholesale-override
        branch always wins.
        """
        return ()

    @classmethod
    def final_reader_anchor(cls) -> type["BaseInputData"]:
        """
        The most-derived class in cls.__mro__ that declares _final_reader_requires in its own
        __dict__; BaseInputData declares the default, so an anchor always exists.
        """
        return next(klass for klass in cls.__mro__ if "_final_reader_requires" in klass.__dict__)

    @classmethod
    def is_final_reader(cls) -> bool:
        """
        Structurally classify whether cls is a final scoped reader; nothing is executed.

        A wholesale load_data override relative to the anchor is always final; otherwise cls
        is final iff the anchor's required hooks are non-empty and all overridden relative to
        the anchor. A class that declares _final_reader_requires is a family base and is
        therefore never final itself: both branches compare relative to the anchor, and
        nothing is overridden relative to itself.
        """
        anchor = cls.final_reader_anchor()
        if cls._is_overridden(anchor, "load_data"):
            return True
        required = cls._final_reader_requires()
        for name in required:
            if not hasattr(anchor, name):
                # Contained: this runs over every registered reader, so one broken plugin must not abort every run.
                raise ValueError(
                    f"Required final-reader hook '{name}' is not defined on anchor class {anchor.__name__}."
                )
        return bool(required) and all(cls._is_overridden(anchor, hook) for hook in required)

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        return True

    @classmethod
    def _has_suffix(cls) -> bool:
        """Check if this class implements suffix() (concrete subclass vs abstract base)."""
        try:
            cls.suffix()  # type: ignore[attr-defined]
            return True
        # Swallows: the probe asks whether suffix() is implemented, and both classes ARE that answer.
        except (NotImplementedError, AttributeError):
            return False

    @classmethod
    def _matches_suffix(cls, path: str) -> bool:
        """Check if a file path matches this class's suffix, or True if no suffix defined."""
        if not cls._has_suffix():
            return True
        return path.endswith(cls.suffix())  # type: ignore[attr-defined]

    @classmethod
    def _resolve_pinned_file(cls, data_access: Any, feature_names: list[str]) -> Optional[str]:
        column_map: dict[str, str] = data_access.column_to_file
        files_registry: dict[str, str] = data_access.files
        pinned_handles: set[str] = {column_map[name] for name in feature_names if name in column_map}
        if not pinned_handles:
            return None
        for name in feature_names:
            if name not in column_map:
                # Marked: containing a half-pinned batch would hand the feature to a reader that ignores the pins.
                raise escalate_match_abort(
                    ValueError(f"Mixed batch: some features pinned, others not: {feature_names}")
                )
        pinned_paths: set[str] = {files_registry[h] for h in pinned_handles}
        if len(pinned_paths) == 1:
            pinned_path: str = next(iter(pinned_paths))
            if not cls._matches_suffix(pinned_path):
                return None
            if cls.validate_columns(pinned_path, feature_names) is False:
                return None
            return pinned_path
        valid_candidates: list[str] = [
            path
            for path in pinned_paths
            if cls._matches_suffix(path) and cls.validate_columns(path, feature_names) is not False
        ]
        if len(valid_candidates) == 1:
            return valid_candidates[0]
        # Marked: same as the mixed batch above.
        raise escalate_match_abort(ValueError(f"Features in batch are pinned to different files: {pinned_paths}"))


def _collect_filtered_subclasses(cls: Any, parent_class: Any) -> list[type[BaseInputData]]:
    result = []
    for subclass in get_all_subclasses(cls):
        if not issubclass(subclass, parent_class):
            continue
        if subclass.is_final_reader():
            result.append(subclass)
    return result


def get_all_filtered_subclasses(cls: Any, parent_class: Any) -> list[type[BaseInputData]]:
    filtered_subclasses = _collect_filtered_subclasses(cls, parent_class)
    if not filtered_subclasses:
        auto_load_group = getattr(parent_class, "_auto_load_group", None)
        if auto_load_group is not None:
            from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

            if auto_load_group not in PluginLoader._disabled_groups:
                PluginLoader().load_group(auto_load_group)
                filtered_subclasses = _collect_filtered_subclasses(cls, parent_class)
    return filtered_subclasses
