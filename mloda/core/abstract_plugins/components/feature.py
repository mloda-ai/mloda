from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4
from mloda.core.abstract_plugins.components.data_types import DataType

if TYPE_CHECKING:
    from mloda.core.abstract_plugins.feature_group import FeatureGroup

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.hashable_dict import _make_hashable
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.options import Options, validate_forwarding_directives
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.components.validators.feature_validator import FeatureValidator


class Feature:
    """Represents a raw feature.

    Attributes:
        name (FeatureName): The name of the feature.
        options (Options): The options associated with the feature.
        domain (Optional[Domain]): The domain of the feature.
        compute_frameworks (Optional[Set[Type[ComputeFramework]]]): The compute frameworks supported by the feature.
        data_type (Optional[DataType]): The data type of the feature.
        initial_requested_data (bool): Whether the data was initially requested.
        link (Optional[Link]): The link associated with the feature.
        index (Optional[Index]): The index associated with the feature.
        feature_group_scope (str | type[FeatureGroup] | None): Resolution-only scope; excluded from identity.

    Quick start (recommended progression)::

        # 1. Bare strings -- simplest, no options or types
        mloda.run_all(["income", "age"])

        # 2. Feature() -- when you need options
        mloda.run_all([Feature("income", {"data_source": "prod"})])

        # 3. Typed helpers -- when you need type enforcement
        mloda.run_all([Feature.int32_of("age"), Feature.double_of("income")])

        # 4. Explicit Options with group/context -- advanced
        mloda.run_all([Feature("income", Options(
            group={"data_source": "prod"},
            context={"debug": True},
        ))])

    Options passed as a plain dict go into ``Options.group`` (affects
    feature group resolution). Use ``Options(context={...})`` for metadata
    that should not affect grouping.

    Class Methods (Convenience):
        not_typed(name, options): Creates a Feature instance without specifying a data type.
        str_of(name, options): Creates a Feature instance with STRING data type.
        int32_of(name, options): Creates a Feature instance with INT32 data type.
        int64_of(name, options): Creates a Feature instance with INT64 data type.
        float_of(name, options): Creates a Feature instance with FLOAT data type.
        double_of(name, options): Creates a Feature instance with DOUBLE data type.
        boolean_of(name, options): Creates a Feature instance with BOOLEAN data type.
        binary_of(name, options): Creates a Feature instance with BINARY data type.
        date_of(name, options): Creates a Feature instance with DATE data type.
        timestamp_millis_of(name, options): Creates a Feature instance with TIMESTAMP_MILLIS data type.
        timestamp_micros_of(name, options): Creates a Feature instance with TIMESTAMP_MICROS data type.
        decimal_of(name, options): Creates a Feature instance with DECIMAL data type.
    """

    def __init__(
        self,
        name: str | FeatureName,
        options: Optional[dict[str, Any] | Options] = None,
        domain: Optional[str] = None,
        compute_framework: Optional[str] = None,
        data_type: Optional[DataType | str] = None,
        initial_requested_data: bool = False,
        link: Optional[Link] = None,
        index: Optional[Index] = None,
        feature_group: str | type[FeatureGroup] | None = None,
        forward_group: frozenset[str] | set[str] | list[str] | tuple[str, ...] | bool | None = None,
        forward_group_exclude: frozenset[str] | set[str] | list[str] | tuple[str, ...] | None = None,
        inherit_context_keys: frozenset[str] | set[str] | list[str] | tuple[str, ...] = frozenset(),
    ):
        if options is None:
            options = {}
        self.name = FeatureName(name) if isinstance(name, str) else name
        self.options = Options(options) if isinstance(options, dict) else options
        self.domain = self._set_domain(domain, self.options.get("domain"))

        cf = self._set_compute_framework(compute_framework, self.options.get("compute_framework"))
        self.compute_frameworks = {cf} if cf else None

        self.uuid = uuid4()

        self.data_type = None
        if data_type is not None:
            if isinstance(data_type, DataType):
                self.data_type = data_type
            elif isinstance(data_type, str):
                self.data_type = DataType(data_type)
            else:
                raise TypeError(
                    f"data_type must be a DataType enum or a string matching a DataType member value, "
                    f"got {type(data_type).__name__}"
                )

        # Engine-stamped consumer metadata; in equality/hash via cycle-safe _child_options_key (#608).
        self.child_options: Optional[Options] = None

        self.initial_requested_data = initial_requested_data

        # LINK and INDEX are excluded from equality and hash, because this way, we can define a single feature of a group with these properties.
        self.link = link
        self.index = index  # Index is a feature currently only used for append/union features.

        # feature_group_scope is resolution-only metadata, excluded from equality and hash like link/index.
        self.feature_group_scope = self._set_feature_group_scope(feature_group)

        # Resolution-only metadata stamped by the engine: one (consumer class name, consumer
        # PROPERTY_MAPPING keys) entry appended per consumer feature group that declares this
        # feature as an input feature; excluded from equality and hash like link/index.
        self.consumer_attributions: list[tuple[str, frozenset[str]]] = []

        # forward_group, forward_group_exclude and inherit_context_keys are merge directives
        # for input features with forward-by-default semantics (None/True inherit all consumer
        # group options, False isolates, an allowlist restricts, exclude subtracts),
        # excluded from equality and hash like link/index.
        # Only the literal False triggers the contradiction guard with a non-empty
        # forward_group_exclude; an EMPTY allowlist frozenset combined with an exclude stays
        # legal because allowlists may be computed dynamically.
        self.forward_group: frozenset[str] | bool | None = (
            forward_group
            if forward_group is None or isinstance(forward_group, bool)
            else self._normalize_allowlist(forward_group, "forward_group")
        )
        self.forward_group_exclude: frozenset[str] = (
            frozenset()
            if forward_group_exclude is None
            else self._normalize_allowlist(forward_group_exclude, "forward_group_exclude")
        )
        validate_forwarding_directives(self.forward_group, self.forward_group_exclude)
        self.inherit_context_keys = self._normalize_allowlist(inherit_context_keys, "inherit_context_keys")

    def add_consumer_attribution(self, name: str, keys: frozenset[str]) -> None:
        """Record a consumer attribution, skipping an identical (name, keys) entry.

        Appended per consumer feature group that declares this feature as an input feature.
        Idempotent so re-stamping the same Feature instance across mloda runs does not grow
        the list unboundedly.
        """
        entry = (name, keys)
        if entry not in self.consumer_attributions:
            self.consumer_attributions.append(entry)

    @staticmethod
    def _normalize_allowlist(
        value: frozenset[str] | set[str] | list[str] | tuple[str, ...], param_name: str
    ) -> frozenset[str]:
        if isinstance(value, str) or not isinstance(value, (frozenset, set, list, tuple)):
            raise TypeError(f"{param_name} must be a set, frozenset, list, or tuple of str, got {type(value).__name__}")
        for element in value:
            if not isinstance(element, str):
                raise TypeError(f"{param_name} elements must be str, got {type(element).__name__}")
        return frozenset(value)

    def _set_feature_group_scope(
        self, feature_group: str | type[FeatureGroup] | None
    ) -> str | type[FeatureGroup] | None:
        if feature_group is None:
            return None
        if isinstance(feature_group, str):
            stripped = feature_group.strip()
            return stripped or None
        from mloda.core.abstract_plugins.feature_group import FeatureGroup

        if feature_group is FeatureGroup:
            raise TypeError("feature_group cannot be the root FeatureGroup base class; a concrete subclass is required")
        if isinstance(feature_group, type) and issubclass(feature_group, FeatureGroup):
            return feature_group
        raise TypeError(
            f"feature_group must be a FeatureGroup subclass, a class-name string, or None, "
            f"got {type(feature_group).__name__}"
        )

    @classmethod
    def not_typed(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> Feature:
        if options is None:
            options = {}
        name = FeatureName(name) if isinstance(name, str) else name
        return cls(name=name, options=options, feature_group=feature_group)

    @classmethod
    def str_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> Feature:
        return cls._typed_of(name, DataType.STRING, options, feature_group)

    @classmethod
    def int32_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> Feature:
        return cls._typed_of(name, DataType.INT32, options, feature_group)

    @classmethod
    def int64_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.INT64, options, feature_group)

    @classmethod
    def float_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.FLOAT, options, feature_group)

    @classmethod
    def double_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.DOUBLE, options, feature_group)

    @classmethod
    def boolean_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.BOOLEAN, options, feature_group)

    @classmethod
    def binary_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.BINARY, options, feature_group)

    @classmethod
    def date_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.DATE, options, feature_group)

    @classmethod
    def timestamp_millis_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.TIMESTAMP_MILLIS, options, feature_group)

    @classmethod
    def timestamp_micros_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.TIMESTAMP_MICROS, options, feature_group)

    @classmethod
    def decimal_of(
        cls,
        name: str | FeatureName,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> "Feature":
        return cls._typed_of(name, DataType.DECIMAL, options, feature_group)

    @classmethod
    def _typed_of(
        cls,
        name: str | FeatureName,
        data_type: DataType,
        options: Optional[dict[str, Any]] = None,
        feature_group: str | type[FeatureGroup] | None = None,
    ) -> Feature:
        if options is None:
            options = {}
        name = FeatureName(name) if isinstance(name, str) else name
        return cls(name=name, data_type=data_type, options=options, feature_group=feature_group)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Feature):
            return False
        return (
            self.name == other.name
            and self.options == other.options
            and self.options.context == other.options.context
            and self.domain == other.domain
            and self.compute_frameworks == other.compute_frameworks
            and self.data_type == other.data_type
            and self._child_options_key() == other._child_options_key()
        )

    def __hash__(self) -> int:
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )

        return hash(
            (
                self.name,
                self.options,
                self.domain,
                compute_frameworks_hashable,
                self.data_type,
                self._child_options_key(),
            )
        )

    def _child_options_key(self) -> Any:
        """Cycle-safe identity view of child_options for __eq__/__hash__ (#608).

        child_options can hold features value-equal to self, so a deep compare recurses
        forever. _reduce mirrors the Feature.__eq__/Options.__eq__ fields, with an id()
        visited guard that collapses an already-seen feature to its name to terminate.
        """
        if self.child_options is None:
            return None
        return self._reduce(self.child_options.group, frozenset())

    @staticmethod
    def _reduce(value: Any, seen: frozenset[int]) -> Any:
        if isinstance(value, Feature):
            if id(value) in seen:
                return ("feature", value.name)
            seen = seen | {id(value)}
            compute_frameworks = frozenset(value.compute_frameworks) if value.compute_frameworks is not None else None
            child_group = Feature._reduce(value.child_options.group, seen) if value.child_options is not None else None
            return (
                "feature",
                value.name,
                Feature._reduce(value.options.group, seen),
                Feature._reduce(value.options.context, seen),
                value.domain,
                compute_frameworks,
                value.data_type,
                child_group,
            )
        if isinstance(value, Options):
            return ("options", Feature._reduce(value.group, seen))
        if isinstance(value, dict):
            return tuple(
                sorted(((key, Feature._reduce(val, seen)) for key, val in value.items()), key=lambda kv: kv[0])
            )
        if isinstance(value, (frozenset, set)):
            return frozenset(Feature._reduce(item, seen) for item in value)
        if isinstance(value, (list, tuple)):
            return tuple(Feature._reduce(item, seen) for item in value)
        return _make_hashable(value)

    def is_different_data_type(self, other: Feature) -> bool:
        return self.name == other.name and self.data_type != other.data_type

    def _split_context_hashable(self, split_keys: frozenset[str]) -> Any:
        """Order-independent view of this feature's context values for the split keys."""
        if not split_keys:
            return ()
        context = self.options.context
        relevant = {key: context[key] for key in split_keys if key in context}
        if not relevant:
            return ()
        return _make_hashable(relevant)

    def _grouping_hash(self, split_keys: frozenset[str] | None, include_data_type: bool) -> int:
        keys = self.options.inherited_context_keys if split_keys is None else split_keys
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        split_context = self._split_context_hashable(keys)
        if include_data_type and self.data_type is not None:
            return hash((self.options, compute_frameworks_hashable, split_context, self.data_type))
        return hash((self.options, compute_frameworks_hashable, split_context))

    def similarity_hash(self, split_keys: frozenset[str] | None = None) -> int:
        """Grouping hash over options, compute framework, split-key context values, and data type.

        When split_keys is None it falls back to THIS feature's own inherited_context_keys, a
        per-feature convenience. Production grouping in
        ExecutionPlan.group_features_by_compute_framework_and_options passes the resolution-wide
        union of every in-scope feature's inherited_context_keys; any caller performing grouping
        must pass that resolution-wide split_keys rather than relying on the per-feature default.
        data_type is excluded when None so None-typed features can join typed groups.
        """
        return self._grouping_hash(split_keys, include_data_type=True)

    def base_similarity_hash(self, split_keys: frozenset[str] | None = None) -> int:
        """similarity_hash without data_type, for lenient grouping of None-typed features.

        split_keys follows the same rule as similarity_hash: None falls back to THIS feature's own
        inherited_context_keys (a per-feature convenience), whereas production grouping in
        ExecutionPlan.group_features_by_compute_framework_and_options passes the resolution-wide
        union of every in-scope feature's inherited_context_keys. Any caller performing grouping
        must pass that resolution-wide split_keys rather than relying on the per-feature default.
        """
        return self._grouping_hash(split_keys, include_data_type=False)

    def _set_domain(self, domain: Optional[str], domain_options: Optional[str]) -> None | Domain:
        if domain:
            return Domain(domain)
        elif domain_options:
            return Domain(domain_options)
        return None

    def _set_compute_framework(
        self, compute_framework: Optional[str], compute_framework_options: Optional[str]
    ) -> Optional[type[ComputeFramework]]:
        if compute_framework:
            return FeatureValidator.validate_and_resolve_compute_framework(
                compute_framework, get_all_subclasses(ComputeFramework), "parameter"
            )
        elif compute_framework_options:
            return FeatureValidator.validate_and_resolve_compute_framework(
                compute_framework_options, get_all_subclasses(ComputeFramework), "options"
            )
        return None

    def get_compute_framework(self) -> type[ComputeFramework]:
        FeatureValidator.validate_compute_frameworks_resolved(self.compute_frameworks, str(self.name))
        assert self.compute_frameworks is not None
        return next(iter(self.compute_frameworks))
