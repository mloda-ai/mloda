from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Set, Type, Union
from uuid import UUID, uuid4
from mloda.core.abstract_plugins.components.data_types import DataType

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.link import Link
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.utils import get_all_subclasses
from mloda.core.abstract_plugins.components.validators.feature_validator import FeatureValidator
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


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
        name: Union[str, FeatureName],
        options: Union[Dict[str, Any], Options] = {},
        domain: Optional[str] = None,
        compute_framework: Optional[str] = None,
        data_type: Optional[Union[DataType, str]] = None,
        initial_requested_data: bool = False,
        link: Optional[Link] = None,
        index: Optional[Index] = None,
    ):
        self.name = FeatureName(name) if isinstance(name, str) else name
        self.options = Options(options) if isinstance(options, dict) else options
        self.domain = self._set_domain(domain, self.options.get("domain"))

        cf = self._set_compute_framework(compute_framework, self.options.get("compute_framework"))
        self.compute_frameworks = {cf} if cf else None

        self.uuid = uuid4()

        self.data_type = None
        if data_type:
            self.data_type = data_type if isinstance(data_type, DataType) else None

        self.child_options: Optional[Options] = None

        self.initial_requested_data = initial_requested_data

        # LINK and INDEX are excluded from equality and hash, because this way, we can define a single feature of a group with these properties.
        self.link = link
        self.index = index  # Index is a feature currently only used for append/union features.

    @classmethod
    def not_typed(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> Feature:
        name = FeatureName(name) if isinstance(name, str) else name
        return cls(name=name, options=options)

    @classmethod
    def str_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> Feature:
        return cls._typed_of(name, DataType.STRING, options)

    @classmethod
    def int32_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> Feature:
        return cls._typed_of(name, DataType.INT32, options)

    @classmethod
    def int64_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.INT64, options)

    @classmethod
    def float_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.FLOAT, options)

    @classmethod
    def double_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.DOUBLE, options)

    @classmethod
    def boolean_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.BOOLEAN, options)

    @classmethod
    def binary_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.BINARY, options)

    @classmethod
    def date_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.DATE, options)

    @classmethod
    def timestamp_millis_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.TIMESTAMP_MILLIS, options)

    @classmethod
    def timestamp_micros_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.TIMESTAMP_MICROS, options)

    @classmethod
    def decimal_of(cls, name: Union[str, FeatureName], options: dict[str, Any] = {}) -> "Feature":
        return cls._typed_of(name, DataType.DECIMAL, options)

    @classmethod
    def _typed_of(cls, name: Union[str, FeatureName], data_type: DataType, options: dict[str, Any] = {}) -> Feature:
        name = FeatureName(name) if isinstance(name, str) else name
        return cls(name=name, data_type=data_type, options=options)

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
            and self.child_options == other.child_options
        )

    def __hash__(self) -> int:
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )

        child_options = copy.deepcopy(self.child_options)
        if child_options:
            if child_options.get(DefaultOptionKeys.in_features):
                val = child_options.get(DefaultOptionKeys.in_features)

                if isinstance(val, frozenset):
                    for v in val:
                        if isinstance(v, Feature):
                            child_options.group[DefaultOptionKeys.in_features] = v.name.name

                if isinstance(val, Feature):
                    child_options.group[DefaultOptionKeys.in_features] = val.name.name

        return hash((self.name, self.options, self.domain, compute_frameworks_hashable, self.data_type, child_options))

    def is_different_data_type(self, other: Feature) -> bool:
        return self.name == other.name and self.data_type != other.data_type

    def has_similarity_properties(self) -> int:
        """Hash for grouping features by compute framework, options, and data type.

        When data_type is None, it's excluded from the hash so None-typed features
        can be grouped with any typed features (handled by grouping logic).
        """
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        if self.data_type is not None:
            return hash((self.options, compute_frameworks_hashable, self.data_type))
        return hash((self.options, compute_frameworks_hashable))

    def base_similarity_properties(self) -> int:
        """Base hash excluding data_type - used for lenient grouping of None-typed features."""
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        return hash((self.options, compute_frameworks_hashable))

    def _set_domain(self, domain: Optional[str], domain_options: Optional[str]) -> Union[None, Domain]:
        if domain:
            return Domain(domain)
        elif domain_options:
            return Domain(domain_options)
        return None

    def _set_compute_framework(
        self, compute_framework: Optional[str], compute_framework_options: Optional[str]
    ) -> Optional[Type[ComputeFramework]]:
        if compute_framework:
            return FeatureValidator.validate_and_resolve_compute_framework(
                compute_framework, get_all_subclasses(ComputeFramework), "parameter"
            )
        elif compute_framework_options:
            return FeatureValidator.validate_and_resolve_compute_framework(
                compute_framework_options, get_all_subclasses(ComputeFramework), "options"
            )
        return None

    def _set_uuid(self, uuid: UUID) -> Feature:
        # use only for testing
        self.uuid = uuid
        return self

    def _set_compute_frameworks(self, compute_frameworks: Set[Type[ComputeFramework]]) -> Feature:
        # use only for testing
        self.compute_frameworks = compute_frameworks
        return self

    def get_compute_framework(self) -> Type[ComputeFramework]:
        FeatureValidator.validate_compute_frameworks_resolved(self.compute_frameworks, str(self.name))
        assert self.compute_frameworks is not None
        return next(iter(self.compute_frameworks))

    def get_name(self) -> str:
        return self.name.name
