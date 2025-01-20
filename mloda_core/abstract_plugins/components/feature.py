from __future__ import annotations

from typing import Any, Dict, Optional, Set, Type, Union
from uuid import UUID, uuid4
from mloda_core.abstract_plugins.components.data_types import DataType

from mloda_core.abstract_plugins.components.domain import Domain
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import Link
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.utils import get_all_subclasses


class Feature:
    """A class to represent a raw feature.


    We intentionally only allow one compute framework per feature if this variable is set by init.
    If this variable is not set, we allow multiple compute frameworks per feature during resolving the feature network.


    Convencience class methods:
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
        self.domain = self.set_domain(domain, self.options.data.get("domain"))

        cf = self.set_compute_framework(compute_framework, self.options.data.get("compute_framework"))
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
            and self.domain == other.domain
            and self.compute_frameworks == other.compute_frameworks
            and self.data_type == other.data_type
            and self.child_options == other.child_options
        )

    def __hash__(self) -> int:
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        return hash(
            (self.name, self.options, self.domain, compute_frameworks_hashable, self.data_type, self.child_options)
        )

    def is_different_data_type(self, other: Feature) -> bool:
        return self.name == other.options and self.data_type == other.data_type

    def has_similarity_properties(self) -> int:
        compute_frameworks_hashable = (
            frozenset(self.compute_frameworks) if self.compute_frameworks is not None else None
        )
        return hash((self.options, compute_frameworks_hashable))

    def set_domain(self, domain: Optional[str], domain_options: Optional[str]) -> Union[None, Domain]:
        if domain:
            return Domain(domain)
        elif domain_options:
            return Domain(domain_options)
        return None

    def set_compute_framework(
        self, compute_framework: Optional[str], compute_framework_options: Optional[str]
    ) -> Optional[Type[ComputeFrameWork]]:
        if compute_framework or compute_framework_options:
            subclasses_compute_frameworks = get_all_subclasses(ComputeFrameWork)

            if compute_framework:
                for subclass in subclasses_compute_frameworks:
                    if compute_framework == subclass.get_class_name():
                        return subclass
                raise ValueError(f"Compute framework via parameter {compute_framework} not found.")

            elif compute_framework_options:
                for subclass in subclasses_compute_frameworks:
                    if compute_framework_options == subclass.get_class_name():
                        return subclass
                raise ValueError(f"Compute framework via options {compute_framework_options} not found.")
        return None

    def _set_uuid(self, uuid: UUID) -> Feature:
        # use only for testing
        self.uuid = uuid
        return self

    def _set_compute_frameworks(self, compute_frameworks: Set[Type[ComputeFrameWork]]) -> Feature:
        # use only for testing
        self.compute_frameworks = compute_frameworks
        return self

    def get_compute_framework(self) -> Type[ComputeFrameWork]:
        if self.compute_frameworks is None:
            raise ValueError(
                f"Feature {self.name} does not have any compute framework. This function can only be called when the frameworks were resolved."
            )

        return next(iter(self.compute_frameworks))

    def get_name(self) -> str:
        return self.name.name
