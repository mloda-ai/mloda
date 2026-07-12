from __future__ import annotations

import inspect
import logging
from typing import Any, ClassVar, Callable, Iterable, Optional, final
from abc import ABC

from mloda.core.abstract_plugins.components.base_artifact import BaseArtifact
from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.data_types import DataType

from mloda.core.abstract_plugins.components.domain import Domain
from mloda.core.abstract_plugins.components.base_feature_group_version import BaseFeatureGroupVersion
from mloda.core.abstract_plugins.components.feature_chainer.feature_chain_parser import (
    CHAIN_SEPARATOR,
    FeatureChainParser,
)
from mloda.core.abstract_plugins.components.subtype_declaration import SubtypeDeclaration
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.input_data.api.api_input_data import ApiInputData
from mloda.core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda.core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda.core.abstract_plugins.components.match_data.match_data import MatchData
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.components.index.index import Index
from mloda.core.abstract_plugins.components.utils import get_all_subclasses

logger = logging.getLogger(__name__)


class FeatureGroup(ABC):
    """Base class for all feature groups.

    Feature groups that derive new features from existing ones (chained
    features) should also inherit from ``FeatureChainParserMixin``. The mixin
    handles feature name parsing via ``PREFIX_PATTERN``/``SUFFIX_PATTERN``,
    ``PROPERTY_MAPPING`` validation, and provides a default ``input_features``
    implementation::

        from mloda.provider import FeatureChainParserMixin, FeatureGroup, DefaultOptionKeys

        class MyFeatureGroup(FeatureChainParserMixin, FeatureGroup):
            PREFIX_PATTERN = r".*__([\\w]+)_my_op$"
            PROPERTY_MAPPING = {
                "operation_type": {
                    "add": "Addition",
                    "sub": "Subtraction",
                    DefaultOptionKeys.context: True,
                    DefaultOptionKeys.strict_validation: True,
                },
            }

            @classmethod
            def calculate_feature(cls, data, features):
                ...

    Subclass ``FeatureGroup`` directly (without the mixin) for primary-source
    feature groups that load raw data and have no input features. In that case,
    implement ``input_features`` (return ``None`` to mark as root) and
    ``match_feature_group_criteria``.

    Optional overrides (with defaults):
        - ``match_feature_group_criteria``: default matches the class name
        - ``domain``: default is the default domain
        - ``compute_framework_rule``: default allows all compute frameworks
        - ``index_columns``: default is None
        - ``return_data_type_rule``: default is None

    See ``FeatureChainParserMixin`` and ``docs/in_depth/property-mapping.md``
    for the full PROPERTY_MAPPING reference.
    """

    PROPERTY_MAPPING: ClassVar[Optional[dict[str, Any]]] = None
    """Override in subclasses to declare configurable parameters.

    Each key is a parameter name. Each value is a dict containing valid
    values, metadata flags (``DefaultOptionKeys.context``,
    ``DefaultOptionKeys.strict_validation``, etc.), and optional validators.
    See ``docs/in_depth/property-mapping.md`` for the full specification.
    """

    SUBTYPES: ClassVar[Optional[SubtypeDeclaration]] = None
    """Declarative subtype dimension of the family; ``None`` means no subtype dimension.
    The derived accessors below are the only surface and must not be overridden."""

    _DERIVED_SUBTYPE_ACCESSORS: ClassVar[tuple[str, ...]] = (
        "subtype_universe",
        "supported_subtypes",
        "resolve_subtype",
        "canonical_subtype",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        FeatureChainParser.validate_property_mapping_defaults(cls.__name__, cls.PROPERTY_MAPPING)
        cls._reject_derived_accessor_overrides()
        cls._validate_subtype_declaration()

    @classmethod
    def _overrides(cls, method_name: str) -> bool:
        """True when cls replaces the FeatureGroup base classmethod."""
        own = getattr(getattr(cls, method_name), "__func__", None)
        base = getattr(getattr(FeatureGroup, method_name), "__func__", None)
        return own is not base

    @classmethod
    def _reject_derived_accessor_overrides(cls) -> None:
        for method_name in cls._DERIVED_SUBTYPE_ACCESSORS:
            if cls._overrides(method_name):
                raise ValueError(
                    f"{cls.__name__} overrides {method_name}(); the subtype accessors are derived from "
                    f"SUBTYPES and must not be overridden."
                )

    @classmethod
    def _validate_subtype_declaration(cls) -> None:
        """Enforce the shape-A class-dependent checks of SUBTYPES at class-definition time."""
        declaration = cls.SUBTYPES
        if declaration is None or declaration.key is None:
            return

        if declaration.key not in cls.declared_option_keys():
            raise ValueError(
                f"{cls.__name__} declares subtype key '{declaration.key}', "
                f"which is not a declared PROPERTY_MAPPING key."
            )

        declared_values = cls.declared_option_values(declaration.key)
        if not declared_values:
            raise ValueError(
                f"{cls.__name__} declares subtype key '{declaration.key}' without an enumerable value space; "
                f"predicate-validated keys must declare a flattened universe with a resolver instead."
            )

        colliding = declaration.family_names() & declared_values
        if colliding:
            raise ValueError(
                f"{cls.__name__} declares parametric families {sorted(colliding)} that collide "
                f"with declared values of subtype key '{declaration.key}'."
            )

        if declaration.supported is not None:
            universe = declared_values | declaration.family_names()
            for framework_name, values in declaration.supported.items():
                overreach = frozenset(values) - universe
                if overreach:
                    raise ValueError(
                        f"{cls.__name__} declares supported subtypes {sorted(overreach)} on "
                        f"{framework_name} that are outside its subtype universe."
                    )

    @classmethod
    def declared_option_keys(cls) -> frozenset[str]:
        """Return the top-level parameter names declared in ``PROPERTY_MAPPING``."""
        if cls.PROPERTY_MAPPING is None:
            return frozenset()
        return frozenset(str(key) for key in cls.PROPERTY_MAPPING)

    @final
    @classmethod
    def declared_option_values(cls, key: str) -> frozenset[str]:
        """Return the stringified enumerable value space of a ``PROPERTY_MAPPING`` key;
        predicate-only and absent keys yield an empty set."""
        if cls.PROPERTY_MAPPING is None or key not in cls.PROPERTY_MAPPING:
            return frozenset()
        extracted = FeatureChainParser._extract_property_values(cls.PROPERTY_MAPPING[key])
        if not isinstance(extracted, (dict, list, tuple, set, frozenset)):
            return frozenset()
        return frozenset(str(value) for value in extracted)

    @final
    @classmethod
    def subtype_universe(cls) -> frozenset[str]:
        """Every subtype this family declares: declared values or literals plus family names."""
        declaration = cls.SUBTYPES
        if declaration is None:
            return frozenset()
        if declaration.key is not None:
            return cls.declared_option_values(declaration.key) | declaration.family_names()
        return frozenset(declaration.universe or ()) | declaration.family_names()

    @final
    @classmethod
    def supported_subtypes(cls, compute_framework: type[ComputeFramework]) -> frozenset[str]:
        """Subtypes supported on a compute framework; frameworks absent from the declaration
        support the full universe."""
        declaration = cls.SUBTYPES
        if declaration is None:
            return frozenset()
        if declaration.supported is not None:
            framework_name = compute_framework.get_class_name()
            if framework_name in declaration.supported:
                return frozenset(declaration.supported[framework_name])
        return cls.subtype_universe()

    @final
    @classmethod
    def resolve_subtype(cls, feature_name: FeatureName | str, options: Options) -> Optional[str]:
        """Resolve the raw subtype of a concrete feature: name parsing first, then options; never raises."""
        declaration = cls.SUBTYPES
        if declaration is None:
            return None

        name = str(feature_name)
        if declaration.resolver is not None:
            return declaration.resolver(name, options)
        if declaration.key is None:
            return None

        source, separator, _ = name.rpartition(CHAIN_SEPARATOR)
        if separator and source:
            patterns = [
                pattern
                for pattern in (getattr(cls, "PREFIX_PATTERN", None), getattr(cls, "SUFFIX_PATTERN", None))
                if isinstance(pattern, str)
            ]
            if patterns:
                parsed, _parsed_source = FeatureChainParser.parse_feature_name(name, patterns)
                if parsed is not None:
                    return parsed

        value = options.get(declaration.key)
        if value is None:
            return None
        return str(value)

    @final
    @classmethod
    def canonical_subtype(cls, subtype: str) -> str:
        """Collapse a parametric instance ``<family>_<digits>`` to its family name, else identity;
        a declared universe member never collapses."""
        declaration = cls.SUBTYPES
        if declaration is None:
            return subtype
        if subtype in cls.subtype_universe():
            return subtype
        stem, separator, tail = subtype.rpartition("_")
        if separator and tail.isdigit() and stem in declaration.family_names():
            return stem
        return subtype

    @final
    @classmethod
    def subtype_support_matrix(cls) -> dict[str, frozenset[str]]:
        """Supported subtypes per declared compute framework; empty for abstract classes and
        families without a subtype dimension. Raises ValueError when the class hand-overrides
        ``supports_compute_framework``, which makes the declared matrix unverifiable."""
        if inspect.isabstract(cls):
            return {}

        universe = cls.subtype_universe()
        if not universe:
            return {}

        if cls._overrides("supports_compute_framework"):
            raise ValueError(
                f"{cls.get_class_name()} overrides supports_compute_framework, so the hand-written hook makes "
                f"the declared subtype support matrix unverifiable; the declaration is not authoritative."
            )

        return {
            compute_framework.get_class_name(): cls.supported_subtypes(compute_framework)
            for compute_framework in cls.compute_framework_definition()
        }

    def __init__(self) -> None:
        pass

    @classmethod
    def description(cls) -> str:
        """
        Returns a description for this feature group.

        The method returns the class's own docstring if it has been overridden from
        the base class's docstring. Otherwise, it falls back to the class name.
        This behavior allows subclasses to easily customize their description.
        """
        base_doc = (FeatureGroup.__doc__ or "").strip()
        current_doc = (cls.__doc__ or "").strip()

        if current_doc and current_doc != base_doc:
            return current_doc
        return cls.get_class_name()

    @classmethod
    def version(cls) -> str:
        """
        Returns a composite version identifier for this feature group.

        The version identifier is generated by combining:
          - the version of the 'mloda' package (retrieved from package metadata),
          - the module name where the feature group is defined, and
          - a SHA-256 hash of the feature group class's source code.

        This composite identifier uniquely represents the implementation state of the feature group,
        making it easier to detect changes, manage compatibility, and debug issues.

        If you need to change the version of the feature group, you can do so by subclassing
        BaseFeatureGroupVersion and overriding the version method. This allows you to create a new version system.
        """
        return BaseFeatureGroupVersion.version(cls)

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        """
        This function should return the input data class used for this feature group.
        """
        return None

    @classmethod
    def validate_input_features(cls, data: Any, features: FeatureSet) -> None:
        """
        Validate the input data before feature calculation.

        Override this method to implement custom input validation logic.
        Raise a ValueError with a descriptive message if validation fails.
        If the method returns without raising, validation is considered passed.
        """
        return None

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        """
        Validate the output data after feature calculation.

        Override this method to implement custom output validation logic.
        Raise a ValueError with a descriptive message if validation fails.
        If the method returns without raising, validation is considered passed.
        """
        return None

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        """
        This function should be used to calculate the feature.
        """
        return None

    @staticmethod
    def artifact() -> type[BaseArtifact] | None:
        """
        Returns the artifact associated with this feature group.

        Artifacts are data generated by a feature group and can be used by other feature groups.
        This is necessary for scenarios such as embeddings, where the output of one feature group
        serves as an input for another, enabling complex data transformations and feature engineering
        workflows.

        This method should be overridden by subclasses to provide the specific artifact
        that the feature group generates or uses. If no artifact is associated with the
        feature group, this method should return None.
        """
        return None

    @final
    @classmethod
    def load_artifact(cls, features: FeatureSet) -> Any:
        """
        Convenience function to load an artifact associated with the given FeatureSet.

        This method utilizes the `artifact` method to retrieve the specific artifact class
        associated with the feature group. It then calls the `load` method of the artifact class
        to load the artifact data.
        """
        artifact = cls.artifact()
        if artifact is None:
            raise ValueError(f"Artifact load is called, but not implemented: {cls.get_class_name()}.")
        return artifact.load(features)

    def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:
        """
        Allows modification of the feature name based on configuration.

        This method provides a hook for subclasses to modify the feature name based on the
        provided options and the initial feature name. The default implementation simply
        returns the original feature name, but subclasses can override this method to implement
        custom logic for feature name modification.

        For sub-column requests (e.g., "base_feature~1"), the feature name is normalized
        to the base feature name if it matches a supported feature via the base name.
        """
        base_name = self.get_column_base_feature(str(feature_name))
        if base_name != str(feature_name) and base_name in self.feature_names_supported():
            return FeatureName(base_name)
        return feature_name

    @staticmethod
    def apply_naming_convention(
        result: Any, feature_name: str, suffix_generator: Optional[Callable[[int], str]] = None
    ) -> dict[str, Any]:
        """
        Applies naming convention to multi-column results.

        For 2D arrays with multiple columns, creates a dictionary mapping
        column names to column data using the pattern: feature_name~0, feature_name~1, etc.
        """
        if hasattr(result, "shape") and len(result.shape) > 1 and result.shape[1] > 1:
            output = {}
            for col_idx in range(result.shape[1]):
                suffix = suffix_generator(col_idx) if suffix_generator else str(col_idx)
                col_name = f"{feature_name}~{suffix}"
                output[col_name] = result[:, col_idx]
            return output
        return {}

    @staticmethod
    def get_column_base_feature(column_name: str) -> str:
        """
        Extracts the base feature name from a column name by stripping the ~N suffix.

        Returns the original name if no ~N suffix exists.
        """
        return column_name.split("~")[0]

    @staticmethod
    def expand_feature_columns(feature_name: str, num_columns: int) -> list[str]:
        """
        Generates a list of column names with ~N suffixes.

        Returns a list of column names following the pattern: ["feature~0", "feature~1", ...]
        """
        return [f"{feature_name}~{i}" for i in range(num_columns)]

    @staticmethod
    def resolve_multi_column_feature(feature_name: str, available_columns: set[str]) -> list[str]:
        """
        Resolves a feature name to its corresponding column(s) in the available columns.

        If the exact feature_name exists in available_columns, returns [feature_name].
        Otherwise, finds all columns matching the pattern {feature_name}~* and returns them sorted.
        If no matches found, returns [feature_name] (caller will handle the error).
        """
        if feature_name in available_columns:
            return [feature_name]

        matching_columns = [col for col in available_columns if col.startswith(f"{feature_name}~")]

        if matching_columns:
            return sorted(matching_columns)

        return [feature_name]

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> Optional[DataType]:
        """
        Specifies a fixed return data type for this feature group, if applicable.

        If this feature group always returns a specific data type, this method should
        return that data type. Otherwise, it should return None, indicating that the
        data type is not fixed and may vary depending on the input or computation.
        """
        return None

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        """
        Defines the input features required by this feature group.

        If this feature depends on other features as input, this method should return a set
        containing those features. If it does not depend on any other features (i.e., it is
        a root feature), it should return None.

        The specific input features may depend on the provided options and the feature name.
        """
        raise NotImplementedError

    @classmethod
    def index_columns(cls) -> Optional[list[Index]]:
        """
        Specifies the index columns used for merging or joining data.

        This method should return a list of Index objects representing the columns to be
        used as indices for merging or joining dataframes. The indices can be defined by
        name, by user-set index, or by the features themselves.

        This method also provides an opportunity to validate given indices against the
        aforementioned sources, if implemented.
        """
        return None

    @classmethod
    def supports_index(cls, index: Index) -> Optional[bool]:
        """
        Check if this feature group supports the given index.

        This method checks the index against the feature group's supported index columns.
        If no index columns are defined, any index is accepted.

        Args:
            index: The index to check for support.

        Returns:
            None: No index constraint defined (accepts any index)
            True: Index is supported
            False: Index is not supported
        """
        supported_index_columns = cls.index_columns()

        if supported_index_columns is None:
            return None

        for supported_index_column in supported_index_columns:
            if index.is_a_part_of(supported_index_column):
                return True

        return False

    @classmethod
    def _matches_input_data(
        cls, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection]
    ) -> bool:
        """
        Helper function to check if the input data matches.
        """
        input_data_class = cls.input_data()

        if input_data_class is None:
            return False

        if isinstance(input_data_class, DataCreator):
            return input_data_class.matches(feature_name, options, None)

        if isinstance(input_data_class, ApiInputData):
            return input_data_class.matches(feature_name, options, None)

        return input_data_class.matches(feature_name, options, data_access_collection)

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        """
        Determines whether this feature group matches the given criteria.

        This method returns True if the feature_name matches the criteria defined by this
        feature group, and False otherwise. The criteria may include the feature name,
        options, and data access collection.

        The if statement contains the rules. Each case has different use cases.
        You can disallow them by removing them. However, often you can just use the default.
        If you want to implement a concrete implementation, e.g. just accept specific names,
        then you can overwrite this function.
        """

        base_feature_name = cls.get_column_base_feature(feature_name)

        if cls._is_root_and_matches_input_data(base_feature_name, options, data_access_collection):
            return True

        if cls._matches_data(base_feature_name, options, data_access_collection):
            return True

        if cls.feature_name_equal_to_class_name(base_feature_name):
            return True

        if cls.feature_name_contains_class_name_as_prefix(base_feature_name):
            return True

        if base_feature_name in cls.feature_names_supported():
            return True

        return False

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        """
        Returns a set of feature names that are explicitly supported by this feature group.

        This method provides a way to specify a set of feature names that this feature group
        is designed to handle. It can be used to add custom feature names to the feature
        group in a simple manner.

        This function is a convenience functionality. It is not necessary to implement this function.
        """
        return set()

    @classmethod
    def feature_name_equal_to_class_name(cls, feature_name: str) -> bool:
        """
        Checks if the given feature name is equal to the class name of this feature group.

        This functionality is useful in cases where the feature name directly corresponds
        to the class name, such as with scores or very specific implementations for
        embeddings.
        """
        return feature_name == cls.get_class_name()

    @classmethod
    def feature_name_contains_class_name_as_prefix(cls, feature_name: str) -> bool:
        """
        Checks if the given feature name starts with the class name of this feature group
        as a prefix.
        """
        return feature_name.startswith(cls.prefix())

    @classmethod
    def get_domain(cls) -> Domain:
        """
        Returns the domain for this feature group.
        """
        return Domain.get_default_domain()

    @classmethod
    def final_filters(cls) -> bool | None:
        """Controls whether the framework applies post-calculation row elimination.

        This method is independent of inline filter reading. ``features.filters``
        is always available inside ``calculate_feature()``, regardless of what
        this method returns. A FeatureGroup may read filters inline (e.g. for
        conditional masking or predicate pushdown) and still request row
        elimination by returning ``True``.

        Returns:
            None:  Defer to the FilterEngine (default).
            False: Skip row elimination (the FeatureGroup handles filters itself).
            True:  Force row elimination after calculation.
        """
        return None

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        """
        Defines which compute frameworks this feature group supports.

        Return ``None`` (the default) to support all available frameworks.
        Return a set of specific ``ComputeFramework`` subclasses to restrict support.
        """
        return None

    @final
    @classmethod
    def compute_framework_definition(cls) -> set[type[ComputeFramework]]:
        """
        Determines the set of compute frameworks supported by this feature group based on the
        `compute_framework_rule`.
        """
        rule = cls.compute_framework_rule()

        if rule is None:
            return get_all_subclasses(ComputeFramework)

        if not isinstance(rule, set):
            raise ValueError(
                f"compute_framework_rule() of {cls.__name__} must return None or a set of "
                f"ComputeFramework subclasses, got {type(rule).__name__}."
            )
        return rule

    @classmethod
    def supports_compute_framework(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        compute_framework: type[ComputeFramework],
    ) -> bool:
        """Per-feature, per-framework capability check evaluated at match time.

        Returns ``True`` (the default) to allow the framework. Override to declare
        that a specific operation (encoded in ``feature_name``/``options``) is
        unsupported on ``compute_framework`` -- return ``False`` and the matcher
        removes that framework from the candidate set for this feature only.
        Unlike ``compute_framework_rule`` (a static class-level set), this hook sees
        the concrete feature, so it can reject an op on one backend while allowing
        others. If every candidate framework is rejected, the matcher surfaces a
        distinguishable error instead of a generic "no feature group" message.

        The default gates a declared canonical subtype by ``supported_subtypes()``;
        everything else (no subtype dimension, unresolved or undeclared subtype) stays open.
        """
        universe = cls.subtype_universe()
        if not universe:
            return True

        subtype = cls.resolve_subtype(feature_name, options)
        if subtype is None:
            return True

        canonical = cls.canonical_subtype(subtype)
        if canonical not in universe:
            return True

        return canonical in cls.supported_subtypes(compute_framework)

    @final
    @classmethod
    def get_class_name(cls) -> str:
        """
        Returns the name of the class.
        """
        return cls.__name__

    def __eq__(self, another: Any) -> bool:
        """
        Checks if this feature group is equal to another object.
        """
        if isinstance(another, FeatureGroup):
            return type(self) is type(another)
        return NotImplemented

    def __hash__(self) -> int:
        """
        Returns the hash code for this feature group.
        """
        return id(type(self))

    @final
    def is_root(self, options: Options, feature_name: str | FeatureName) -> bool:
        """
        Determines whether this feature is a root feature (i.e., does not depend on any
        other features).
        """
        try:
            if not isinstance(feature_name, FeatureName):
                feature_name = FeatureName(feature_name)

            if self.input_features(options, feature_name) is None:
                # No input features declared, so this is a root feature.
                return True
        except NotImplementedError:
            # input_features not implemented means this is a root feature.
            return True
        except Exception:
            # Errors in input_features (e.g. validation failures for this feature name)
            # mean the feature group does not match, so it is not a root.
            logger.debug(
                "%s.input_features raised an exception for feature '%s'",
                type(self).__name__,
                feature_name,
                exc_info=True,
            )
        return False

    @classmethod
    def prefix(cls) -> str:
        """
        Returns the prefix used for feature names associated with this feature group.

        This is a convention, which means we can refer to a class via this.
        """
        return f"{cls.get_class_name()}_"

    @classmethod
    def _is_root_and_matches_input_data(
        cls, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection]
    ) -> bool:
        """
        Checks if the feature group is a root and matches input data.
        """
        return cls().is_root(options, feature_name) and cls._matches_input_data(
            feature_name, options, data_access_collection
        )

    @final
    @classmethod
    def _matches_data(
        cls, feature_name: str, options: Options, data_access_collection: Optional[DataAccessCollection]
    ) -> bool:
        """
        This functionality is for matching data, when a data access is necessary.
        This is relevant for compute frameworks which need a connection object.

        To be used, create a class like this:

        class MyMatchData(FeatureGroup, MatchData):
            ...

        and then create the function match_data_access.
        """

        if not issubclass(cls, MatchData):
            return False

        return cls.matches(feature_name, options, data_access_collection)


def format_feature_group_class(fg_class: type[FeatureGroup]) -> str:
    """Format a single FeatureGroup class for error messages."""
    return f"{fg_class.__name__} ({fg_class.__module__})"


def format_feature_group_classes(feature_groups: Iterable[type[FeatureGroup]], include_domain: bool = False) -> str:
    """Format FeatureGroup classes for error messages."""
    lines = []
    for fg_class in feature_groups:
        line = f"  - {fg_class.__name__} ({fg_class.__module__})"
        if include_domain:
            domain = fg_class.get_domain()
            line += f" [domain: {domain.name}]"
        lines.append(line)
    return "\n".join(lines)
