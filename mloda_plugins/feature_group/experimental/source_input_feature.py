# Top-Level Documentation for LLMs

"""
This module is designed to be used as a mixin or inherited class for defining input features.

It allows defining input features that originate from:
    - other feature
    - ApiInputData
    - DataCreator
    - Local Feature scope
    - Global Feature scope

Further, it allows defining:
    - Join operations between features from different/same origins.
    - Mloda requires an Index for Append and Merges, but not for Joins.

**Key Classes:**
    - `SourceInputFeature`: An abstract class used as a base or mixin for defining input features.
    - `SourceInputFeatureComposite`: A composite class providing the core logic for handling source definitions.
    - `SourceTuple`:  A named tuple defining the structure for a complex source feature, including properties, joins, and merges.

**Usage:**
   - The `SourceInputFeature` class should be either inherited or used as a mixin.
   - Input features are defined with a `frozenset` in the `mloda_source_feature` option in the feature options.
   - The elements of the `frozenset` can be strings(simple feature name) or `SourceTuple`(complex feature definition).

   **Example of defining input features:**
        ```python
        Feature(name="target_feature",
                options={
                    DefaultOptionKeys.mloda_source_feature: frozenset(["source_feature_1",
                     SourceTuple(feature_name="source_feature_2",
                                  source_class=MyFeatureGroup,
                                  source_value="value"
                                  )
                    ])
                })
        ```
"""

from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, Type, Union
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.link import JoinType, Link
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class SourceInputFeature(AbstractFeatureGroup):
    """
    This feature group focuses on defining input features, especially when they originate
    from other sources or require joins/merges.

    You can use this class in two ways:
        1. **Inheritance:** Inherit from `SourceInputFeature` and define your input features within its scope.
        2. **Mixin:**  Use `SourceInputFeatureComposite` as a mixin to add input feature handling to another class.

    **Key Requirement:**
        - Your feature options must include `DefaultOptionKeys.mloda_source_feature`, which
          specifies the source feature(s).

    **Source Definition:**

    You define your input sources using a `frozenset`. Each element of the frozenset can be:
        - A `str`: Represents a simple source feature name.
        - A `tuple`: Represents a complex source feature with properties, joins, and merges
            using the `SourceTuple` structure.

    **How to define a target feature with source feature(s):**
    ```python
    Feature(name="target_feature",
            options={
                DefaultOptionKeys.mloda_source_feature: frozenset(["source_feature_1", "source_feature_2"])
            })
    ```

    **Available options:**
        - input_feature: The definition of the input feature.
        - api: Used for defining api connections.
        - creator: Defines the function to create the input feature.
        - local feature scope: Defines parameters for local feature scope, that is not the entire pipeline.
        - global feature scope: Defines the global feature scope parameters.

    **Additionally, you can define the following options within `SourceTuple`:**
        - joins:  Specifies join operations between features.
        - merges:  Specifies merge operations between features.
    """

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        return SourceInputFeatureComposite.input_features(options, feature_name)


class SourceTuple(NamedTuple):
    """
    Defines the structure for a complex source feature.

    A tuple that describes a source feature with properties, joins, and merges.

    Attributes:
        feature_name: The name of the feature.
        source_class: (Optional) The source class of the feature, can be an `AbstractFeatureGroup` class or a `str` representing a scope.
        source_value: (Optional) The value associated with the source class, if applicable.
        left_link: (Optional)  A tuple containing the left-side `AbstractFeatureGroup` class and index for join operations.
        right_link: (Optional) A tuple containing the right-side `AbstractFeatureGroup` class and index for join operations.
        join_type: (Optional) The type of join operation (`JoinType`).
        merge_index: (Optional) The index to use for merge operations.
    """

    feature_name: str
    source_class: Optional[Type[Union[AbstractFeatureGroup, str]]] = None
    source_value: Optional[str] = None
    left_link: Optional[Tuple[Type[AbstractFeatureGroup], Union[str, Index]]] = None
    right_link: Optional[Tuple[Type[AbstractFeatureGroup], Union[str, Index]]] = None
    join_type: Optional[JoinType] = None
    merge_index: Optional[Union[str, Index]] = None


class SourceInputFeatureComposite:
    """
    A composite class that handles the logic for defining input features using a source definition.
    """

    @classmethod
    def input_features(cls, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
        """
        Retrieves the set of input features based on the provided options.

        Args:
             options: The options associated with the feature, including source definitions.
             feature_name: The name of the feature, not used by this method.

        Returns:
             A set of Feature objects representing the input features, or None if no input features are defined.

        Raises:
            ValueError: If the `mloda_source_feature` option is missing.
            ValueError: If a source tuple is invalid.
        """

        mloda_source = options.get(DefaultOptionKeys.mloda_source_feature)
        if mloda_source is None:
            raise ValueError(f"Option '{DefaultOptionKeys.mloda_source_feature}' is required for this feature.")

        features = set()
        for source in mloda_source:
            feature = cls._create_feature(source)
            features.add(feature)

        if options.get("initial_requested_data"):
            for feature in features:
                feature.initial_requested_data = True  # Set all features to initial requested data

        return features

    @classmethod
    def _create_feature(cls, source: Union[str, SourceTuple]) -> Feature:
        """
        Helper method to create a Feature object from a source definition.
        """
        if isinstance(source, str):
            return Feature(name=source)
        else:
            try:
                source_tuple = SourceTuple(*source)
            except TypeError as e:
                raise ValueError(f"Invalid source tuple: {source}") from e
            return cls._handle_tuple(source_tuple)

    @classmethod
    def _handle_tuple(cls, source: SourceTuple) -> Feature:
        """
        Handles the creation of a Feature with dependent properties and join definitions.

        Args:
            source: A SourceTuple containing feature information.

        Returns:
            A Feature object.

        Details:
            - Only feature_name is required in SourceTuple.
            - For local feature scope, source_class and source_value can be defined.
            - For merge and join operations, left_link and right_link classes and join_type can be defined.
            - For append and union operations, merge_index can be added.
        """

        properties: Dict[str, Any] = {}
        if source.source_class:
            properties = {
                source.source_class.__name__
                if isinstance(source.source_class, type)
                else str(source.source_class): source.source_value
            }

        link, index = None, None

        if source.left_link is not None and source.right_link is not None and source.join_type is not None:
            link = cls._handle_link(source.left_link, source.right_link, source.join_type)

        if source.merge_index:
            index = Index((source.merge_index,)) if isinstance(source.merge_index, str) else source.merge_index

        return Feature(name=source.feature_name, link=link, index=index, options=properties)

    @classmethod
    def _handle_link(
        cls,
        left_link: Tuple[Type[AbstractFeatureGroup], Union[str, Index]],
        right_link: Tuple[Type[AbstractFeatureGroup], Union[str, Index]],
        join_type: Any,
    ) -> Link:
        """
        Creates a Link object for joining data from different source features.

        Args:
           left_link: Tuple containing the left-side feature group class and index.
           right_link: Tuple containing the right-side feature group class and index.
           join_type: The JoinType of the link.

        Returns:
           A Link object for joining data.

        Raises:
            ValueError: If any of the link inputs are missing.
        """

        if right_link is None or left_link is None or join_type is None:
            raise ValueError(f"Link classes are required for handling link: {left_link} {right_link} {join_type}.")

        left_link_cls, left_index = left_link
        right_link_cls, right_index = right_link

        left_index = Index((left_index,)) if isinstance(left_index, str) else left_index
        right_index = Index((right_index,)) if isinstance(right_index, str) else right_index

        join_func = cls._get_join_func(join_type)

        link_obj = join_func(
            (left_link_cls, left_index),
            (right_link_cls, right_index),
        )

        if isinstance(link_obj, Link):
            return link_obj
        raise ValueError(f"Failed to create link for join type {join_type}, {link_obj}")

    @classmethod
    def _get_join_func(cls, join_type: JoinType) -> Any:
        """
        Retrieves the correct Link method for the given JoinType.
        """

        jointype = JoinType(join_type) if isinstance(join_type, str) else join_type
        if jointype not in JoinType:
            raise ValueError(f"Join type {jointype} is not supported.")

        join_func_mapping = {
            JoinType.APPEND: Link.append,
            JoinType.UNION: Link.union,
            JoinType.OUTER: Link.outer,
            JoinType.INNER: Link.inner,
            JoinType.LEFT: Link.left,
            JoinType.RIGHT: Link.right,
        }
        return join_func_mapping[jointype]
