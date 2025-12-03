from typing import Any, Dict, Optional, Type, Set, List, Union
from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.options import Options
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.data_types import DataType
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.index.index import Index
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData


class DynamicFeatureGroupCreator:
    """
    Base class for dynamically creating AbstractFeatureGroup subclasses at runtime.

    This factory enables programmatic creation of feature groups by specifying their
    behavior through a properties dictionary. It's useful for generating feature groups
    on-the-fly without writing explicit class definitions, particularly in scenarios
    requiring dynamic behavior based on runtime conditions.

    ## Key Capabilities

    - Create feature group classes at runtime without explicit class definitions
    - Override specific methods (calculate_feature, match_feature_group_criteria, etc.)
    - Inherit from any AbstractFeatureGroup subclass (e.g., ReadFileFeature, SourceInputFeature)
    - Cache created classes to avoid duplicate definitions
    - Support full feature group lifecycle customization

    ## Common Use Cases

    - Creating file-specific reader feature groups dynamically
    - Building feature groups with computed matching criteria
    - Generating temporary feature groups for joins or transformations
    - Extending existing feature groups with custom behavior

    ## Available Method Overrides

    All methods from AbstractFeatureGroup can be overridden via the properties dictionary:

    - `set_feature_name`: Customize feature name resolution
    - `match_feature_group_criteria`: Define custom matching logic
    - `input_data`: Specify input data sources
    - `validate_input_features`: Add input validation logic
    - `calculate_feature`: Define feature calculation logic
    - `validate_output_features`: Add output validation logic
    - `artifact`: Specify artifact types
    - `compute_framework_rule`: Define compute framework compatibility
    - `return_data_type_rule`: Specify return data types
    - `input_features`: Define input feature dependencies
    - `index_columns`: Specify index column requirements
    - `supports_index`: Define index support logic

    ## Usage Examples

    ### Basic Dynamic Feature Group

    ```python
    from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory import (
        DynamicFeatureGroupCreator
    )
    from mloda_core.abstract_plugins.components.feature_name import FeatureName

    # Define custom behavior
    properties = {
        "calculate_feature": lambda cls, data, features: data * 2,
        "match_feature_group_criteria": lambda cls, feature_name, options, dac: (
            feature_name == FeatureName("double_value")
        ),
    }

    # Create the dynamic feature group class
    DoubleValueFG = DynamicFeatureGroupCreator.create(
        properties=properties,
        class_name="DoubleValueFeatureGroup"
    )

    # Use like any other feature group
    # The class is now registered and can match "double_value" features
    ```

    ### Dynamic File Reader Feature Group

    ```python
    from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
    from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader

    properties = {
        "input_data": lambda: PyFileReader(),
        "calculate_feature": lambda cls, data, features: process_file_content(data),
        "match_feature_group_criteria": lambda cls, fn, opts, dac: (
            fn == FeatureName("my_file_reader")
        ),
    }

    # Inherit from ReadFileFeature for file reading capabilities
    MyFileReaderFG = DynamicFeatureGroupCreator.create(
        properties=properties,
        class_name="MyFileReaderFeatureGroup",
        feature_group_cls=ReadFileFeature
    )
    ```

    ### Dynamic Join Feature Group

    ```python
    from mloda_plugins.feature_group.experimental.source_input_feature import SourceInputFeature

    def calculate_join(cls, data, features):
        # Combine columns from joined data
        data["combined"] = data[data.columns]
        return data

    properties = {
        "calculate_feature": calculate_join,
    }

    # Create a join feature group
    JoinFG = DynamicFeatureGroupCreator.create(
        properties=properties,
        class_name="DynamicJoinFeatureGroup",
        feature_group_cls=SourceInputFeature
    )
    ```

    ### Complex Custom Logic

    ```python
    from mloda_core.abstract_plugins.components.feature import Feature

    def custom_input_features(self, options, feature_name):
        # Return dynamically determined input features
        return {Feature(name="computed_input_feature")}

    def custom_match_criteria(cls, feature_name, options, dac):
        # Match based on runtime conditions
        if isinstance(feature_name, FeatureName):
            feature_name = feature_name.name
        return "custom_prefix" in feature_name

    properties = {
        "match_feature_group_criteria": custom_match_criteria,
        "input_features": custom_input_features,
        "compute_framework_rule": lambda: {PandasDataFrame},
    }

    CustomFG = DynamicFeatureGroupCreator.create(
        properties=properties,
        class_name="CustomLogicFeatureGroup"
    )
    ```

    ## Parameters

    ### create() Method Parameters

    - `properties`: Dictionary mapping method names to callable implementations
    - `class_name`: Name for the dynamically created class (used for caching)
    - `feature_group_cls`: Base class to inherit from (default: AbstractFeatureGroup)

    ## Implementation Details

    - Created classes are cached in `_created_classes` dictionary
    - Requesting the same `class_name` twice returns the cached class
    - Properties use lambda functions or regular functions as method implementations
    - Method signatures must match the original AbstractFeatureGroup signatures
    - Unspecified methods fall back to parent class implementations

    ## Requirements

    - Properties dictionary with valid method names from AbstractFeatureGroup
    - Callable implementations matching expected method signatures
    - Unique class_name for each distinct feature group type
    - Base class must be AbstractFeatureGroup or its subclass

    ## Real-World Example

    See `ConcatenatedFileContent` in `read_context_files.py` for a production
    example that uses DynamicFeatureGroupCreator to create file-reading feature
    groups on-the-fly for joining multiple files.
    """

    _created_classes: Dict[str, Type[AbstractFeatureGroup]] = {}  # Store created classes

    @staticmethod
    def create(
        properties: Dict[str, Any],
        class_name: str = "DynamicFeatureGroup",
        feature_group_cls: Type[AbstractFeatureGroup] = AbstractFeatureGroup,
    ) -> Type[AbstractFeatureGroup]:
        """
        Creates a new AbstractFeatureGroup subclass with the given properties.

        Args:
            properties: A dictionary containing the properties for the new class.
            class_name: The name of the new class.

        Returns:
            A new AbstractFeatureGroup subclass.
        """

        if class_name in DynamicFeatureGroupCreator._created_classes:
            return DynamicFeatureGroupCreator._created_classes[class_name]

        def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:  # type: ignore[no-untyped-def]
            if "set_feature_name" in properties:
                return properties["set_feature_name"](self, config, feature_name)  # type: ignore[no-any-return]
            return feature_name

        def match_feature_group_criteria(  # type: ignore[no-untyped-def]
            cls,
            feature_name: Union[FeatureName, str],
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if "match_feature_group_criteria" in properties:
                return properties["match_feature_group_criteria"](cls, feature_name, options, data_access_collection)  # type: ignore[no-any-return]
            return super(new_class, cls).match_feature_group_criteria(feature_name, options, data_access_collection)  # type: ignore[misc, arg-type, no-any-return]

        def input_data(cls) -> Optional[BaseInputData]:  # type: ignore[no-untyped-def]
            if "input_data" in properties:
                return properties["input_data"]()  # type: ignore[no-any-return]
            return super(new_class, cls).input_data()  # type: ignore[misc, arg-type, no-any-return]

        def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:  # type: ignore[no-untyped-def]
            if "validate_input_features" in properties:
                return properties["validate_input_features"](cls, data, features)  # type: ignore[no-any-return]
            return super(new_class, cls).validate_input_features(data, features)  # type: ignore[misc, arg-type, no-any-return]

        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:  # type: ignore[no-untyped-def]
            if "calculate_feature" in properties:
                return properties["calculate_feature"](cls, data, features)
            return super(new_class, cls).calculate_feature(data, features)  # type: ignore[misc, arg-type]

        def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:  # type: ignore[no-untyped-def]
            if "validate_output_features" in properties:
                return properties["validate_output_features"](cls, data, features)  # type: ignore[no-any-return]
            return super(new_class, cls).validate_output_features(data, features)  # type: ignore[misc, arg-type, no-any-return]

        def artifact(cls) -> Optional[Type[Any]]:  # type: ignore[no-untyped-def]
            if "artifact" in properties:
                return properties["artifact"]()  # type: ignore[no-any-return]
            return super(new_class, cls).artifact()  # type: ignore[misc, arg-type, no-any-return]

        def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:  # type: ignore[no-untyped-def]
            if "compute_framework_rule" in properties:
                return properties["compute_framework_rule"]()  # type: ignore[no-any-return]
            return super(new_class, cls).compute_framework_rule()  # type: ignore[misc, arg-type, no-any-return]

        def return_data_type_rule(cls, feature: Any) -> Optional[DataType]:  # type: ignore[no-untyped-def]
            if "return_data_type_rule" in properties:
                return properties["return_data_type_rule"](cls, feature)  # type: ignore[no-any-return]
            return super(new_class, cls).return_data_type_rule(feature)  # type: ignore[misc, arg-type, no-any-return]

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Any]]:  # type: ignore[no-untyped-def]
            if "input_features" in properties:
                return properties["input_features"](self, options, feature_name)  # type: ignore[no-any-return]
            return super(new_class, self).input_features(options, feature_name)  # type: ignore[misc, arg-type, no-any-return]

        def index_columns(cls) -> Optional[List[Index]]:  # type: ignore[no-untyped-def]
            if "index_columns" in properties:
                return properties["index_columns"]()  # type: ignore[no-any-return]
            return super(new_class, cls).index_columns()  # type: ignore[misc, arg-type, no-any-return]

        def supports_index(cls, index: Index) -> Optional[bool]:  # type: ignore[no-untyped-def]
            if "supports_index" in properties:
                return properties["supports_index"](cls, index)  # type: ignore[no-any-return]
            return super(new_class, cls).supports_index(index)  # type: ignore[misc, arg-type, no-any-return]

        new_class = type(
            class_name,
            (feature_group_cls,),
            {
                "set_feature_name": set_feature_name,
                "match_feature_group_criteria": classmethod(match_feature_group_criteria),
                "input_data": classmethod(input_data),
                "validate_input_features": classmethod(validate_input_features),
                "calculate_feature": classmethod(calculate_feature),
                "validate_output_features": classmethod(validate_output_features),
                "artifact": classmethod(artifact),
                "compute_framework_rule": classmethod(compute_framework_rule),
                "return_data_type_rule": classmethod(return_data_type_rule),
                "input_features": input_features,
                "index_columns": classmethod(index_columns),
                "supports_index": classmethod(supports_index),
            },
        )
        DynamicFeatureGroupCreator._created_classes[class_name] = new_class
        return new_class
