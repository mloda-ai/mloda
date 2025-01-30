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
    A class to dynamically create AbstractFeatureGroup subclasses based on a dictionary of properties.
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

        def set_feature_name(self, config: Options, feature_name: FeatureName) -> FeatureName:  # type: ignore
            if "set_feature_name" in properties:
                return properties["set_feature_name"](self, config, feature_name)  # type: ignore
            return feature_name

        def match_feature_group_criteria(  # type: ignore
            cls,
            feature_name: Union[FeatureName, str],
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if "match_feature_group_criteria" in properties:
                return properties["match_feature_group_criteria"](cls, feature_name, options, data_access_collection)  # type: ignore
            return super(new_class, cls).match_feature_group_criteria(feature_name, options, data_access_collection)  # type: ignore

        def input_data(cls) -> Optional[BaseInputData]:  # type: ignore
            if "input_data" in properties:
                return properties["input_data"]()  # type: ignore
            return super(new_class, cls).input_data()  # type: ignore

        def validate_input_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:  # type: ignore
            if "validate_input_features" in properties:
                return properties["validate_input_features"](cls, data, features)  # type: ignore
            return super(new_class, cls).validate_input_features(data, features)  # type: ignore

        def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:  # type: ignore
            if "calculate_feature" in properties:
                return properties["calculate_feature"](cls, data, features)
            return super(new_class, cls).calculate_feature(data, features)  # type: ignore

        def validate_output_features(cls, data: Any, features: FeatureSet) -> Optional[bool]:  # type: ignore
            if "validate_output_features" in properties:
                return properties["validate_output_features"](cls, data, features)  # type: ignore
            return super(new_class, cls).validate_output_features(data, features)  # type: ignore

        def artifact(cls) -> Optional[Type[Any]]:  # type: ignore
            if "artifact" in properties:
                return properties["artifact"]()  # type: ignore
            return super(new_class, cls).artifact()  # type: ignore

        def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:  # type: ignore
            if "compute_framework_rule" in properties:
                return properties["compute_framework_rule"]()  # type: ignore
            return super(new_class, cls).compute_framework_rule()  # type: ignore

        def return_data_type_rule(cls, feature: Any) -> Optional[DataType]:  # type: ignore
            if "return_data_type_rule" in properties:
                return properties["return_data_type_rule"](cls, feature)  # type: ignore
            return super(new_class, cls).return_data_type_rule(feature)  # type: ignore

        def input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Any]]:  # type: ignore
            if "input_features" in properties:
                return properties["input_features"](self, options, feature_name)  # type: ignore
            return super(new_class, self).input_features(options, feature_name)  # type: ignore

        def index_columns(cls) -> Optional[List[Index]]:  # type: ignore
            if "index_columns" in properties:
                return properties["index_columns"]()  # type: ignore
            return super(new_class, cls).index_columns()  # type: ignore

        def supports_index(cls, index: Index) -> Optional[bool]:  # type: ignore
            if "supports_index" in properties:
                return properties["supports_index"](cls, index)  # type: ignore
            return super(new_class, cls).supports_index(index)  # type: ignore

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
