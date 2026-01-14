from typing import Any, Dict, List, Optional, Tuple, Type, Set, Union
from mloda.provider import FeatureGroup
from mloda.user import Feature
from mloda.user import Options
from mloda.user import FeatureName
from mloda.user import DataAccessCollection
from mloda.provider import FeatureSet
from mloda.user import DataType
from mloda.provider import ComputeFramework
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.experimental.source_input_feature import (
    SourceInputFeature,
    SourceInputFeatureComposite,
    SourceTuple,
)
from mloda_plugins.feature_group.input_data.read_file import ReadFile
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature


class TestDynamicFeatureGroupFactory:
    def test_dynamic_feature_group_creator(self) -> None:
        # Example property dictionary
        properties: Dict[str, Any] = {
            "calculate_feature": lambda cls, data, features: f"Calculated with {cls.__name__}",
            "match_feature_group_criteria": lambda cls, feature_name, options, data_access_collection: feature_name
            == FeatureName("test_feature"),
            "return_data_type_rule": lambda cls, feature: DataType.STRING,
            "input_features": lambda self, options, feature_name: None,
        }

        # Create a dynamic feature group
        DynamicTestFeatureGroup = DynamicFeatureGroupCreator.create(properties, class_name="DynamicTestFeatureGroup")

        # Test match criteria
        options = Options()
        assert DynamicTestFeatureGroup.match_feature_group_criteria(FeatureName("test_feature"), options)
        assert not DynamicTestFeatureGroup.match_feature_group_criteria(FeatureName("other_feature"), options)

        # Test calculate feature
        features = FeatureSet()
        features.add(Feature("example"))
        data: Dict[Any, Any] = {}  # Example data
        result = DynamicTestFeatureGroup.calculate_feature(data, features)
        assert result == "Calculated with DynamicTestFeatureGroup"

        # Test data type rule
        feature = Any
        result_type = DynamicTestFeatureGroup.return_data_type_rule(feature)  # type: ignore
        assert result_type == DataType.STRING

        # Test input features, as no error was thrown.
        result_names = DynamicTestFeatureGroup().input_features(options, Any)  # type: ignore
        assert result_names is None

        # Test that the created class is a subclass of FeatureGroup
        assert issubclass(DynamicTestFeatureGroup, FeatureGroup)

    def test_dynamic_feature_group_creator_with_readfile_feature(self) -> None:
        class MockReadFile(ReadFile):
            @classmethod
            def suffix(cls) -> Tuple[str, ...]:
                return (".mock",)

            @classmethod
            def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
                return "mock_data"

        # Example property dictionary
        properties: Dict[str, Any] = {
            "input_data": lambda: MockReadFile(),
            "calculate_feature": lambda cls, data, features: f"Calculated with {cls.__name__} and {data}",
            "match_feature_group_criteria": lambda cls, feature_name, options, data_access_collection: feature_name
            == FeatureName("test_file_feature"),
        }

        # Create a dynamic feature group
        DynamicTestFeatureGroup = DynamicFeatureGroupCreator.create(
            properties, class_name="DynamicTestFileFeatureGroup", feature_group_cls=ReadFileFeature
        )

        # Test match criteria
        options = Options()
        assert DynamicTestFeatureGroup.match_feature_group_criteria(FeatureName("test_file_feature"), options)
        assert not DynamicTestFeatureGroup.match_feature_group_criteria(FeatureName("other_feature"), options)

        # Test input data
        assert isinstance(DynamicTestFeatureGroup.input_data(), MockReadFile)

        # Test calculate feature
        features = FeatureSet()
        features.add(Feature("example"))
        data = "mock_data"  # Example data
        result = DynamicTestFeatureGroup.calculate_feature(data, features)
        assert result == "Calculated with DynamicTestFileFeatureGroup and mock_data"

        # Test that the created class is a subclass of FeatureGroup
        assert issubclass(DynamicTestFeatureGroup, FeatureGroup)
        assert issubclass(DynamicTestFeatureGroup, ReadFileFeature)

    def test_dynamic_feature_group_creator_with_complex_logic(self) -> None:
        def custom_set_feature_name(self: Any, config: Options, feature_name: FeatureName) -> FeatureName:
            return FeatureName(f"custom_{feature_name.name}")

        def custom_match_criteria(
            cls: Type[FeatureGroup],
            feature_name: Union[FeatureName, str],
            options: Options,
            data_access_collection: Optional[DataAccessCollection] = None,
        ) -> bool:
            if isinstance(feature_name, FeatureName):
                feature_name = feature_name.name
            return "custom" in feature_name

        def custom_input_features(self: Any, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
            return {Feature(name="custom_input_feature")}

        def custom_compute_framework_rule() -> Union[bool, Set[Type[ComputeFramework]]]:
            return {PandasDataFrame}

        def custom_index_columns() -> Optional[List[Index]]:
            return [Index(("a",)), Index(("b", "c"))]

        def custom_supports_index(cls, index: Index) -> Optional[bool]:  # type: ignore
            return index.is_multi_index() is False

        properties: Dict[str, Any] = {
            "set_feature_name": custom_set_feature_name,
            "match_feature_group_criteria": custom_match_criteria,
            "input_features": custom_input_features,
            "compute_framework_rule": custom_compute_framework_rule,
            "index_columns": custom_index_columns,
            "supports_index": custom_supports_index,
        }

        # Create a dynamic feature group
        DynamicTestFeatureGroup = DynamicFeatureGroupCreator.create(properties, class_name="DynamicComplexFeatureGroup")

        # Test custom set feature name logic
        options = Options()
        feature_name = FeatureName("test_feature")
        new_feature_name = DynamicTestFeatureGroup().set_feature_name(options, feature_name)
        assert new_feature_name == FeatureName("custom_test_feature")

        # Test custom match criteria logic
        options = Options()
        assert DynamicTestFeatureGroup.match_feature_group_criteria("custom_feature", options)
        assert not DynamicTestFeatureGroup.match_feature_group_criteria("test_feature", options)

        # Test custom input features logic
        options = Options()
        input_features = DynamicTestFeatureGroup().input_features(options, feature_name)
        assert input_features == {Feature(name="custom_input_feature")}

        # Test compute framework rule logic
        assert DynamicTestFeatureGroup.compute_framework_rule() == {PandasDataFrame}

        # Test index columns
        assert DynamicTestFeatureGroup.index_columns() == [Index(("a",)), Index(("b", "c"))]

        # Test supports index
        assert DynamicTestFeatureGroup.supports_index(Index(("a",))) is True
        assert DynamicTestFeatureGroup.supports_index(Index(("a", "b"))) is False

        # Test that the created class is a subclass of FeatureGroup
        assert issubclass(DynamicTestFeatureGroup, FeatureGroup)

    def test_dynamic_feature_group_with_source_input_composite_with_initial_requested_data_and_simple_string(
        self,
    ) -> None:
        """
        Test case for creating a dynamic feature group using SourceInputFeatureComposite with a simple string.
        """

        def custom_input_features(self, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:  # type: ignore
            return SourceInputFeatureComposite.input_features(options, feature_name)

        properties: Dict[str, Any] = {
            "input_features": custom_input_features,
        }

        # Define a dynamic feature group with source input handling
        DynamicTestFeatureGroup = DynamicFeatureGroupCreator.create(
            properties, class_name="DynamicTestSourceInputFeatureGroup", feature_group_cls=SourceInputFeature
        )

        # Define options including in_features
        options = Options(
            {
                DefaultOptionKeys.in_features: frozenset(["source_feature_1"]),
                "initial_requested_data": True,
            }
        )
        feature_name = FeatureName("test_feature")
        # Get input features from the dynamic feature group
        input_features = DynamicTestFeatureGroup().input_features(options, feature_name)
        assert input_features is not None
        assert len(input_features) == 1

        feature = next(iter(input_features))
        assert feature.name == "source_feature_1"
        assert feature.initial_requested_data is True

        assert issubclass(DynamicTestFeatureGroup, FeatureGroup)

    def test_dynamic_feature_group_with_source_input_composite_inheritance(self) -> None:
        """
        Test case for creating a dynamic feature group using SourceInputFeatureComposite inheriting from SourceInputFeature.
        """

        def custom_input_features(self: Any, options: Options, feature_name: FeatureName) -> Optional[Set[Feature]]:
            return SourceInputFeatureComposite.input_features(options, feature_name)

        properties: Dict[str, Any] = {
            "input_features": custom_input_features,
        }

        # Define a dynamic feature group with source input handling
        DynamicTestFeatureGroup = DynamicFeatureGroupCreator.create(
            properties, class_name="DynamicTestSourceInputFeatureGroup", feature_group_cls=SourceInputFeature
        )

        class ConcreteFeatureGroup(DynamicTestFeatureGroup):  # type: ignore
            pass

        # Define options including in_features
        options = Options(
            {
                DefaultOptionKeys.in_features: frozenset(
                    [
                        SourceTuple(
                            feature_name="source_feature_1", source_class=ReadFileFeature, source_value="test.csv"
                        )
                    ]
                )
            }
        )
        feature_name = FeatureName("test_feature")
        # Get input features from the dynamic feature group
        input_features = ConcreteFeatureGroup().input_features(options, feature_name)
        assert input_features is not None
        assert len(input_features) == 1

        feature = next(iter(input_features))
        assert feature.name == "source_feature_1"
        assert feature.options.get("ReadFileFeature") == "test.csv"

        assert issubclass(ConcreteFeatureGroup, FeatureGroup)
        assert issubclass(ConcreteFeatureGroup, SourceInputFeature)
