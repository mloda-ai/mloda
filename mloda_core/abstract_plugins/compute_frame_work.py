from abc import ABC
from typing import Any, List, Optional, Set, Type, Union, final
from uuid import UUID, uuid4
from mloda_core.abstract_plugins.components.merge.base_merge_engine import BaseMergeEngine
import pyarrow as pa

from mloda_core.abstract_plugins.components.cfw_transformer import ComputeFrameworkTransformMap
from mloda_core.abstract_plugins.function_extender import WrapperFunctionExtender, WrapperFunctionEnum
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.parallelization_modes import ParallelizationModes
from mloda_core.filter.filter_engine import BaseFilterEngine
from mloda_core.runtime.flight.flight_server import FlightServer


class ComputeFrameWork(ABC):
    """
    Documentation ComputeFrameWork:

    This class is used to define the compute framework.

    A compute framework must be in the namespace of the python project and must inherit from this class.
    This way, we can run feature computation on multiple frameworks.
    Usecases:
    - Online and Offline computation
    - Testing
    - Migrations from one framework to another

    As you can have multiple compute frameworks, you can define a feature group to be computed on multiple compute frameworks.
    However, in the space of a run, it is necessary that there is only exactly one way to compute a feature of feature group,
    we can limit the compute framework from three sides:
    1) feature definition - only one compute framework can be set for one feature
    2) feature group definition - compute frameworks that can be used for the feature group
    3) from api request side - limits compute frameworks which can be used

    Of course, for migrations, we have allow multiple definitions of the same feature group on different compute frameworks.
    This usecase however is currently not supported, as one could just run the module twice and compare the result datasets for now.
    """

    def __init__(
        self,
        mode: ParallelizationModes,
        children_if_root: frozenset[UUID],
        uuid: UUID = uuid4(),
        function_extender: Optional[Set[WrapperFunctionExtender]] = None,
    ) -> None:
        """This class is initialized step execution."""
        self.mode = mode
        self.data: Any = None
        self.children_if_root = children_if_root
        self.already_calculated_children_tracker: Set[UUID] = set()
        self.column_names: Set[str] = set()
        self.function_extender = function_extender if function_extender is not None else set()

        self.uuid = uuid

        # collects all datasets which were created based on this feature group except if it is the !final! feature
        self.object_ids: List[str] = []

    @staticmethod
    def expected_data_framework() -> Any:
        """
        This function should return the expected data framework for the compute framework.
        However, we only need to set it if we really want to be sure that the datatype is correct.
        """
        return None

    def filter_engine(self) -> Type[BaseFilterEngine]:
        """
        This function should return the filtered data.
        The BaseFilterEngine should be overwritten by the appropriate ComputeFrameWork if needed
        """
        raise NotImplementedError

    @staticmethod
    def get_framework_transform_functions(from_other: bool, other: Any) -> Any:
        """
        This function is used to transfor data from one expected data framework to another.
        """
        return None, {}

    def transform(
        self,
        data: Any,
        feature_names: Set[str],
        transform_map: ComputeFrameworkTransformMap = ComputeFrameworkTransformMap(),
    ) -> Any:
        """This function should be used to transform the data.
        The idea here is that we can transform the data to a common format.
        At the end, this format is represented by the expected_data_framework
        and thus, defined by the compute framework abstraction.

        If you wish not to transform the data, you can leave this.
        This is not relevant if you stay in one compute framework as you don t need to switch it.
        """
        transformed_data = self.transform_refactored(data, transform_map)
        if transformed_data is not None:
            return transformed_data
        return data

    @final
    def transform_refactored(self, data: Any, transform_map: ComputeFrameworkTransformMap) -> Any:
        """
        This part is also refactored to be more readable.

        The part, where we add single columns etc is not done yet.
        """
        if isinstance(data, self.expected_data_framework()):
            return data

        if transform_map.transform_map:
            return transform_map.transform(data, type(data), self.expected_data_framework(), {})

        func, param = self.get_framework_transform_functions(from_other=True, other=type(data))
        if func:
            return func(data, param)

        return None

    def select_data_by_column_names(self, data: Any, selected_feature_names: Set[FeatureName]) -> Any:
        """
        If you only want to store the requested features, implement this functionality depending on your framework.

        e.g. if you are using pyarrow, you can use the following code:
        return data.select([f.name for f in selected_feature_names])
        """
        return data

    def merge_engine(self) -> Type[BaseMergeEngine]:
        """
        This function should return a subclass of the BaseMergeEngine.
        With this, we can merge data from the same compute framework.

        This implementation is optional.
        """
        raise NotImplementedError(
            f"Merge functionality is for this compute framework not implemented {self.__class__.__name__}."
        )

    def set_column_names(self) -> None:
        pass

    @final
    def run_calculation(
        self, feature_group: Any, features: Any, location: str | None, data: Optional[Any] = None
    ) -> Optional[Any]:
        # case multiprocessing or case base api input feature
        if data is not None:
            self.data = data

        self.run_validate_input_features(feature_group, features)

        # every one does this
        features = self.set_filter_engine(features)
        data = self.run_calculate_feature(feature_group, features)
        data = self.run_final_filter(data, features)

        names = features.get_all_names()
        self.data = self.transform(data, names)

        self.set_column_names()

        self.run_validate_output_features(feature_group, features)

        # case threading/sync
        if not location:
            return None

        # case multiprocessing
        # return data to be used in next step of this framework in this process
        if len(self.children_if_root) > len(self.already_calculated_children_tracker) + len(features.features):
            return self.data

        # upload finished dataset to flight server
        self.data = self.upload_finished_data(location)
        return self.data

    @final
    def run_final_filter(self, data: Any, features: Any) -> Any:
        if features.filter_engine is None:
            return data

        try:
            if features.filter_engine().final_filters() is False:
                return data
        except NotImplementedError:
            return data

        filter_engine = features.filter_engine()
        return filter_engine.apply_filters(data, features)

    @final
    def set_filter_engine(self, features: Any) -> Any:
        """We set the filter engine of the feature set here.
        With this, we can run filters on the data during the calculate process.

        This is needed as we have different times when we want to apply filters.

        -   We can apply filters on final data sets e.g. reading csv like pandas
        -   We can apply filters on the data during read e.g. reading from a database
        -   We can apply filters on the data during the calculation process e.g. pyarrow
        """
        try:
            features.filter_engine = self.filter_engine

            if not issubclass(features.filter_engine(), BaseFilterEngine):
                raise ValueError(f"Filter engine {self.filter_engine} not supported by {self.__class__.__name__}")

        except NotImplementedError:
            pass
        return features

    @final
    def run_validate_input_features(self, feature_group: Any, features: Any) -> Any:
        if self.data is None:
            return

        extender = self.get_function_extender(WrapperFunctionEnum.VALIDATE_INPUT_FEATURE)
        if extender is None:
            result = feature_group.validate_input_features(self.data, features)
        else:
            result = extender(feature_group.validate_input_features, self.data, features)

        if result is None or result is True:
            return
        raise ValueError(result)

    @final
    def run_validate_output_features(self, feature_group: Any, features: Any) -> Any:
        if self.data is None:
            return
        extender = self.get_function_extender(WrapperFunctionEnum.VALIDATE_OUTPUT_FEATURE)
        if extender is None:
            result = feature_group.validate_output_features(self.data, features)
        else:
            result = extender(feature_group.validate_output_features, self.data, features)

        if result is None or result is True:
            return
        raise ValueError(result)

    @final
    def get_column_names(self) -> Set[str]:
        return self.column_names

    @classmethod
    @final
    def get_class_name(cls) -> str:
        return cls.__name__

    @final
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComputeFrameWork):
            return False
        return self.get_class_name() == other.get_class_name() and self.children_if_root == other.children_if_root

    @final
    def __hash__(self) -> int:
        return hash((self.get_class_name(), self.children_if_root))

    @final
    def validate_expected_framework(self, location: Optional[str] = None) -> None:
        if self.expected_data_framework() is None:
            return

        if self.data is None:
            return

        # If location is a string, it means it is a uuid of the object in arrow flight.
        if isinstance(location, str) and self.data is not None:
            return

        if not isinstance(self.data, self.expected_data_framework()):
            raise ValueError(f"Data type {type(self.data)} is not supported by {self.__class__.__name__}")

    @final
    def add_already_calculated_children_and_drop_if_possible(
        self, children: Set[UUID], location: Optional[str] = None
    ) -> Union[bool, frozenset[UUID]]:
        # if len(self.object_ids) > 1:
        #    if location is None:
        #        raise ValueError("Location is not set")
        #    self.drop_data(set(self.object_ids[:-1]), location)

        self.already_calculated_children_tracker.update(children)

        if self.children_if_root.issubset(self.already_calculated_children_tracker):
            self.drop_last_data(location)
            return True

        if len(self.object_ids) > 0:
            return self.children_if_root

        return False

    @final
    def get_function_extender(self, wrapper_function_enum: WrapperFunctionEnum) -> Optional[WrapperFunctionExtender]:
        found_extender = None
        for extender in self.function_extender:
            if wrapper_function_enum in extender.wraps():
                if found_extender is not None:
                    raise ValueError(
                        f"Multiple function_extender found for {wrapper_function_enum}, {found_extender.__name__}, {extender.__name__}"
                    )
                found_extender = extender

        if found_extender is not None:
            return found_extender
        return None

    @final
    def run_calculate_feature(self, feature_group: Any, features: Any) -> Any:
        extender = self.get_function_extender(WrapperFunctionEnum.FEATURE_GROUP_CALCULATE_FEATURE)

        if extender is None:
            return feature_group.calculate_feature(self.data, features)
        return extender(feature_group.calculate_feature, self.data, features)

    @final
    def set_data(self, data: Any) -> None:
        self.data = data

    @final
    def get_data(self) -> Any:
        return self.data

    @final
    def get_object_ids(self) -> List[str]:
        return self.object_ids

    @final
    def upload_finished_data(self, location: str) -> str:
        """Uploads table by using its own cfw uuid."""
        return self.upload_table(location, self.uuid)

    @final
    def upload_table(self, location: str, object_id: Optional[UUID] = None) -> str:
        if object_id is None:
            object_id = uuid4()
        _object_id = str(object_id)

        func, param = self.get_framework_transform_functions(from_other=False, other=pa.Table)
        if func:
            self.data = func(self.data, param)

        FlightServer.upload_table(location, self.data, _object_id)
        self.object_ids.append(_object_id)
        return _object_id

    @final
    @classmethod
    def convert_flyserver_data_back(cls, data: Any) -> Any:
        if not isinstance(data, pa.Table):
            return data
        if isinstance(data, cls.expected_data_framework()):
            return data

        func, param = cls.get_framework_transform_functions(from_other=True, other=pa.Table)
        if func:
            return func(data, param)

        raise ValueError(
            f"Conversion from {type(data)} to {cls.expected_data_framework()} is not supported. This can be created, when a flyserver was used."
        )

    @final
    def drop_data(self, table_keys: Set[str], location: str) -> None:
        FlightServer.drop_tables(location, table_keys)

    @final
    def drop_last_data(self, location: Optional[str] = None) -> None:
        if isinstance(self.data, str) and location:
            self.drop_data({self.data}, location)

        self.data = None

    @final
    def get_uuid(self) -> UUID:
        if self.uuid is None:
            raise ValueError("UUID is not set")
        return self.uuid
