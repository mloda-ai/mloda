from abc import ABC
from typing import Any, List, Optional, Tuple, Type

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.options import Options


import logging

from mloda_core.abstract_plugins.components.utils import get_all_subclasses


logger = logging.getLogger(__name__)


class BaseInputData(ABC):
    def __init__(self) -> None:
        pass

    @classmethod
    def data_access_name(cls) -> str:
        """This function should return the name of the data access."""
        return cls.__name__

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
        We check for the feature scrope data access if any child classes match the data access.
        """
        subcls = get_all_filtereted_subclasses(BaseInputData, cls)
        for sub in subcls:
            for key, value in options.data.items():
                _key = cls.deal_with_base_input_data_name_as_cls_or_str(key)

                if _key == sub.data_access_name():
                    matched_data_access = sub.match_subclass_data_access(value, [feature_name])  # type: ignore
                    if matched_data_access:
                        cls.add_base_input_data_to_options(sub, matched_data_access, options)
                        return True
                    break  # This case is if a feature requests an input feature, which should have scoped access.
        return False

    @classmethod
    def deal_with_base_input_data_name_as_cls_or_str(cls, key: Any) -> str:
        if hasattr(key, "get_class_name"):
            if not issubclass(key, BaseInputData):
                raise ValueError(f"Key {key} is not a subclass of BaseInputData.")
            key = key.get_class_name()

        if not isinstance(key, str):
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

        data_access_cls, matched_data_access = cls.match_data_access([feature_name], data_access_collection)
        if data_access_cls is False:
            return False

        cls.add_base_input_data_to_options(data_access_cls, matched_data_access, options)
        return True

    @classmethod
    def match_data_access(
        cls,
        feature_names: List[str],
        data_access_collection: DataAccessCollection,
    ) -> Tuple[Any, Any]:
        """
        We check for data access collection if any child classes match the data access.
        """
        subcls = get_all_filtereted_subclasses(BaseInputData, cls)

        for sub in subcls:
            matched_data_access = sub.match_subclass_data_access(data_access_collection, feature_names)  # type: ignore
            if matched_data_access:
                return (sub, matched_data_access)
        return False, False

    @classmethod
    def add_base_input_data_to_options(
        cls, cls_to_be_added: Type["BaseInputData"], matched_data_access: Any, options: Options
    ) -> None:
        """
        Adding the found data access class to the options.
        """

        if options.get("BaseInputData"):
            if options.get("BaseInputData") == (cls_to_be_added, matched_data_access):
                return
            already_cls_to_be_added, _ = options.get("BaseInputData")
            raise ValueError(
                f"BaseInputData already set with different values. {cls_to_be_added} != {already_cls_to_be_added}"
            )
        options.add("BaseInputData", (cls_to_be_added, matched_data_access))

    def load(self, features: FeatureSet) -> Any:
        """
        This class should be implemented in intermediary child classes, which use scoped data access.
        """
        raise NotImplementedError

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        """
        This funtion should be implemented in final child classes, which use scoped data access.
        """
        raise NotImplementedError

    @classmethod
    def supports_scoped_data_access(cls) -> bool:
        """
        As we assume that load_data are only implemented in final child classes of scoped data access,
        we can use this function to check if the class supports scoped data access and is the final child class.
        """
        try:
            cls.load_data(None, None)  # type: ignore
        except NotImplementedError:
            return False
        except AttributeError:  # Expected as cls.load_data(None, None) should raise an error
            return True

        return True

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__


def get_all_filtereted_subclasses(cls: Any, parent_class: Any) -> List[Type[BaseInputData]]:
    new_cls = []
    for _cls in get_all_subclasses(cls):
        # This ensures that only the classes that are subclasses of the parent class are considered.
        # ReadFile -> CsvReader
        if not issubclass(_cls, parent_class):
            continue

        if _cls.supports_scoped_data_access():
            new_cls.append(_cls)
    return new_cls
