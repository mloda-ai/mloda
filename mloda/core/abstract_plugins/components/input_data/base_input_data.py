from abc import ABC
from typing import Any, Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options


from mloda.core.abstract_plugins.components.utils import get_all_subclasses


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
        We check for the feature scope data access if any child classes match the data access.
        """
        subclasses = get_all_filtered_subclasses(BaseInputData, cls)
        for subclass in subclasses:
            for key, value in options.items():
                _key = cls.deal_with_base_input_data_name_as_cls_or_str(key)

                if _key == subclass.data_access_name():
                    matched_data_access = subclass.match_subclass_data_access(value, [feature_name], options=options)  # type: ignore[attr-defined]
                    if matched_data_access:
                        cls.add_base_input_data_to_options(subclass, matched_data_access, options)
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

        data_access_cls, matched_data_access = cls.match_data_access(
            [feature_name], data_access_collection, options=options
        )
        if data_access_cls is None:
            return False

        cls.add_base_input_data_to_options(data_access_cls, matched_data_access, options)
        return True

    @classmethod
    def match_data_access(
        cls,
        feature_names: list[str],
        data_access_collection: DataAccessCollection,
        options: Optional[Options] = None,
    ) -> tuple[Any, Any]:
        """
        We check for data access collection if any child classes match the data access.
        """
        subclasses = get_all_filtered_subclasses(BaseInputData, cls)

        for subclass in subclasses:
            matched_data_access = subclass.match_subclass_data_access(  # type: ignore[attr-defined]
                data_access_collection, feature_names, options=options
            )
            if matched_data_access:
                return (subclass, matched_data_access)
        return None, None

    @classmethod
    def add_base_input_data_to_options(
        cls, cls_to_be_added: type["BaseInputData"], matched_data_access: Any, options: Options
    ) -> None:
        """
        Adding the found data access class to the options.
        """

        if options.get("BaseInputData"):
            existing_data = options.get("BaseInputData")
            if existing_data == (cls_to_be_added, matched_data_access):
                return

            already_cls_to_be_added, _ = existing_data
            raise ValueError(
                f"BaseInputData already set with different values. {cls_to_be_added} != {already_cls_to_be_added}"
            )
        options.add_to_group("BaseInputData", (cls_to_be_added, matched_data_access))

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
            cls.load_data(None, None)  # type: ignore[arg-type]
        except NotImplementedError:
            return False
        except AttributeError:  # Expected as cls.load_data(None, None) should raise an error
            return True

        return True

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def validate_columns(cls, file_name: str, feature_names: list[str]) -> bool:
        return True

    @classmethod
    def _has_suffix(cls) -> bool:
        """Check if this class implements suffix() (concrete subclass vs abstract base)."""
        try:
            cls.suffix()  # type: ignore[attr-defined]
            return True
        except NotImplementedError:
            return False

    @classmethod
    def _matches_suffix(cls, path: str) -> bool:
        """Check if a file path matches this class's suffix, or True if no suffix defined."""
        if not cls._has_suffix():
            return True
        return path.endswith(cls.suffix())  # type: ignore[attr-defined]

    @classmethod
    def _resolve_pinned_file(cls, data_access: Any, feature_names: list[str]) -> Optional[str]:
        column_map: dict[str, str] = data_access.column_to_file
        pinned_paths: set[str] = {column_map[name] for name in feature_names if name in column_map}
        if not pinned_paths:
            return None
        for name in feature_names:
            if name not in column_map:
                raise ValueError(f"Mixed batch: some features pinned, others not: {feature_names}")
        if len(pinned_paths) == 1:
            pinned_path: str = next(iter(pinned_paths))
            if not cls._matches_suffix(pinned_path):
                return None
            if cls.validate_columns(pinned_path, feature_names) is False:
                return None
            return pinned_path
        valid_candidates: list[str] = [
            path
            for path in pinned_paths
            if cls._matches_suffix(path) and cls.validate_columns(path, feature_names) is not False
        ]
        if len(valid_candidates) == 1:
            return valid_candidates[0]
        raise ValueError(f"Features in batch are pinned to different files: {pinned_paths}")


def _collect_filtered_subclasses(cls: Any, parent_class: Any) -> list[type[BaseInputData]]:
    result = []
    for subclass in get_all_subclasses(cls):
        if not issubclass(subclass, parent_class):
            continue
        if subclass.supports_scoped_data_access():
            result.append(subclass)
    return result


def get_all_filtered_subclasses(cls: Any, parent_class: Any) -> list[type[BaseInputData]]:
    filtered_subclasses = _collect_filtered_subclasses(cls, parent_class)
    if not filtered_subclasses:
        auto_load_group = getattr(parent_class, "_auto_load_group", None)
        if auto_load_group is not None:
            from mloda.core.abstract_plugins.plugin_loader.plugin_loader import PluginLoader

            if auto_load_group not in PluginLoader._disabled_groups:
                PluginLoader().load_group(auto_load_group)
                filtered_subclasses = _collect_filtered_subclasses(cls, parent_class)
    return filtered_subclasses
