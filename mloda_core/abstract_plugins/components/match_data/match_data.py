from typing import Any, Optional

from mloda_core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda_core.abstract_plugins.components.options import Options


import logging

logger = logging.getLogger(__name__)


class MatchData:
    @classmethod
    def matches(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        """
        We look if feature scope data access or global scope access is set.

        Feature scope access are set via options per feature,
        whereas global scope access is set via data_access_collection.
        """
        if cls.feature_scope_data_access(options, feature_name) is True:
            return True

        if cls.global_scope_data_access(feature_name, options, data_access_collection) is True:
            return True
        return False

    @classmethod
    def feature_scope_data_access(cls, options: Options, feature_name: str) -> bool:
        """
        We check for the feature scope data access if any options match directly the framework connection and matching logic.
        """

        cls_name = cls.get_class_name()

        if not options.get(cls_name):
            return False

        framework_connection_object = options.get(cls_name)

        matched_data_access = cls.match_data_access(feature_name, options, None, framework_connection_object)
        if matched_data_access:
            return True
        return False

    @classmethod
    def global_scope_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection],
    ) -> bool:
        """
        We check for global scope data access if any data access collection matches the framework connection and matching logic."""

        if data_access_collection is None:
            return False

        matched_data_access = cls.match_data_access(feature_name, options, data_access_collection, None)
        if matched_data_access is False:
            return False

        # We need to add the matched data access to the options of the feature. This way it is linked throughout mloda.
        cls.add_base_input_data_to_options(matched_data_access, options)
        return True

    @classmethod
    def match_data_access(
        cls,
        feature_name: str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
        framework_connection_object: Optional[Any] = None,
    ) -> Any:
        """
        We check for data access collection if any child classes match the data access.
        """
        raise NotImplementedError()

    @classmethod
    def add_base_input_data_to_options(cls, matched_data_access: Any, options: Options) -> None:
        """
        Adding the found data access class to the options.
        """

        cls_name = cls.get_class_name()

        if options.get(cls_name):
            existing_data = options.get(cls_name)
            if existing_data == matched_data_access:
                return

            raise ValueError(f"{cls_name} already set with different values. {existing_data} != {matched_data_access}")
        options.add(cls_name, matched_data_access)

    @classmethod
    def get_class_name(cls) -> str:
        """
        Returns the name of the class.
        """
        return cls.__name__
