"""
TestDataCreator classes for creating test data in different compute frameworks.
"""

from typing import Any, Dict, Optional, Set, Type, Union
import pandas as pd

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.input_data.creator.data_creator import DataCreator
from mloda_core.abstract_plugins.components.input_data.base_input_data import BaseInputData
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable


class ATestDataCreator(AbstractFeatureGroup):
    """
    Base class for creating test data in different compute frameworks.
    Subclasses should set the compute_framework class variable and implement
    any framework-specific data transformations.

    Default is PandasDataFrame.

    If needed, you can overwrite the conversion dictionary to add more
    compute frameworks and their conversion functions.
    """

    compute_framework: Type[ComputeFrameWork] = PandasDataFrame

    conversion = {
        PandasDataFrame: pd.DataFrame,
        PyArrowTable: None,
    }

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        """Return a DataCreator with the supported feature names."""
        return DataCreator(set(cls.get_raw_data().keys()))

    @classmethod
    def get_raw_data(cls) -> Dict[str, Any]:
        """Return the raw data as a dictionary."""
        return {}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raw_data = cls.get_raw_data()
        return cls.transform_format_for_testing(raw_data)

    @classmethod
    def transform_format_for_testing(cls, data: Dict[str, Any]) -> Any:
        """
        Transform the data to the appropriate format for the compute framework.
        """
        if cls.compute_framework in cls.conversion:
            conversion_func = cls.conversion[cls.compute_framework]
            if conversion_func is not None:
                return conversion_func(data)
            return data
        raise ValueError(f"Unsupported compute framework: {cls.compute_framework} for conversion in {cls.conversion}.")

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        """Return the compute framework for this data creator."""

        if issubclass(cls.compute_framework, ComputeFrameWork):
            return {cls.compute_framework}
        raise ValueError(f"compute_framework must be a subclass of ComputeFrameWork, not {cls.compute_framework}")
