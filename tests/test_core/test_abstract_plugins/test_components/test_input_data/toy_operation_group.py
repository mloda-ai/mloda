"""Toy operation group served by a Pandas and a PyArrow Implementation; import lazily inside tests."""

from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import OperationFeatureGroup, SubtypeDeclaration, property_spec
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

OPTOY_FEATURE = "optoy_value"
OPTOY_CHILD_FEATURE = "optoy_child_value"
OPTOY_BARE_FEATURE = "optoy_bare_value"
OPTOY_KEY = "optoy_kind"
OPTOY_PANDAS_OFF = "optoy_pandas_off"
PANDAS_MARK = 1
PYARROW_MARK = 2


class _NamedFeatureGroup(OperationFeatureGroup):
    FEATURE: ClassVar[str] = OPTOY_FEATURE

    SUBTYPES = SubtypeDeclaration(key=OPTOY_KEY)
    PROPERTY_MAPPING = {
        OPTOY_KEY: property_spec(
            "Toy operation kind.",
            strict=True,
            allowed_values={"sum": "Sum", "median": "Median"},
        ),
    }

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name) == cls.FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return None


class OpToyFG(_NamedFeatureGroup):
    """Operation group with a Pandas and a PyArrow implementation."""


class OpToyChildFG(OpToyFG):
    """Subclass group with no implementation of its own."""

    FEATURE = OPTOY_CHILD_FEATURE


class OpToyBareFG(_NamedFeatureGroup):
    """Operation group with no implementation."""

    FEATURE = OPTOY_BARE_FEATURE


class OpToyPandas(Implementation):
    feature_group = OpToyFG
    compute_framework = PandasDataFrame
    divergent = {"sum": "skips NaN in sum"}
    served_by: ClassVar[list[Any]] = []

    @classmethod
    def accepts(cls, feature_name: str, options: Options) -> bool:
        return options.get(OPTOY_PANDAS_OFF) is None

    @classmethod
    def calculate(cls, group: Any, data: Any, features: Any) -> Any:
        cls.served_by.append(group)
        return pd.DataFrame({OPTOY_FEATURE: [PANDAS_MARK], OPTOY_CHILD_FEATURE: [PANDAS_MARK]})


class OpToyPyArrow(Implementation):
    feature_group = OpToyFG
    compute_framework = PyArrowTable
    supported = frozenset({"sum"})
    served_by: ClassVar[list[Any]] = []

    @classmethod
    def calculate(cls, group: Any, data: Any, features: Any) -> Any:
        cls.served_by.append(group)
        return pa.table({OPTOY_FEATURE: [PYARROW_MARK], OPTOY_CHILD_FEATURE: [PYARROW_MARK]})
