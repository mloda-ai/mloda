"""Toy operation group served by a Pandas and a PyArrow Implementation; import lazily inside tests."""

from typing import Any, ClassVar

import pandas as pd
import pyarrow as pa

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.implementation import Implementation
from mloda.provider import (
    BaseInputData,
    ComputeFramework,
    DataCreator,
    FeatureGroup,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)
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


OPTOY_CHAIN_SRC = "optoy_chain_src"
OPTOY_CHAIN_SUFFIX = "__optoy_chain"
CHAIN_LOG: list[tuple[str, str]] = []


class OpToyChainSourceFG(FeatureGroup):
    """Pandas-only source for the chain toy."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({OPTOY_CHAIN_SRC})

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]]:
        return {PandasDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({OPTOY_CHAIN_SRC: [1, 2, 3]})


class OpToyChainFG(OperationFeatureGroup):
    """Operation group whose feature `<src>__optoy_chain` takes `<src>` as input."""

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: DataAccessCollection | None = None,
    ) -> bool:
        return str(feature_name).endswith(OPTOY_CHAIN_SUFFIX)

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(str(feature_name)[: -len(OPTOY_CHAIN_SUFFIX)])}


class OpToyChainPandas(Implementation):
    feature_group = OpToyChainFG
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(cls, group: Any, data: Any, features: Any) -> Any:
        for name in sorted(features.get_all_names()):
            CHAIN_LOG.append((name, cls.__name__))
            data[name] = PANDAS_MARK
        return data


class OpToyChainPyArrow(Implementation):
    feature_group = OpToyChainFG
    compute_framework = PyArrowTable

    @classmethod
    def calculate(cls, group: Any, data: Any, features: Any) -> Any:
        for name in sorted(features.get_all_names()):
            CHAIN_LOG.append((name, cls.__name__))
            data = data.append_column(name, pa.array([PYARROW_MARK] * data.num_rows))
        return data
