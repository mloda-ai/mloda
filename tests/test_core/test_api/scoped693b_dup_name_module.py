"""A second module holding a FeatureGroup whose class NAME collides with one in the test module.

Scoping by class-name string is the only way a user can disambiguate two same-named feature groups,
so the ambiguity error resolve_feature renders for them must itself distinguish them (name + module,
like the engine's format_feature_group_class). This module supplies the twin that lives elsewhere.
"""

from typing import Optional

from mloda.core.abstract_plugins.components.data_access_collection import DataAccessCollection
from mloda.core.abstract_plugins.components.feature import Feature
from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.core.abstract_plugins.compute_framework import ComputeFramework
from mloda.core.abstract_plugins.feature_group import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


DUP_NAME_FEATURE = "scoped_probe_693b_dup_name"


class Scoped693BDupNameProbe(FeatureGroup):
    """Same class name as the probe in the test module, different module."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame, PythonDictFramework}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Optional[DataAccessCollection] = None,
    ) -> bool:
        return str(feature_name) == DUP_NAME_FEATURE

    def input_features(self, options: Options, feature_name: FeatureName) -> Optional[set[Feature]]:
        return None
