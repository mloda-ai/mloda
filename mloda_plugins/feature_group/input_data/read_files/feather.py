from typing import Any, Tuple

from pyarrow import feather as pyarrow_feather

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class FeatherReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".feather",
            ".feather",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pyarrow_feather.read_table(source=data_access, columns=list(features.get_all_names()))
