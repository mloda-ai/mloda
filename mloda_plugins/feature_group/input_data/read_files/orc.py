from typing import Any, Tuple

from pyarrow import orc as pyarrow_orc

from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_plugins.feature_group.input_data.read_file import ReadFile


class OrcReader(ReadFile):
    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (
            ".orc",
            ".ORC",
        )

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        return pyarrow_orc.read_table(source=data_access, columns=list(features.get_all_names()))
