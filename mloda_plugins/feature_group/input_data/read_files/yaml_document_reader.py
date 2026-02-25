import yaml
from pathlib import Path
from typing import Any, Tuple

from mloda_plugins.feature_group.input_data.read_document import ReadDocument

from mloda.provider import FeatureSet


class YamlDocumentReader(ReadDocument):
    """Load entire YAML file as a single document value for RAG pipelines."""

    @classmethod
    def suffix(cls) -> Tuple[str, ...]:
        return (".yaml", ".yml")

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        file_path = features.get_options_key(cls.__name__)

        with open(file_path, "r", encoding="utf-8") as f:
            documents = list(yaml.safe_load_all(f))

        if len(documents) == 1:
            content = documents[0]
        else:
            content = documents

        yaml_string = yaml.dump(content)
        file_type = Path(file_path).suffix.lstrip(".")

        return [{cls.get_class_name(): yaml_string, "source": file_path, "file_type": file_type}]
