from mloda.core.optional_dependency import require
from mloda.provider import ReadDocumentFG
from mloda_plugins.feature_group.input_data.file_suffixes import YAML_SUFFIXES


class YamlFG(ReadDocumentFG):
    @classmethod
    def suffixes(cls) -> tuple[str, ...]:
        return YAML_SUFFIXES

    @classmethod
    def read_text(cls, path: str) -> str:
        yaml = require("yaml", "reading YAML documents")
        with open(path, "r", encoding="utf-8") as f:
            documents = list(yaml.safe_load_all(f))
        content = documents[0] if len(documents) == 1 else documents
        return str(yaml.dump(content))
