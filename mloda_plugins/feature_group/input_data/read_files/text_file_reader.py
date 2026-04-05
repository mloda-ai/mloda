from typing import Any

from mloda_plugins.feature_group.input_data.read_document import ReadDocument

from mloda.provider import FeatureSet


class TextFileReader(ReadDocument):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".text",)

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        file_path = features.get_options_key(cls.__name__)

        with open(file_path, encoding="utf-8") as file:
            content = file.read()

        file_type = cls.suffix()[0].lstrip(".")

        return [{cls.get_class_name(): content, "source": file_path, "file_type": file_type}]


class PyFileReader(TextFileReader):
    @classmethod
    def suffix(cls) -> tuple[str, ...]:
        return (".py",)
