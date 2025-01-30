import os
from pathlib import Path
from typing import Any, List, Set


from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature import Feature
from mloda_core.abstract_plugins.components.feature_name import FeatureName
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.components.link import JoinType
from mloda_core.abstract_plugins.components.options import Options
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys
from mloda_plugins.feature_group.experimental.dynamic_feature_group_factory.dynamic_feature_group_factory import (
    DynamicFeatureGroupCreator,
)
from mloda_plugins.feature_group.experimental.source_input_feature import (
    SourceInputFeature,
    SourceTuple,
)
from mloda_plugins.feature_group.input_data.read_file_feature import ReadFileFeature
from mloda_plugins.feature_group.input_data.read_files.text_file_reader import PyFileReader


try:
    import pandas as pd
except ImportError:
    pd = None


class ConcatenatedFileContent(AbstractFeatureGroup):
    """
    A feature group that reads and combines content from Python files within a directory.

    It creates a set of features, each corresponding to a file,
    and joins them together for appending. It then concatenates all file
    contents into a single string.

    It uses dynamic feature group creation to create the single reader feature groups (a feature group per file).
    """

    # This feature should just be created once mlodaAPI run.
    join_feature_name = "FGConcatenatedFileContent_JoinLLMFiles"

    def input_features(self, options: Options, feature_name: FeatureName) -> Set[Feature] | None:
        target_folder = options.get("target_folder")
        if not target_folder:
            raise ValueError(f"The option 'target_folder' is required for {self.get_class_name()}.")

        disallowed_files = list(options.get("disallowed_files")) if options.get("disallowed_files") else ["__init__.py"]
        file_type = options.get("file_type") or "py"
        file_paths = find_file_paths(list(target_folder), file_type, not_allowed_files_names=disallowed_files)

        self._create_join_class(self.join_feature_name)
        return self._create_source_tuples(file_paths, feature_name)

    def _create_join_class(self, class_name: str) -> None:
        """Creates a Join class on the fly using DynamicFeatureGroupCreator."""

        def calculate_feature(cls: Any, data: Any, features: FeatureSet) -> Any:
            """Calculate feature for the dynamically created Join Class."""
            data[class_name] = data[data.columns]
            return data

        properties = {"calculate_feature": calculate_feature}

        DynamicFeatureGroupCreator.create(
            properties=properties, class_name=class_name, feature_group_cls=SourceInputFeature
        )

    def _create_source_tuples(self, file_paths: List[str], feature_name: FeatureName) -> Set[Feature]:
        """
        Creates the source tuples for reading the python files.

        Returns:
                 A set of Feature objects representing the input files.
        """
        set_source_tuples = set()
        short_f_list = [os.path.split(f)[-1] for f in file_paths]

        for cnt, f in enumerate(file_paths):
            short_f = os.path.split(f)[-1]
            right_link, left_link = None, None

            if cnt != len(file_paths) - 1:
                left_link = (ReadFileFeature, short_f_list[cnt])
                right_link = (ReadFileFeature, short_f_list[cnt + 1])

            source_tuple = SourceTuple(
                feature_name=short_f,
                source_class=PyFileReader.get_class_name(),  # type: ignore
                source_value=f,  # We use the file path, not the content.
                left_link=left_link,
                right_link=right_link,
                join_type=JoinType.APPEND,
                merge_index=short_f,
            )
            set_source_tuples.add(source_tuple)

        return {
            Feature(
                name=self.join_feature_name,
                options={DefaultOptionKeys.mloda_source_feature: frozenset(set_source_tuples)},
            )
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        if cls.join_feature_name not in data:
            raise ValueError(f"Feature {cls.join_feature_name} not found in the data.")

        combined_code = data[cls.join_feature_name].astype(str).str.cat(sep="\n\\nA new file begins here\\n")

        return pd.DataFrame({cls.get_class_name(): [combined_code]})


def find_file_paths(
    root_directory: List[Path | str], suffix: str, not_allowed_files_names: List[str] = []
) -> List[str]:
    file_paths = set()

    if not isinstance(root_directory, list):
        raise ValueError(f"Root directory must be a list. Got: {root_directory}")

    for root_dir in root_directory:
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        for file_path in root_dir.rglob(f"*.{suffix}"):  # Matches all files
            if file_path.name in not_allowed_files_names:
                continue

            file_paths.add(str(file_path.resolve()))

    if not file_paths:
        raise ValueError(f"No files found in the root directory: {root_directory}.")

    return list(file_paths)
