import os
from typing import Any, Dict, List, Set, Type, Union
import logging

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet
from mloda_core.abstract_plugins.compute_frame_work import ComputeFrameWork
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataframe

logger = logging.getLogger(__name__)


class ListDirectoryFeatureGroup(AbstractFeatureGroup):
    """
    A Feature Group that generates a string representation of a directory's file structure.

    Purpose:
    This class crawls a given directory (defaults to the project root), applying .gitignore rules
    to filter out unwanted files and directories. It then creates a tree-like string
    that represents the directory structure, including full paths, file types, and depth.
    This representation can be used as input for LLMs to understand the codebase's organization.

    Key Features:
    - Ignores hidden directories (starting with ".") and __pycache__.
    - Applies .gitignore patterns for flexible file and directory exclusion.
    - Generates a string representation of the directory tree including full paths.
    - Identifies each item as either a file or a directory.
    - Provides the depth of each item within the directory structure.

    Example:

    ```
    project_root/
    ├── file1.txt (file) (depth: 1)
    ├── dir1 (dir) (depth: 1)
    │   └── file2.py (file) (depth: 2)
    └── ...
    ```
    """

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        project_root = os.getcwd()  # Get project root (assumed to be CWD)
        file_structure: Dict[str, Any] = {}

        # Load ignore patterns from .gitignore
        ignore_patterns = cls._load_gitignore_patterns(project_root)

        for root, dirs, files in os.walk(project_root):
            # Get relative path from project root
            relative_root = os.path.relpath(root, project_root)
            if relative_root == ".":
                relative_root = ""  # Keep root directory clean in listing

            # Fully exclude hidden directories (starting with ".") and __pycache__
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d != "__pycache__"
                and not cls._is_ignored(os.path.join(relative_root, d), ignore_patterns)
            ]

            # Initialize dictionary structure
            current_level = file_structure
            for part in relative_root.split(os.sep):
                if part:
                    current_level = current_level.setdefault(part, {})

            # Collect allowed directories
            for d in dirs:
                current_level[d] = {}

            # Filter files based on .gitignore
            for f in files:
                file_path = os.path.join(relative_root, f)
                if "__init__.py" in file_path:
                    continue
                if not cls._is_ignored(file_path, ignore_patterns):
                    current_level[f] = None  # Files are stored as None in the structure

        # Generate formatted tree string
        tree_string = cls._generate_full_path_tree_string(file_structure, project_root)
        return {cls.get_class_name(): [tree_string]}  # Ensuring the entire tree is a single string inside a list

    @staticmethod
    def _load_gitignore_patterns(project_root: str) -> Set[str]:
        """Reads and processes the .gitignore file (basic pattern matching)."""
        gitignore_path = os.path.join(project_root, ".gitignore")
        ignore_patterns: Set[str] = set()

        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Ignore empty lines and comments
                        ignore_patterns.add(line)

        return ignore_patterns

    @staticmethod
    def _is_ignored(file_path: str, ignore_patterns: Set[str]) -> bool:
        """Checks if a file/directory should be ignored based on .gitignore patterns."""
        for pattern in ignore_patterns:
            if pattern.endswith("/"):  # Directory exclusion
                if file_path.startswith(pattern.rstrip("/")):
                    return True
            elif "*" in pattern:  # Basic wildcard support (e.g., *.log)
                if file_path.endswith(pattern.lstrip("*")):
                    return True
            elif file_path == pattern:  # Exact file/directory match
                return True
        return False

    @classmethod
    def _generate_full_path_tree_string(
        cls, file_structure: Dict[str, Any], project_root: str, prefix: str = "", current_path: str = "", depth: int = 0
    ) -> str:
        """Recursively generates a full path tree string with file type indicators and depth."""
        lines: List[str] = []
        items = sorted(file_structure.items())  # Sort items alphabetically

        for index, (name, content) in enumerate(items):
            is_last = index == len(items) - 1
            connector = "└── " if is_last else "├── "

            new_path = os.path.join(current_path, name)  # use os.path.join here to construct new path
            if current_path == "":
                new_path = name  # at the root directory, there is no sub directories so don't join,
            file_type = "(dir)" if isinstance(content, dict) else "(file)"
            lines.append(f"{prefix}{connector}{project_root}/{new_path} {file_type} (depth: {depth})")

            if isinstance(content, dict) and content:
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                lines.append(
                    cls._generate_full_path_tree_string(content, project_root, new_prefix, new_path, depth + 1)
                )

        return "\n".join(lines)

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFrameWork]]]:
        return {PandasDataframe}
