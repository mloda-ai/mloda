import os
from typing import Any, Dict, List, Set
import logging

from mloda_core.abstract_plugins.abstract_feature_group import AbstractFeatureGroup
from mloda_core.abstract_plugins.components.feature_set import FeatureSet

logger = logging.getLogger(__name__)


class ListDirectoryFeatureGroup(AbstractFeatureGroup):
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
                if not cls._is_ignored(file_path, ignore_patterns):
                    current_level[f] = None  # Files are stored as None in the structure

        # Generate formatted tree string
        tree_string = cls._generate_tree_string(file_structure)
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
    def _generate_tree_string(cls, file_structure: Dict[str, Any], prefix: str = "") -> str:
        """Recursively generates a properly formatted tree-like string representation."""
        lines: List[str] = []
        items = list(file_structure.items())
        for index, (name, content) in enumerate(items):
            is_last = index == len(items) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if isinstance(content, dict) and content:
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                lines.append(cls._generate_tree_string(content, new_prefix))
        return "\n".join(lines)  # Ensure the whole structure is kept as one string
