import os
from pathlib import Path
import pytest
import shutil

from mktestdocs import check_md_file


from typing import Set, Any
import time
from mloda.steward import ExtenderHook, Extender
from mloda.user import PluginLoader
import logging

logger = logging.getLogger(__name__)

# Load all plugins before running documentation tests
PluginLoader.all()


# We need this to test DokuExtender
class DokuExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        logger.error(f"Time taken: {time.time() - start}")
        return result


class DokuValidateInputFeatureExtender(Extender):
    def wraps(self) -> Set[ExtenderHook]:
        return {ExtenderHook.VALIDATE_INPUT_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken: {time.time() - start}")
        return result


@pytest.mark.parametrize("fpath", Path("docs").glob("**/*.md"), ids=str)
def test_files_good(fpath: Any) -> None:
    check_md_file(fpath=fpath, memory=True)


# Temporarily disabled - README being refactored
# def test_readme() -> None:
#     """
#     Test all Python code blocks in README.md.
#
#     This test uses mktestdocs to extract and execute all Python code blocks
#     from the README, ensuring all examples are correct and runnable.
#
#     Note: README examples use DataCreator (in-memory data generation),
#     so no external files are needed.
#     """
#     readme_path = Path("README.md")
#
#     if not readme_path.exists():
#         pytest.skip("README.md not found in repository root")
#
#     # Run mktestdocs on README with memory mode to avoid file pollution
#     check_md_file(fpath=readme_path, memory=True)
