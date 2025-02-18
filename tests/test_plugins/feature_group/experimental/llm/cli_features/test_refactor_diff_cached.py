import os

import pytest

from mloda_plugins.feature_group.experimental.llm.cli_features.refactor_git_cached import (
    DiffCachedFeatureGroup,
    DiffFeatureGroup,
    RunRefactorDiffCached,
)


class TestRunRefactorDiffCached:
    @pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
    def test_get_tool_output_by_feature_group_(self) -> None:
        obj = RunRefactorDiffCached()
        assert "</git_diff_cached>" in obj.get_tool_output_by_feature_group_(DiffCachedFeatureGroup)
        assert "</git_diff>" in obj.get_tool_output_by_feature_group_(DiffFeatureGroup)


@pytest.mark.skipif(os.environ.get("GEMINI_API_KEY") is None, reason="GEMINI KEY NOT SET")
def test_refactor_diff_cached_run() -> None:
    obj = RunRefactorDiffCached()
    obj.run()
