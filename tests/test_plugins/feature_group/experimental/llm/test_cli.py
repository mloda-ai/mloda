import os
import pytest
import subprocess  # nosec


CLI_ENTRY_POINT = "mloda"  # Entry point defined in pyproject.toml


@pytest.mark.skipif(
    not os.environ.get("MLODA_INSTALLED_TEST"),
    reason="CLI test only runs in the installed testenv (tox -e installed).",
)
@pytest.mark.parametrize("feature_group", ["InstalledPackagesFeatureGroup"])
def test_mloda_entry_point_execution(feature_group: str) -> None:
    """Tests that the CLI entry point executes without errors."""
    result = subprocess.run(  # nosec
        [CLI_ENTRY_POINT, feature_group],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"CLI script failed with error:\n{result.stderr}"
    assert "annotated-types" in result.stdout
