import sys
import pytest
import subprocess  # nosec


CLI_ENTRY_POINT = "mloda"  # Entry point defined in pyproject.toml


def _is_mloda_installed() -> bool:
    """Checks if 'mloda' is installed using pip list."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=True)  # nosec
        return "mloda" in result.stdout.lower()  # Case-insensitive check
    except subprocess.CalledProcessError:
        return False  # pip list failed, assume mloda is not installed
    except FileNotFoundError:
        return False  # pip is not installed, assume mloda is not installed


@pytest.mark.skipif(
    not _is_mloda_installed(),
    reason="""Skipping mloda CLI test because mloda is not installed. 
            You can run this test either with pip install -e . or as tox -e installed.""",
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
